"""
Speed benchmark script for dual model training: Shampoo vs Muon
Trains two CifarNet models simultaneously and tracks singular values of weight matrices.

This script:
- Trains one CifarNet model with Shampoo optimizer
- Trains one CifarNet model with Muon optimizer
- Records histogram of singular values at specified intervals for both models
- Compares their performance side-by-side
- Uses the architecture from shmapooV2.py (CifarNet with whitening layer)

Example usage:
    python speed_cifar10.py --batch-size 2000 --svd-freq 20
    python speed_cifar10.py --batch-sweep-count 300 --use-augmentation
"""

import os
import sys
import uuid
from math import ceil
from typing import Union
import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import tyro
from tqdm import trange

from src.architectures import CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet
from src.utils import convert_dataclasses

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = True

#############################################
#        Shampoo Optimizer (from shmapooV2) #
#############################################
@torch.no_grad()
def _matrix_power(
    matrix: torch.Tensor,
    p: int,
    num_iters: int = 60,
    ridge_epsilon: float = 1e-6,
    error_tolerance: float = 1e-6,
):
    """
    Compute matrix^(-1/p) for symmetric PSD `matrix` using the coupled-Newton iteration.
    Runs on CPU in float32, returns in the input dtype.
    """
    assert matrix.ndim == 2 and matrix.size(0) == matrix.size(1), "matrix must be square"
    assert p > 0 and isinstance(p, int), "p must be a positive integer"

    # Work on CPU float32 for stability; remember original dtype/device to cast back.
    device_in, dtype_in = matrix.device, matrix.dtype
    A = matrix.to(torch.float32).cpu()

    n = A.size(0)
    I = torch.eye(n, dtype=torch.float32)

    # Helper: spectral norm (largest singular value) for scaling
    def spectral_norm(m: torch.Tensor) -> float:
        # power iteration for symmetric PSD (largest eigenvalue)
        v = torch.randn(n, 1, dtype=torch.float32)
        v /= v.norm() + 1e-12
        last = 0.0
        for _ in range(100):
            v = m @ v
            nv = v.norm().item()
            if nv < 1e-30:
                return 0.0
            v /= nv
            # Rayleigh quotient as estimate
            lam = torch.dot(v.flatten(), (m @ v).flatten()).item()
            if abs(lam - last) <= 1e-6 * max(1.0, abs(last)):
                break
            last = lam
        return max(lam, 0.0)

    # Scale ridge by max eigenvalue (Optax style)
    max_ev = spectral_norm(A)
    scaled_ridge = ridge_epsilon * max(max_ev, 1e-16)

    # 1x1 shortcut
    if n == 1:
        out = (A + scaled_ridge).pow(-1.0 / p)
        return out.to(device=device_in, dtype=dtype_in)

    # Damped matrix
    Ad = A + scaled_ridge * I

    # z scaling (Higham/Iannazzo/Optax): z = (1+p) / (2 * ||Ad||_2)
    # use spectral norm as ||·||_2
    z = (1.0 + p) / max(2.0 * spectral_norm(Ad), 1e-16)

    # Initialize iteration state
    alpha = -1.0 / p
    M = Ad * z                  # mat_m
    H = I * (z ** (1.0 / p))    # mat_h

    # Error = max|M - I|
    def max_abs(x: torch.Tensor) -> float:
        return x.abs().max().item()

    err = max_abs(M - I)
    i = 0
    run_step = True
    prev_err = err
    prev_H = H.clone()

    # Fast integer power by repeated squaring
    def mat_power(mat: torch.Tensor, k: int) -> torch.Tensor:
        assert k >= 1
        result = I.clone()
        base = mat
        e = k
        while e > 0:
            if (e & 1) == 1:
                result = result @ base
            e >>= 1
            if e:
                base = base @ base
        return result

    # Coupled Newton iterations
    while (i < num_iters) and (err > error_tolerance) and run_step:
        M_i = (1.0 - alpha) * I + alpha * M
        M_next = mat_power(M_i, p) @ M
        H_next = H @ M_i

        new_err = max_abs(M_next - I)

        # Bound error growth: allow at most 1.2× increase
        run_step = new_err < (prev_err * 1.2 + 1e-12)

        # Accept step
        i += 1
        prev_H = H.clone()
        prev_err = err
        M, H, err = M_next, H_next, new_err

    # If the last step overshot (run_step False), revert to previous H
    H_final = H if run_step else prev_H

    return H_final.to(device=device_in)




def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class Shampoo(torch.optim.Optimizer):
    r"""Implements Shampoo Optimizer Algorithm.

    It has been proposed in `Shampoo: Preconditioned Stochastic Tensor
    Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)

    __ https://arxiv.org/abs/1802.09568

    Note:
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
        order_multiplier: int = 2,
        nesterov: bool = True,
    ):

        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if epsilon < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if update_freq < 1:
            raise ValueError('Invalid momentum value: {}'.format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            nesterov=nesterov,
        )
        self.order_multiplier = order_multiplier
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure = None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # precondition matrices
                        state['precond_{}'.format(dim_id)] = group[
                            'epsilon'
                        ] * torch.eye(dim, device=grad.device, dtype=torch.float32)
                        state[
                            'inv_precond_{dim_id}'.format(dim_id=dim_id)
                        ] = torch.zeros(dim, dim, device=grad.device, dtype=torch.float32)

                if momentum > 0:
                    state['momentum_buffer'].lerp_(grad, 1-momentum)

                update = grad.lerp(state['momentum_buffer'], momentum) if group['nesterov'] else state['momentum_buffer']
                update = update*(1/(1-momentum**(state['step']+2))) if group['nesterov'] else update*(1/(1-momentum**(state['step']+1)))
                update32 = update.to(torch.float32)
                # if weight_decay > 0:
                #     grad.add_(p.data, alpha=group['weight_decay'])

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state['precond_{}'.format(dim_id)]
                    inv_precond = state['inv_precond_{}'.format(dim_id)]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    update32 = update32.transpose_(0, dim_id).contiguous()
                    update32 = update32.view(dim, -1)


                    g32 = grad.to(torch.float32)
                    g32_t = g32.t()
                    precond.add_(g32 @ g32_t)
                    if state['step'] % group['update_freq'] == 0:
                        inv_precond.copy_(_matrix_power(precond,  (order*self.order_multiplier)))

                    if dim_id == order - 1:
                        # finally
                        update32 = update32.t() @ inv_precond
                        # grad: (-1, last_dim)
                        grad = update32.view(original_size).to(grad.dtype)
                    else:
                        # if not final
                        update32 = inv_precond @ update32
                        # grad (dim, -1)
                        update32 = update32.view(transposed_size).to(grad.dtype)

                state['step'] += 1
                # state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-group['lr'])

        return loss


class Adam(torch.optim.Optimizer):
    def __init__(self, param_groups, lr, betas, eps, weight_decay):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = 0
                state["step"] += 1
                update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                        state["step"], group["betas"], group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

        return loss


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']  
                # buf.mul_(momentum).add_(g)
                # g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                buf.lerp_(g, 1 - momentum)
                update = g.lerp_(buf, momentum) if group['nesterov'] else buf
                update = update*(1/(1-momentum**(state['step']+2))) if group['nesterov'] else update*(1/(1-momentum**(state['step']+1)))
                # p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = zeropower_via_newtonschulz5(update.reshape(len(g), -1)).view(g.shape) # whiten the update
                update *= max(1, g.size(-2) / g.size(-1))**0.5
                state['step'] += 1
                p.data.mul_(1-lr*group['weight_decay'])
                p.data.add_(update, alpha=-lr) # take a step

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device('cuda'))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################

# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding='same', bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths['block1']),
            ConvGroup(widths['block1'], widths['block2']),
            ConvGroup(widths['block2'], widths['block3']),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths['block3'], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


#############################################
#          Singular Value Tracking          #
#############################################

@torch.no_grad()
def compute_singular_values(model, param_names=None):
    """
    Compute singular values for all 2D+ parameters in the model.
    
    Args:
        model: PyTorch model
        param_names: Optional list of parameter names to track. If None, track all 2D+ params.
    
    Returns:
        Dictionary mapping parameter names to their singular values (as numpy arrays)
    """
    singular_values = {}
    
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        if param_names is not None and name not in param_names:
            continue
            
        # Reshape to 2D if needed
        if param.ndim == 2:
            matrix = param.data
        else:
            # For higher dimensional tensors, reshape to 2D
            matrix = param.data.reshape(param.size(0), -1)
        
        # Compute SVD (only singular values)
        try:
            # Move to CPU for SVD computation
            matrix_cpu = matrix.cpu().float()
            _, s, _ = torch.svd(matrix_cpu)
            singular_values[name] = s.numpy()
        except Exception as e:
            print(f"Warning: Could not compute SVD for {name}: {e}")
            continue
    
    return singular_values


#############################################
#                 Logging                  #
#############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))


logging_columns_list = ['epoch', 'opt', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'lr']


def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.6f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


#############################################
#               Evaluation                 #
#############################################

def evaluate(model, loader):
    """Evaluate model on loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        test_images = loader.normalize(loader.images)
        # Process in batches to avoid memory issues
        batch_size = 2000
        for i in range(0, len(test_images), batch_size):
            inputs = test_images[i:i+batch_size]
            labels = loader.labels[i:i+batch_size]
            
            outputs = model(inputs)
            
            # Compute loss
            loss = F.cross_entropy(outputs, labels, reduction='sum')
            total_loss += loss.item()
            
            # Compute accuracy
            total_correct += (outputs.argmax(1) == labels).float().sum().item()
            total_samples += len(inputs)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


#############################################
#            Main Training Loop             #
#############################################

ValidArch = Union[CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet]


def main(
    data_path: str = "cifar10",       # path to store CIFAR10 data
    batch_size: int = 2000,           # batch size for training
    lr_bias: float = 0.053,           # learning rate for biases
    lr_filters_shampoo: float = 0.24, # learning rate for filter params (Shampoo)
    lr_filters_muon: float = 0.02,    # learning rate for filter params (Muon)
    lr_head: float = 0.67,            # learning rate for head/output layer
    momentum_sgd: float = 0.85,       # momentum for SGD optimizer
    momentum_shampoo: float = 0.9,    # momentum for Shampoo optimizer 
    shampoo_order: int = 2,           # order for Shampoo optimizer
    momentum_muon: float = 0.9,       # momentum for Muon optimizer
    weight_decay: float = 1e-4,       # weight decay
    use_augmentation: bool = True,    # whether to use data augmentation
    label_smoothing: float = 0.2,     # label smoothing parameter
    device: str = "cuda",             # cuda or cpu
    seed: int = 0,                    # random seed
    save_results: bool = True,        # whether to save results
    svd_freq: int = 20,               # how often to record singular values (in steps)
    batch_sweep_count: int = 300,     # number of batches to process (affects total steps)
):
    print("=" * 80)
    print("Dual Model Training: Shampoo vs Muon")
    print("=" * 80)
    
    # Read code for saving
    with open(sys.argv[0]) as f:
        code = f.read()
    
    # collect configs that were passed in
    # Filter out file handles and other unpicklable objects
    config = convert_dataclasses({k: v for k, v in locals().items() 
                                   if k not in ['f', 'code']})
    config["cmd"] = " ".join(sys.argv)
    
    # set random seed
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    # load the dataset using CifarLoader
    print("\n[1/4] Loading Data...")
    aug = dict(flip=True, translate=2) if use_augmentation else {}
    train_loader = CifarLoader(data_path, train=True, batch_size=batch_size, aug=aug)
    test_loader = CifarLoader(data_path, train=False, batch_size=2000)
    total_train_steps = ceil(batch_sweep_count * len(train_loader))
    total_epochs = ceil(total_train_steps / len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    print(f"  - Training samples: {len(train_loader.images)}")
    print(f"  - Test samples: {len(test_loader.images)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Total steps: {total_train_steps}")
    print(f"  - Total epochs: {total_epochs}")
    
    # instantiate TWO models (one for each optimizer)
    print("\n[2/4] Creating Models...")
    model_shampoo = CifarNet().to(device).to(memory_format=torch.channels_last)
    model_muon = CifarNet().to(device).to(memory_format=torch.channels_last)
    
    # Initialize both models with the same weights
    model_shampoo.reset()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model_shampoo.init_whiten(train_images)
    
    # Copy initialization to muon model
    model_muon.load_state_dict(model_shampoo.state_dict())
    
    print(f"  - Architecture: CifarNet")
    print(f"  - Total parameters: {sum(p.numel() for p in model_shampoo.parameters()):,}")
    
    #############################################
    # Setup optimizers for SHAMPOO model
    #############################################
    print("\n[3/4] Setting up Optimizers...")
    
    # Shampoo model optimizers
    filter_params_shampoo = [p for p in model_shampoo.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases_shampoo = [p for n, p in model_shampoo.named_parameters() if 'norm' in n and p.requires_grad]
    
    param_configs_sgd_shampoo = [
        dict(params=[model_shampoo.whiten.bias], lr=lr_bias, weight_decay=weight_decay/lr_bias),
        dict(params=norm_biases_shampoo, lr=lr_bias, weight_decay=weight_decay/lr_bias),
        dict(params=[model_shampoo.head.weight], lr=lr_head, weight_decay=weight_decay/lr_head)
    ]
    
    optimizer_sgd_shampoo = torch.optim.SGD(param_configs_sgd_shampoo, momentum=momentum_sgd, nesterov=True)
    optimizer_shampoo = Shampoo(filter_params_shampoo, lr=lr_filters_shampoo, momentum=momentum_shampoo, 
                                weight_decay=weight_decay, order_multiplier=shampoo_order)
    
    optimizers_shampoo = [optimizer_sgd_shampoo, optimizer_shampoo]
    
    # Muon model optimizers
    filter_params_muon = [p for p in model_muon.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases_muon = [p for n, p in model_muon.named_parameters() if 'norm' in n and p.requires_grad]
    
    param_configs_sgd_muon = [
        dict(params=[model_muon.whiten.bias], lr=lr_bias, weight_decay=weight_decay/lr_bias),
        dict(params=norm_biases_muon, lr=lr_bias, weight_decay=weight_decay/lr_bias),
        dict(params=[model_muon.head.weight], lr=lr_head, weight_decay=weight_decay/lr_head)
    ]
    
    optimizer_sgd_muon = torch.optim.SGD(param_configs_sgd_muon, momentum=momentum_sgd, nesterov=True)
    optimizer_muon = Muon(filter_params_muon, lr=lr_filters_muon, momentum=momentum_muon, 
                          nesterov=True, weight_decay=weight_decay)
    
    optimizers_muon = [optimizer_sgd_muon, optimizer_muon]
    
    # Set initial learning rates
    for opt in optimizers_shampoo + optimizers_muon:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    print(f"  - Shampoo model: {len(filter_params_shampoo)} filter params")
    print(f"  - Muon model: {len(filter_params_muon)} filter params")
    
    # Get parameter names for SVD tracking (only filter parameters - 4D conv weights)
    filter_param_names_shampoo = [n for n, p in model_shampoo.named_parameters() 
                                  if len(p.shape) == 4 and p.requires_grad]
    filter_param_names_muon = [n for n, p in model_muon.named_parameters() 
                               if len(p.shape) == 4 and p.requires_grad]
    
    # Storage for logging
    RUN_LOGS_SHAMPOO = []
    RUN_LOGS_MUON = []
    SINGULAR_VALUES_SHAMPOO = []  # List of (step, sv_dict)
    SINGULAR_VALUES_MUON = []     # List of (step, sv_dict)
    
    # Print header
    print("\n[4/4] Training...")
    print_columns(logging_columns_list, is_head=True)
    
    step = 0
    #############################################
    # Training loop
    #############################################
    
    for epoch in range(total_epochs):
        model_shampoo.train()
        model_muon.train()
        
        # Metrics for Shampoo model
        epoch_loss_shampoo = 0.0
        epoch_correct_shampoo = 0
        epoch_samples_shampoo = 0
        
        # Metrics for Muon model
        epoch_loss_muon = 0.0
        epoch_correct_muon = 0
        epoch_samples_muon = 0
        
        # Training
        for inputs, labels in train_loader:
            #############################################
            # Train Shampoo model
            #############################################
            outputs_shampoo = model_shampoo(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss_shampoo = F.cross_entropy(outputs_shampoo, labels, 
                                          label_smoothing=label_smoothing, reduction='sum')
            loss_shampoo.backward()
            
            # Update learning rate
            # Adjust whiten bias LR for first group
            for group in optimizer_sgd_shampoo.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            # Adjust other groups
            for group in optimizer_sgd_shampoo.param_groups[1:] + optimizer_shampoo.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            
            # Optimizer step
            for opt in optimizers_shampoo:
                opt.step()
            
            # Zero gradients
            model_shampoo.zero_grad(set_to_none=True)
            
            # Track metrics
            with torch.no_grad():
                epoch_loss_shampoo += loss_shampoo.item()
                epoch_correct_shampoo += (outputs_shampoo.argmax(1) == labels).float().sum().item()
                epoch_samples_shampoo += len(inputs)
            
            #############################################
            # Train Muon model
            #############################################
            outputs_muon = model_muon(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss_muon = F.cross_entropy(outputs_muon, labels, 
                                       label_smoothing=label_smoothing, reduction='sum')
            loss_muon.backward()
            
            # Update learning rate
            # Adjust whiten bias LR for first group
            for group in optimizer_sgd_muon.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            # Adjust other groups
            for group in optimizer_sgd_muon.param_groups[1:] + optimizer_muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            
            # Optimizer step
            for opt in optimizers_muon:
                opt.step()
            
            # Zero gradients
            model_muon.zero_grad(set_to_none=True)
            
            # Track metrics
            with torch.no_grad():
                epoch_loss_muon += loss_muon.item()
                epoch_correct_muon += (outputs_muon.argmax(1) == labels).float().sum().item()
                epoch_samples_muon += len(inputs)
            
            #############################################
            # Record singular values every svd_freq steps
            #############################################
            if step % svd_freq == 0:
                print(f"\n  [Step {step}] Computing singular values...")
                sv_shampoo = compute_singular_values(model_shampoo, filter_param_names_shampoo)
                sv_muon = compute_singular_values(model_muon, filter_param_names_muon)
                SINGULAR_VALUES_SHAMPOO.append((step, sv_shampoo))
                SINGULAR_VALUES_MUON.append((step, sv_muon))
                print(f"  [Step {step}] Recorded SVD for {len(sv_shampoo)} layers (Shampoo) and {len(sv_muon)} layers (Muon)")
            
            step += 1
            if step >= total_train_steps:
                break
        
        # Compute training metrics for Shampoo
        train_loss_shampoo = epoch_loss_shampoo / epoch_samples_shampoo
        train_acc_shampoo = epoch_correct_shampoo / epoch_samples_shampoo
        
        # Compute training metrics for Muon
        train_loss_muon = epoch_loss_muon / epoch_samples_muon
        train_acc_muon = epoch_correct_muon / epoch_samples_muon
        
        # Evaluate on test set
        test_loss_shampoo, test_acc_shampoo = evaluate(model_shampoo, test_loader)
        test_loss_muon, test_acc_muon = evaluate(model_muon, test_loader)
        
        # Get current learning rate
        current_lr_shampoo = optimizers_shampoo[0].param_groups[0]['lr'] if optimizers_shampoo else 0.0
        current_lr_muon = optimizers_muon[0].param_groups[0]['lr'] if optimizers_muon else 0.0
        
        # Log Shampoo
        log_dict_shampoo = {
            'epoch': epoch,
            'opt': 'Shampoo',
            'train_loss': train_loss_shampoo,
            'train_acc': train_acc_shampoo,
            'test_loss': test_loss_shampoo,
            'test_acc': test_acc_shampoo,
            'lr': current_lr_shampoo,
        }
        print_training_details(log_dict_shampoo, is_final_entry=False)
        RUN_LOGS_SHAMPOO.append([epoch, train_loss_shampoo, train_acc_shampoo, 
                                 test_loss_shampoo, test_acc_shampoo, current_lr_shampoo])
        
        # Log Muon
        log_dict_muon = {
            'epoch': epoch,
            'opt': 'Muon',
            'train_loss': train_loss_muon,
            'train_acc': train_acc_muon,
            'test_loss': test_loss_muon,
            'test_acc': test_acc_muon,
            'lr': current_lr_muon,
        }
        print_training_details(log_dict_muon, is_final_entry=(epoch == total_epochs - 1))
        RUN_LOGS_MUON.append([epoch, train_loss_muon, train_acc_muon, 
                              test_loss_muon, test_acc_muon, current_lr_muon])
        
        if step >= total_train_steps:
            break
    
    #############################################
    # Save results
    #############################################
    
    if save_results:
        print("\n" + "=" * 80)
        print("Saving Results...")
        log_dir = os.path.join('logs', 'dual_training_' + str(uuid.uuid4()))
        os.makedirs(log_dir, exist_ok=True)
        
        # Save configuration and code
        log_path = os.path.join(log_dir, 'config.pt')
        torch.save(dict(
            code=code, 
            config=config, 
            test_acc_shampoo=test_acc_shampoo,
            test_acc_muon=test_acc_muon,
        ), log_path)
        
        # Save metrics
        np.save(
            os.path.join(log_dir, "metrics_shampoo.npy"),
            np.array(RUN_LOGS_SHAMPOO, dtype=object),
            allow_pickle=True
        )
        np.save(
            os.path.join(log_dir, "metrics_muon.npy"),
            np.array(RUN_LOGS_MUON, dtype=object),
            allow_pickle=True
        )
        
        # Save singular values
        with open(os.path.join(log_dir, "singular_values_shampoo.pkl"), 'wb') as f:
            pickle.dump(SINGULAR_VALUES_SHAMPOO, f)
        with open(os.path.join(log_dir, "singular_values_muon.pkl"), 'wb') as f:
            pickle.dump(SINGULAR_VALUES_MUON, f)
        
        # Save models
        torch.save(model_shampoo.state_dict(), os.path.join(log_dir, "model_shampoo.pt"))
        torch.save(model_muon.state_dict(), os.path.join(log_dir, "model_muon.pt"))
        
        print(f"  - Results saved to: {os.path.abspath(log_dir)}")
        print(f"  - Config: config.pt")
        print(f"  - Metrics: metrics_shampoo.npy, metrics_muon.npy")
        print(f"  - Singular values: singular_values_shampoo.pkl, singular_values_muon.pkl")
        print(f"  - Models: model_shampoo.pt, model_muon.pt")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final Test Accuracy (Shampoo): {test_acc_shampoo:.4f}")
    print(f"Final Test Accuracy (Muon):    {test_acc_muon:.4f}")
    print(f"Difference (Shampoo - Muon):   {test_acc_shampoo - test_acc_muon:+.4f}")
    print("=" * 80)
    
    return {
        'test_acc_shampoo': test_acc_shampoo,
        'test_acc_muon': test_acc_muon,
        'singular_values_shampoo': SINGULAR_VALUES_SHAMPOO,
        'singular_values_muon': SINGULAR_VALUES_MUON,
    }


if __name__ == "__main__":
    args = tyro.cli(main, config=[tyro.conf.ConsolidateSubcommandArgs])

