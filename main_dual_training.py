"""
Main training script for multi-model training with configurable optimizers.
Trains multiple models simultaneously with different optimizers and tracks metrics.

This script:
- Trains multiple models with user-specified optimizers
- Records metrics and singular values
- Uses the dual_training_loggers infrastructure
- Compares their performance side-by-side

Example usage:
    python main_dual_training.py --arch mlp
    python main_dual_training.py --arch cnn --use-augmentation
"""

import os
import sys
import uuid
from math import ceil
from typing import Union, List, Dict, Any
from dataclasses import dataclass
import pickle
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import tyro
from tqdm import trange

from src.architectures import CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet, CifarNet
from src.advanced_optimizers import Shampoo, Muon, Adam
from src.utils import convert_dataclasses
from src.dual_training_loggers import create_default_loggers, LossAndAccuracyLogger

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
#          Optimizer Configuration          #
#############################################

@dataclass
class MuonConfig:
    """Configuration for Muon optimizer."""
    lr: float = 0.0005
    momentum: float = 0.9
    
    def __str__(self):
        return f"Muon_lr{self.lr}_mom{self.momentum}"

@dataclass  
class ShampooConfig:
    """Configuration for Shampoo optimizer."""
    lr: float = 0.0005
    momentum: float = 0.9
    order_multiplier: int = 2
    
    def __str__(self):
        return f"Shampoo_lr{self.lr}_mom{self.momentum}_order{self.order_multiplier}"

@dataclass
class AdamConfig:
    """Configuration for Adam optimizer (used for biases/heads)."""
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    
    def __str__(self):
        return f"Adam_lr{self.lr}_b1{self.beta1}_b2{self.beta2}"


OptimizerConfig = Union[MuonConfig, ShampooConfig, AdamConfig]


def parse_optimizer_config(config_str: str) -> OptimizerConfig:
    """
    Parse optimizer config from CLI string format.
    
    Format: "OptimizerType:param1=value1,param2=value2,..."
    
    Examples:
        "Muon:lr=0.0005,momentum=0.9"
        "Shampoo:lr=0.0005,momentum=0.9,order_multiplier=2"
        "Adam:lr=0.01,beta1=0.9,beta2=0.95"
    
    Args:
        config_str: String specification of optimizer config
        
    Returns:
        OptimizerConfig instance
    """
    if ':' not in config_str:
        raise ValueError(f"Config string must contain ':' separator. Got: {config_str}")
    
    parts = config_str.split(':', 1)
    opt_type = parts[0].strip().lower()
    
    # Parse parameters
    params = {}
    if len(parts) > 1 and parts[1].strip():
        for param in parts[1].split(','):
            param = param.strip()
            if '=' not in param:
                raise ValueError(f"Parameter must be in format 'key=value'. Got: {param}")
            
            key, val = param.split('=', 1)
            key = key.strip()
            val = val.strip()
            
            # Try to convert to appropriate type
            try:
                # Check if it's an integer
                if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                    params[key] = int(val)
                # Check if it's a float
                else:
                    params[key] = float(val)
            except ValueError:
                # Keep as string if conversion fails
                params[key] = val
    
    # Create appropriate config
    if opt_type == 'muon':
        return MuonConfig(**params)
    elif opt_type == 'shampoo':
        return ShampooConfig(**params)
    elif opt_type == 'adam':
        return AdamConfig(**params)
    else:
        raise ValueError(
            f"Unknown optimizer type: {opt_type}. "
            f"Valid types: 'muon', 'shampoo', 'adam'"
        )


def create_optimizer(config: OptimizerConfig, filter_params, head_params, bias_params, 
                     weight_decay, weight_decay_misc, lr_head, lr_bias):
    """
    Create optimizer instances based on configuration.
    
    Returns:
        List of optimizer instances for this model
    """
    optimizers = []
    
    # Create Adam optimizer for biases and heads
    param_configs_adam = []
    if len(bias_params) > 0:
        param_configs_adam.append(dict(params=bias_params, lr=lr_bias, weight_decay=weight_decay_misc/lr_bias))
    if len(head_params) > 0:
        param_configs_adam.append(dict(params=head_params, lr=lr_head, weight_decay=weight_decay_misc/lr_head))
    
    if param_configs_adam:
        if isinstance(config, AdamConfig):
            adam_opt = Adam(param_configs_adam, lr=config.lr, betas=(config.beta1, config.beta2),
                          eps=1e-10, weight_decay=weight_decay)
        else:
            adam_opt = Adam(param_configs_adam, lr=lr_bias, betas=(0.9, 0.95),
                          eps=1e-10, weight_decay=weight_decay)
        optimizers.append(adam_opt)
    
    # Create main optimizer for filter params
    if len(filter_params) > 0:
        if isinstance(config, MuonConfig):
            main_opt = Muon(filter_params, lr=config.lr, momentum=config.momentum,
                          nesterov=True, weight_decay=weight_decay)
        elif isinstance(config, ShampooConfig):
            main_opt = Shampoo(filter_params, lr=config.lr, momentum=config.momentum,
                             weight_decay=weight_decay, order_multiplier=config.order_multiplier)
        elif isinstance(config, AdamConfig):
            # If using Adam for everything
            return optimizers  # Only use the Adam created above
        else:
            raise ValueError(f"Unknown optimizer config: {type(config)}")
        
        optimizers.append(main_opt)
    
    return optimizers


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

ValidArch = Union[CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet, CifarNet]


def main(
    arch: ValidArch,        
    optimizer_configs: List[OptimizerConfig] = None,  # List of optimizer configurations (programmatic use)
    optimizer_configs_str: List[str] = None,          # List of optimizer config strings (CLI use)
    data_path: str = "cifar10",       # path to store CIFAR10 data
    batch_size: int = 16192,   # batch size for training
    lr_bias: float = 0.01,            # learning rate for biases
    lr_head: float = 0.01,        # learning rate for head/output layer
    weight_decay: float = 1,     # weight decay
    weight_decay_misc: float = 1e-4,     # weight decay for miscellaneous parameters
    use_augmentation: bool = True,    # whether to use data augmentation
    label_smoothing: float = 0.2,     # label smoothing parameter
    device: str = "cuda",             # cuda or cpu
    seed: int = 0,                    # random seed
    save_results: bool = True,        # whether to save results
    svd_freq: int = 20,               # how often to record singular values (in steps)
    total_train_steps: int = 400,     # total training steps
):
    # Parse string configs if provided via CLI
    if optimizer_configs_str is not None:
        try:
            optimizer_configs = [parse_optimizer_config(s) for s in optimizer_configs_str]
        except Exception as e:
            print(f"Error parsing optimizer configs: {e}")
            print("\nExpected format: 'OptimizerType:param1=value1,param2=value2'")
            print("Examples:")
            print("  Muon:lr=0.0005,momentum=0.9")
            print("  Shampoo:lr=0.0005,momentum=0.9,order_multiplier=2")
            raise
    # Default optimizer configurations if none provided
    elif optimizer_configs is None:
        optimizer_configs = [
            ShampooConfig(lr=0.0005, momentum=0.9, order_multiplier=1),
            ShampooConfig(lr=0.0005, momentum=0.9, order_multiplier=2),
            MuonConfig(lr=0.0005, momentum=0.9),
        ]
    
    print("=" * 80)
    print(f"Multi-Model Training: {len(optimizer_configs)} Optimizers")
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
    print("\n[1/5] Loading Data...")
    aug = dict(flip=True, translate=2) if use_augmentation else {}
    train_loader = CifarLoader(data_path, train=True, batch_size=batch_size, aug=aug)
    test_loader = CifarLoader(data_path, train=False, batch_size=2000)
    batch_sweep_count = 1
    total_train_steps = ceil(batch_sweep_count * len(train_loader))
    total_epochs = ceil(total_train_steps / len(train_loader))

    print(f"  - Training samples: {len(train_loader.images)}")
    print(f"  - Test samples: {len(test_loader.images)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Total steps: {total_train_steps}")
    print(f"  - Total epochs: {total_epochs}")
    
    # Create models - one for each optimizer config
    print("\n[2/5] Creating Models...")
    models = {}
    base_model = arch.create(input_shape=(3, 32, 32), output_dim=10).to(device)
    base_state_dict = base_model.state_dict()
    
    for i, opt_config in enumerate(optimizer_configs):
        model_name = f"{opt_config}"
        model = arch.create(input_shape=(3, 32, 32), output_dim=10).to(device)
        model.load_state_dict(base_state_dict)  # All start with same weights
        models[model_name] = model
        print(f"  - Model {i+1}: {model_name}")
    
    print(f"  - Architecture: {arch.__class__.__name__}")
    print(f"  - Total parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"  - Number of models: {len(models)}")
    
    #############################################
    # Setup optimizers for all models
    #############################################
    print("\n[3/5] Setting up Optimizers...")
    
    optimizers_dict = {}  # model_name -> list of optimizers
    filter_param_names_dict = {}  # model_name -> list of filter param names
    
    for model_name, (opt_config, model) in zip(models.keys(), zip(optimizer_configs, models.values())):
        # Separate parameters by type
        filter_params = [p for n, p in model.named_parameters() 
                        if ((p.ndim >= 2 and "embed" not in n and "head" not in n) and p.requires_grad)]
        head_params = [p for n, p in model.named_parameters() 
                      if (("embed" in n or 'cls' in n or 'head' in n) and p.requires_grad and p.ndim >= 2)]
        bias_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        
        # Create optimizers for this model
        opts = create_optimizer(opt_config, filter_params, head_params, bias_params,
                               weight_decay, weight_decay_misc, lr_head, lr_bias)
        
        # Set initial learning rates
        for opt in opts:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        optimizers_dict[model_name] = opts
        
        # Store filter param names for SVD tracking
        filter_param_names = [n for n, p in model.named_parameters() 
                             if ((p.ndim >= 2 and "embed" not in n and "head" not in n) and p.requires_grad)]
        filter_param_names_dict[model_name] = filter_param_names
        
        print(f"  - {model_name}: {len(filter_params)} filter, {len(head_params)} head, {len(bias_params)} bias params")
    
    # Storage for logging
    model_logs = {name: [] for name in models.keys()}  # Per-model training logs
    singular_values_logs = {name: [] for name in models.keys()}  # Per-model SVD logs
    
    # Create loggers
    print("\n[4/5] Setting up Loggers...")
    logger_suite = create_default_loggers(
        label_smoothing=label_smoothing,
        track_singular_values=False  # We handle SVD separately
    )
    print(f"  - Process loggers: {len(logger_suite['process'])}")
    print(f"  - Group loggers: {len(logger_suite['group'])}")
    
    # Print header
    print("\n[5/5] Training...")
    print_columns(logging_columns_list, is_head=True)
    
    step = 0
    #############################################
    # Training loop
    #############################################
    
    for epoch in range(total_epochs):
        # Set all models to train mode
        for model in models.values():
            model.train()
        
        # Metrics for each model
        epoch_metrics = {name: {'loss': 0.0, 'correct': 0, 'samples': 0} 
                        for name in models.keys()}
        
        # Training
        for inputs, labels in train_loader:
            # Train each model
            for model_name, model in models.items():
                # Forward pass
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels, 
                                     label_smoothing=label_smoothing, reduction='sum')
                loss.backward()
                
                # Update learning rate
                for opt in optimizers_dict[model_name]:
                    for group in opt.param_groups:
                        group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                
                # Optimizer step
                for opt in optimizers_dict[model_name]:
                    opt.step()
                
                # Zero gradients
                model.zero_grad(set_to_none=True)
                
                # Track metrics
                with torch.no_grad():
                    epoch_metrics[model_name]['loss'] += loss.item()
                    epoch_metrics[model_name]['correct'] += (outputs.argmax(1) == labels).float().sum().item()
                    epoch_metrics[model_name]['samples'] += len(inputs)
            
            #############################################
            # Record singular values every svd_freq steps
            #############################################
            if step % svd_freq == 0:
                print(f"\n  [Step {step}] Computing singular values...")
                for model_name, model in models.items():
                    sv = compute_singular_values(model, filter_param_names_dict[model_name])
                    singular_values_logs[model_name].append((step, sv))
                print(f"  [Step {step}] Recorded SVD for {len(models)} models")
            
            step += 1
            if step >= total_train_steps:
                break
        
        # Evaluate and log each model
        for i, (model_name, model) in enumerate(models.items()):
            # Compute training metrics
            metrics = epoch_metrics[model_name]
            train_loss = metrics['loss'] / metrics['samples']
            train_acc = metrics['correct'] / metrics['samples']
            
            # Evaluate on test set
            test_loss, test_acc = evaluate(model, test_loader)
            
            # Get current learning rate
            current_lr = optimizers_dict[model_name][0].param_groups[0]['lr'] if optimizers_dict[model_name] else 0.0
            
            # Print log
            log_dict = {
                'epoch': epoch,
                'opt': model_name[:15],  # Truncate if too long
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': current_lr,
            }
            is_last = (i == len(models) - 1) and (epoch == total_epochs - 1)
            print_training_details(log_dict, is_final_entry=is_last)
            
            # Store log
            model_logs[model_name].append({
                'epoch': epoch,
                'step': step,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': current_lr,
            })
        
        if step >= total_train_steps:
            break
    
    #############################################
    # Save results
    #############################################
    
    if save_results:
        print("\n" + "=" * 80)
        print("Saving Results...")
        
        # Create descriptive directory name
        optimizer_names = "_vs_".join([str(cfg).replace("_", "-")[:20] for cfg in optimizer_configs])
        log_dir = os.path.join('logs', f'multi_training_{optimizer_names}_{str(uuid.uuid4())[:8]}')
        os.makedirs(log_dir, exist_ok=True)
        
        # Save configuration and code
        config_data = {
            'code': code,
            'config': config,
            'optimizer_configs': [str(cfg) for cfg in optimizer_configs],
        }
        
        # Add final test accuracies to config
        for model_name in models.keys():
            if model_logs[model_name]:
                config_data[f'test_acc_{model_name}'] = model_logs[model_name][-1]['test_acc']
        
        torch.save(config_data, os.path.join(log_dir, 'config.pt'))
        
        # Save metrics for each model
        for model_name in models.keys():
            # Save as numpy array (old format for compatibility)
            metrics_array = np.array([
                [log['epoch'], log['train_loss'], log['train_acc'], 
                 log['test_loss'], log['test_acc'], log['lr']]
                for log in model_logs[model_name]
            ])
            safe_name = model_name.replace(".", "_").replace("/", "_")
            np.save(
                os.path.join(log_dir, f"metrics_{safe_name}.npy"),
                metrics_array,
                allow_pickle=True
            )
            
            # Also save as JSON for easy reading
            with open(os.path.join(log_dir, f"metrics_{safe_name}.json"), 'w') as f:
                json.dump(model_logs[model_name], f, indent=2)
        
        # Save singular values for each model
        for model_name in models.keys():
            safe_name = model_name.replace(".", "_").replace("/", "_")
            with open(os.path.join(log_dir, f"singular_values_{safe_name}.pkl"), 'wb') as f:
                pickle.dump(singular_values_logs[model_name], f)
        
        # Save models
        for model_name, model in models.items():
            safe_name = model_name.replace(".", "_").replace("/", "_")
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_{safe_name}.pt"))
        
        # Save a README describing the experiment
        readme_content = f"""# Multi-Model Training Experiment

## Configuration
- Architecture: {arch.__class__.__name__}
- Total Parameters: {sum(p.numel() for p in base_model.parameters()):,}
- Batch Size: {batch_size}
- Total Steps: {total_train_steps}
- Total Epochs: {total_epochs}

## Optimizers
"""
        for i, (opt_config, model_name) in enumerate(zip(optimizer_configs, models.keys())):
            final_test_acc = model_logs[model_name][-1]['test_acc'] if model_logs[model_name] else 0.0
            readme_content += f"{i+1}. {model_name}\n"
            readme_content += f"   - Final Test Accuracy: {final_test_acc:.4f}\n"
        
        with open(os.path.join(log_dir, "README.md"), 'w') as f:
            f.write(readme_content)

        print(f"  - Results saved to: {os.path.abspath(log_dir)}")
        print(f"  - Config: config.pt, README.md")
        print(f"  - Metrics: metrics_*.npy and metrics_*.json for each model")
        print(f"  - Singular values: singular_values_*.pkl for each model")
        print(f"  - Models: model_*.pt for each model")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # Print final results for all models
    for model_name in models.keys():
        if model_logs[model_name]:
            final_test_acc = model_logs[model_name][-1]['test_acc']
            print(f"Final Test Accuracy ({model_name[:30]}): {final_test_acc:.4f}")
    
    print("=" * 80)
    
    # Return results
    results = {
        'model_logs': model_logs,
        'singular_values': singular_values_logs,
    }
    for model_name in models.keys():
        if model_logs[model_name]:
            results[f'test_acc_{model_name}'] = model_logs[model_name][-1]['test_acc']
    
    return results


if __name__ == "__main__":
    args = tyro.cli(main, config=[tyro.conf.ConsolidateSubcommandArgs])

