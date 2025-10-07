"""
Main training script using Shampoo optimizer approach from shmapooV2.py
with model/data loading from main.py

This script uses the CifarLoader from shmapooV2.py for efficient CIFAR-10 data loading
and the Shampoo optimizer for second-order optimization.

Example usage:
    python main_shampoo.py --arch mlp --epochs 100 --batch-size 2000 --lr-filters 0.24
    
    python main_shampoo.py --arch cnn --epochs 50 --lr-bias 0.01 --lr-head 0.1 --use-augmentation
    
    python main_shampoo.py --arch resnet --data-path ./data/cifar10 --save-results
"""

import json
import os
import sys
import uuid
from math import ceil
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import git
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import tyro
from tqdm import trange
from tyro.conf import arg, subcommand

from src.architectures import CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet
from src.utils import convert_dataclasses

# Import optimizers from shmapooV2.py
# We'll need to copy the Shampoo and Muon optimizer classes

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
# Note: CifarLoader uses randomness, so we can't use deterministic algorithms
# torch.use_deterministic_algorithms(True)
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
        )
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
                    grad.mul_(1 - momentum).add_(
                        state['momentum_buffer'], alpha=momentum
                    )

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state['precond_{}'.format(dim_id)]
                    inv_precond = state['inv_precond_{}'.format(dim_id)]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()

                    g32 = grad.to(torch.float32)
                    g32_t = g32.t()
                    precond.add_(g32 @ g32_t)
                    if state['step'] % group['update_freq'] == 0:
                        inv_precond.copy_(_matrix_power(precond,  (order)))

                    if dim_id == order - 1:
                        # finally
                        g32 = g32.t() @ inv_precond
                        # grad: (-1, last_dim)
                        grad = g32.view(original_size).to(grad.dtype)
                    else:
                        # if not final
                        g32 = inv_precond @ g32
                        # grad (dim, -1)
                        grad = g32.view(transposed_size).to(grad.dtype)

                state['step'] += 1
                state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-group['lr'])

        return loss


class Adam(torch.optim.Optimizer):
    def __init__(self, param_groups):
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


logging_columns_list = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'lr']


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
    arch: ValidArch,                  # which architecture to train
    data_path: str = "cifar10",       # path to store CIFAR10 data
    batch_size: int = 2000,           # batch size for trainin               
    lr_bias: float = 0.01,            # learning rate for biases
    lr_filters: float = 0.24,         # learning rate for filter params (Shampoo)
    lr_head: float = 0.1,             # learning rate for head/output layer
    momentum_sgd: float = 0.85,       # momentum for SGD optimizer
    momentum_shampoo: float = 0.9,    # momentum for Shampoo optimizer
    weight_decay: float = 1e-4,       # weight decay
    use_augmentation: bool = True,    # whether to use data augmentation
    label_smoothing: float = 0.2,     # label smoothing parameter
    device: str = "cuda",             # cuda or cpu
    seed: int = 0,                    # random seed
    expid: Optional[str] = None,      # optionally, an experiment id (defaults to a random UUID)
    save_results: bool = True,        # whether to save results
):
    print("Starting training with Shampoo optimizer")
    
    # collect configs that were passed in
    config = convert_dataclasses(locals())
    try:
        config["git_hash"] = git.Repo(".").git.rev_parse("HEAD")
    except:
        config["git_hash"] = "unknown"
    config["cmd"] = " ".join(sys.argv)
    
    # experiment id defaults to random uuid
    expid = expid or uuid.uuid4().hex
    
    # set random seed
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    # load the dataset using CifarLoader
    print("Loading Data")
    aug = dict(flip=True, translate=2) if use_augmentation else {}
    train_loader = CifarLoader(data_path, train=True, batch_size=batch_size, aug=aug)
    test_loader = CifarLoader(data_path, train=False, batch_size=2000)
    total_train_steps = ceil(8 * len(train_loader))

    # instantiate the model as a PyTorch module
    print("Creating Model")
    # CIFAR10: input_shape = (3, 32, 32), output_dim = 10
    model = arch.create(input_shape=(3, 32, 32), output_dim=10).to(device)
    
    #############################################
    # NEW: Manual optimization setup like shmapooV2.py
    #############################################
    
    

    filter_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    head_params = [p for n, p in model.named_parameters() if ("embed" in n or "cls" in n)]
    bias_params = [p for p in model.parameters() if p.ndim < 2]

    # Create optimizers
    # Optimizer 1: SGD for biases and head
    param_configs_sgd = []
    if len(bias_params) > 0:
        param_configs_sgd.append(dict(params=bias_params, lr=lr_bias, weight_decay=weight_decay/lr_bias))
    if len(head_params) > 0:
        param_configs_sgd.append(dict(params=head_params, lr=lr_head, weight_decay=weight_decay/lr_head))
    
    optimizer1 = Adam(param_configs_sgd, lr=lr_bias, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay) if param_configs_sgd else None
    
    # Optimizer 2: Shampoo for filter parameters
    optimizer2 = Shampoo(filter_params, lr=lr_filters, momentum=momentum_shampoo, weight_decay=weight_decay) if len(filter_params) > 0 else None
    
    optimizers = [opt for opt in [optimizer1, optimizer2] if opt is not None]
    
    # Set initial learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    # Storage for logging
    RUN_LOGS = []
    
    # Print header
    print_columns(logging_columns_list, is_head=True)
    

    step = 0
    #############################################
    # Training loop
    #############################################
    
    for epoch in range(ceil(total_train_steps / len(train_loader))):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        # Training
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, label_smoothing=label_smoothing, reduction='sum')
            
            # Backward pass
            loss.backward()
            
            for group in optimizer1.param_groups+optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)

            
            # Optimizer step
            for opt in optimizers:
                opt.step()
            
            # Zero gradients
            model.zero_grad(set_to_none=True)
            
            # Track metrics
            with torch.no_grad():
                epoch_loss += loss.item()
                epoch_correct += (outputs.argmax(1) == labels).float().sum().item()
                epoch_samples += len(inputs)
            step += 1
            if step >= total_train_steps:
                break
        
        # Compute training metrics
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader)
        
        # Get current learning rate (from first optimizer)
        current_lr = optimizers[0].param_groups[0]['lr'] if optimizers else 0.0
        
        # Log
        log_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': current_lr,
        }
        print_training_details(log_dict, is_final_entry=(epoch == epochs - 1))
        
        # Store log
        RUN_LOGS.append([epoch, train_loss, train_acc, test_loss, test_acc, current_lr])
    
    #############################################
    # Save results
    #############################################
    
    if save_results:
        # Create experiment folder
        experiment_dir = Path(os.environ.get("EXPERIMENT_DIR", "experiments"))
        folder = experiment_dir / expid
        folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to: {folder}")
        
        # Save config
        with open(folder / "config.json", "w") as config_file:
            json.dump(config, config_file, indent=4)
        
        # Save logs
        np.save(
            folder / "training_logs.npy",
            np.array(RUN_LOGS, dtype=object),
            allow_pickle=True
        )
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'logs': RUN_LOGS,
        }, folder / "model.pt")
        
        print(f"Saved model and logs to {folder}")
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    return test_acc


if __name__ == "__main__":
    args = tyro.cli(main, config=[tyro.conf.ConsolidateSubcommandArgs])

