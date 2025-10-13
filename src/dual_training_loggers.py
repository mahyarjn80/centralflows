"""
Loggers for dual training experiments (comparing Shampoo with different order_multipliers, Muon, etc.)

This module provides loggers specifically designed for the dual/triple model training setup,
following the same pattern as src/loggers.py but adapted for torch.nn.Module models
rather than functional models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelLogger:
    """Base class for loggers that record properties of a single model during training."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log properties of the model and optimizer.
        
        Args:
            model: The neural network model
            optimizer: The optimizer being used
            **kwargs: Additional context (e.g., data loaders, device)
            
        Returns:
            Dictionary of logged values
        """
        raise NotImplementedError()


class GroupLogger:
    """Base class for loggers that record collective properties of multiple models."""
    
    def log(self, models: Dict[str, nn.Module], **kwargs) -> Dict[str, Any]:
        """Log collective properties across multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances
            **kwargs: Additional context
            
        Returns:
            Dictionary of logged values
        """
        raise NotImplementedError()


@dataclass
class LossAndAccuracyLogger(ModelLogger):
    """Logs loss and accuracy on train or test set."""
    
    split: Literal["train", "test"]  # Which split to evaluate on
    label_smoothing: float = 0.0     # Label smoothing for loss computation
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Compute and log loss and accuracy."""
        loader = kwargs.get(f'{self.split}_loader')
        device = kwargs.get('device', 'cuda')
        batch_size = kwargs.get('batch_size', 2000)

        
        if loader is None:
            return {}
        
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            # For CifarLoader-style loaders
            if hasattr(loader, 'normalize') and hasattr(loader, 'images'):
                images = loader.normalize(loader.images)
                labels = loader.labels
                
                # Process in batches
                for i in range(0, len(images), batch_size):
                    inputs = images[i:i+batch_size]
                    targets = labels[i:i+batch_size]
                    
                    outputs = model(inputs)
                    loss = F.cross_entropy(
                        outputs, targets,
                        label_smoothing=self.label_smoothing,
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_correct += (outputs.argmax(1) == targets).float().sum().item()
                    total_samples += len(inputs)
            else:
                # For standard PyTorch DataLoader
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = F.cross_entropy(
                        outputs, targets,
                        label_smoothing=self.label_smoothing,
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_correct += (outputs.argmax(1) == targets).float().sum().item()
                    total_samples += len(inputs)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {
            f'{self.split}_loss': avg_loss,
            f'{self.split}_acc': avg_acc,
        }


class OptimizerStateLogger(ModelLogger):
    """Logs optimizer state information."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log optimizer-specific state."""
        out = {}
        
        # Log learning rates
        lrs = [group['lr'] for group in optimizer.param_groups]
        out['lr'] = lrs[0] if len(lrs) == 1 else lrs
        
        # Log Shampoo-specific info
        if hasattr(optimizer, 'order_multiplier'):
            out['order_multiplier'] = optimizer.order_multiplier
            
        # Count parameters with state
        params_with_state = len([p for p in model.parameters() if p in optimizer.state])
        out['params_with_state'] = params_with_state
        
        # Log step count (if available)
        if optimizer.state:
            steps = [state.get('step', 0) for state in optimizer.state.values() if 'step' in state]
            if steps:
                out['optimizer_step'] = max(steps)
        
        return {'opt': out}


class SingularValueLogger(ModelLogger):
    """Logs singular values of weight matrices."""
    
    def __init__(self, param_name_filter=None):
        """Initialize singular value logger.
        
        Args:
            param_name_filter: Optional function to filter which parameters to track.
                             Should return True for parameters to track.
        """
        self.param_name_filter = param_name_filter or (lambda n, p: p.ndim >= 2)
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Compute singular values for tracked parameters."""
        singular_values = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not self.param_name_filter(name, param):
                    continue
                
                # Reshape to 2D if needed
                if param.ndim == 2:
                    matrix = param.data
                else:
                    matrix = param.data.reshape(param.size(0), -1)
                
                try:
                    # Compute SVD (only singular values)
                    matrix_cpu = matrix.cpu().float()
                    _, s, _ = torch.svd(matrix_cpu)
                    singular_values[name] = s.numpy()
                except Exception as e:
                    # Skip if SVD fails
                    continue
        
        return {'singular_values': singular_values}


class GradientNormLogger(ModelLogger):
    """Logs gradient norms."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log gradient statistics."""
        total_norm = 0.0
        param_norms = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm().item()
                    total_norm += grad_norm ** 2
                    param_norms[name] = grad_norm
        
        total_norm = total_norm ** 0.5
        
        return {
            'grad_norm': total_norm,
            'param_grad_norms': param_norms,
        }


class ParameterStatsLogger(ModelLogger):
    """Logs parameter statistics (norms, means, stds)."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log parameter statistics."""
        stats = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                stats[f'{name}_norm'] = param.data.norm().item()
                stats[f'{name}_mean'] = param.data.mean().item()
                stats[f'{name}_std'] = param.data.std().item()
        
        return {'param_stats': stats}


class ModelDistanceLogger(GroupLogger):
    """Logs distances between models' parameters."""
    
    def log(self, models: Dict[str, nn.Module], **kwargs) -> Dict[str, Any]:
        """Compute pairwise parameter distances between models."""
        distances = {}
        names = sorted(models.keys())
        
        with torch.no_grad():
            for i in range(len(names)):
                for j in range(i):
                    model_i, model_j = models[names[i]], models[names[j]]
                    
                    # Compute L2 distance between all parameters
                    total_dist = 0.0
                    for (name_i, param_i), (name_j, param_j) in zip(
                        model_i.named_parameters(),
                        model_j.named_parameters()
                    ):
                        if name_i == name_j:  # Make sure parameters correspond
                            total_dist += (param_i - param_j).norm().item() ** 2
                    
                    total_dist = total_dist ** 0.5
                    distances[f'{names[j]}_to_{names[i]}'] = total_dist
        
        return {'model_distances': distances}


class TrainingProgressLogger(ModelLogger):
    """Logs training progress metrics (epoch, step, etc.)."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log training progress information."""
        return {
            'epoch': kwargs.get('epoch', 0),
            'step': kwargs.get('step', 0),
            'total_steps': kwargs.get('total_steps', 0),
        }


def create_default_loggers(
    label_smoothing: float = 0.0,
    track_singular_values: bool = True,
    singular_value_freq: int = 1,
) -> Dict[str, list]:
    """Create a default set of loggers for dual training.
    
    Args:
        label_smoothing: Label smoothing parameter
        track_singular_values: Whether to track singular values
        singular_value_freq: How often to log singular values (every N calls)
        
    Returns:
        Dictionary mapping 'process' and 'group' to lists of loggers
    """
    process_loggers = [
        LossAndAccuracyLogger(split='train', label_smoothing=label_smoothing),
        LossAndAccuracyLogger(split='test', label_smoothing=label_smoothing),
        OptimizerStateLogger(),
        GradientNormLogger(),
        TrainingProgressLogger(),
    ]
    
    if track_singular_values:
        # Filter for filter/weight parameters (not biases or embeddings)
        def filter_params(name, param):
            return (param.ndim >= 2 and 
                   "embed" not in name and 
                   "head" not in name and
                   "bias" not in name)
        
        process_loggers.append(SingularValueLogger(param_name_filter=filter_params))
    
    group_loggers = [
        ModelDistanceLogger(),
    ]
    
    return {
        'process': process_loggers,
        'group': group_loggers,
    }

