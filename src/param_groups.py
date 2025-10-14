"""
Parameter grouping strategies for different model architectures.

This module provides functions to separate model parameters into appropriate groups
for optimization. Different architectures may benefit from different grouping strategies.
"""

from typing import List, Tuple
import torch.nn as nn


def vit_param_groups(model: nn.Module) -> Tuple[List, List, List]:
    """
    Parameter grouping strategy for Vision Transformer (ViT) models.
    
    Groups:
    - Filter params: 2D+ parameters (weight matrices), excluding embeddings and heads
    - Head params: Embeddings, classification heads, and output layers (2D+ only)
    - Bias params: 1D parameters (biases, layer norm scales, etc.)
    
    Args:
        model: ViT model
        
    Returns:
        Tuple of (filter_params, head_params, bias_params)
    """
    filter_params = [
        p for n, p in model.named_parameters() 
        if ((p.ndim >= 2 and "embed" not in n and "head" not in n) and p.requires_grad)
    ]
    filter_names = [n for n, p in model.named_parameters() if ((p.ndim >= 2 and "embed" not in n and "head" not in n) and p.requires_grad)]
    
    head_params = [
        p for n, p in model.named_parameters() 
        if (("embed" in n or 'cls' in n or 'head' in n) and p.requires_grad and p.ndim >= 2)
    ]
    
    bias_params = [
        p for p in model.parameters() 
        if p.requires_grad and p.ndim < 2
    ]
    
    return filter_params, head_params, bias_params, filter_names


def cifarnet_param_groups(model: nn.Module) -> Tuple[List, List, List]:
    """
    Parameter grouping strategy for CifarNet models.
    
    Groups:
    - Filter params: 4D convolutional parameters
    - Head params: Head/output layer parameters
    - Bias params: Normalization parameters and biases
    
    Args:
        model: CifarNet model
        
    Returns:
        Tuple of (filter_params, head_params, bias_params)
    """

    

       # Shampoo model optimizers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    filter_names = [n for n, p in model.named_parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases_shampoo = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    
    # Combine normalization biases and whitening bias
    bias_params = norm_biases_shampoo + [model.whiten.bias]

    head_params = [model.head.weight]
    

    return filter_params, head_params, bias_params, filter_names


def get_param_groups(
    model: nn.Module,
    strategy: str = 'vit'
) -> Tuple[List, List, List]:
    """
    Get parameter groups for a model using the specified strategy.
    
    Args:
        model: PyTorch model
        strategy: Name of the grouping strategy to use. Options:
            - 'vit': Vision Transformer grouping (default)
            - 'cifarnet': CifarNet grouping
            
    Returns:
        Tuple of (filter_params, head_params, bias_params)
        
    Raises:
        ValueError: If strategy is not recognized
        
    Example:
        >>> model = ViT(...)
        >>> filter_params, head_params, bias_params = get_param_groups(model, 'vit')
        >>> print(f"Filter params: {len(filter_params)}")
    """
    strategy = strategy.lower()
    
    if strategy == 'vit':
        return vit_param_groups(model)
    elif strategy == 'cifarnet':
        return cifarnet_param_groups(model)
    else:
        raise ValueError(
            f"Unknown parameter grouping strategy: '{strategy}'. "
            f"Available strategies: 'vit', 'cifarnet'"
        )
