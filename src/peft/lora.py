#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) implementation for STRATO-PEFT experimental framework.

This module implements the LoRA method for parameter-efficient fine-tuning.

Reference:
    LoRA: Low-Rank Adaptation of Large Language Models
    https://arxiv.org/abs/2106.09685

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .base_peft import BasePEFT, PEFTMetrics


class LoRALayer(nn.Module):
    """
    LoRA layer implementation.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0
    ):
        """
        Initialize LoRA layer.
        
        Args:
            original_layer: Original linear layer to adapt
            rank: LoRA rank
            alpha: LoRA scaling parameter
            dropout: Dropout rate
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Get dimensions - support both Linear and Conv1D
        from transformers.pytorch_utils import Conv1D
        
        if hasattr(original_layer, 'in_features') and hasattr(original_layer, 'out_features'):
            # Standard Linear layer
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        elif isinstance(original_layer, Conv1D):
            # GPT2's Conv1D layer - note the dimension order is different
            in_features = original_layer.weight.shape[0]  # Conv1D has transposed weights
            out_features = original_layer.weight.shape[1]
        else:
            raise ValueError(f"Unsupported layer type: {type(original_layer)}")
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def _init_weights(self) -> None:
        """
        Initialize LoRA weights.
        """
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA adaptation - need to handle different input shapes for Conv1D vs Linear
        from transformers.pytorch_utils import Conv1D
        
        if isinstance(self.original_layer, Conv1D):
            # For Conv1D, we need to ensure correct tensor shapes
            # Conv1D expects input as (..., in_features)
            input_shape = x.shape
            if len(input_shape) > 2:
                # Reshape to 2D for linear operations
                x_reshaped = x.view(-1, input_shape[-1])
                lora_output = self.lora_B(self.dropout(self.lora_A(x_reshaped)))
                # Reshape back to original shape
                lora_output = lora_output.view(*input_shape[:-1], -1)
            else:
                lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        else:
            # Standard Linear layer
            lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        
        # Scale and combine
        scaling = self.alpha / self.rank
        return original_output + scaling * lora_output
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """
        Get LoRA parameters.
        
        Returns:
            List of LoRA parameters
        """
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())


class LoRAPEFT(BasePEFT):
    """
    LoRA PEFT method implementation.
    """
    
    def _initialize_peft(self) -> None:
        """
        Initialize LoRA-specific components.
        """
        self.lora_layers: Dict[str, LoRALayer] = {}
        self.target_modules = self.config.target_modules
        self.rank = self.config.rank
        self.alpha = self.config.alpha
        self.dropout = getattr(self.config, 'dropout', 0.0)
        
        self.logger.info(f"Initializing LoRA with rank={self.rank}, alpha={self.alpha}, dropout={self.dropout}")
        self.logger.info(f"Target modules: {self.target_modules}")
    
    def apply_peft(self) -> nn.Module:
        """
        Apply LoRA to the model.
        
        Returns:
            The adapted model with LoRA applied
        """
        if self.is_applied:
            self.logger.warning("LoRA already applied")
            return self.adapted_model
        
        self.logger.info("Applying LoRA to model...")
        
        # Create a copy of the model
        self.adapted_model = self.model
        
        # Apply LoRA to target modules
        modules_replaced = 0
        for name, module in self.adapted_model.named_modules():
            if self._should_replace_module(name, module):
                self._replace_module_with_lora(name, module)
                modules_replaced += 1
        
        if modules_replaced == 0:
            self.logger.warning("No modules were replaced with LoRA layers")
        else:
            self.logger.info(f"Replaced {modules_replaced} modules with LoRA layers")
        
        self.is_applied = True
        self.log_peft_info()
        
        return self.adapted_model
    
    def _should_replace_module(self, name: str, module: nn.Module) -> bool:
        """
        Check if a module should be replaced with LoRA.
        
        Args:
            name: Module name
            module: Module instance
            
        Returns:
            True if module should be replaced
        """
        # Check if it's a linear layer or Conv1D (used by GPT2)
        from transformers.pytorch_utils import Conv1D
        if not isinstance(module, (nn.Linear, Conv1D)):
            return False
        
        # Check if name matches target modules
        for target in self.target_modules:
            if target in name:
                return True
        
        return False
    
    def _replace_module_with_lora(self, name: str, module: nn.Module) -> None:
        """
        Replace a module with LoRA layer.
        
        Args:
            name: Module name
            module: Module to replace
        """
        # Create LoRA layer
        lora_layer = LoRALayer(
            original_layer=module,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        # Replace module in the model
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent_module = dict(self.adapted_model.named_modules())[parent_name]
        else:
            parent_module = self.adapted_model
        
        setattr(parent_module, child_name, lora_layer)
        
        # Store reference
        self.lora_layers[name] = lora_layer
        
        self.logger.debug(f"Replaced {name} with LoRA layer (rank={self.rank})")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get the list of trainable LoRA parameters.
        
        Returns:
            List of trainable parameters
        """
        if not self.is_applied:
            return []
        
        trainable_params = []
        for lora_layer in self.lora_layers.values():
            trainable_params.extend(lora_layer.get_lora_parameters())
        
        return trainable_params
    
    def get_peft_metrics(self) -> PEFTMetrics:
        """
        Get LoRA-specific metrics.
        
        Returns:
            PEFT metrics
        """
        if not self.is_applied:
            return PEFTMetrics(
                trainable_params=0,
                total_params=0,
                trainable_ratio=0.0,
                memory_usage_mb=0.0
            )
        
        trainable_params, total_params, trainable_ratio = self.get_parameter_count()
        memory_usage = self.get_memory_usage()
        
        # LoRA-specific metrics
        rank_distribution = {name: self.rank for name in self.lora_layers.keys()}
        
        # Calculate adaptation efficiency (parameters saved vs. full fine-tuning)
        full_finetune_params = sum(p.numel() for p in self.model.parameters())
        adaptation_efficiency = 1.0 - (trainable_params / full_finetune_params)
        
        return PEFTMetrics(
            trainable_params=trainable_params,
            total_params=total_params,
            trainable_ratio=trainable_ratio,
            memory_usage_mb=memory_usage,
            rank_distribution=rank_distribution,
            adaptation_efficiency=adaptation_efficiency
        )
    
    def save_peft_weights(self, save_path: str) -> None:
        """
        Save LoRA weights.
        
        Args:
            save_path: Path to save the weights
        """
        if not self.is_applied:
            raise RuntimeError("LoRA must be applied before saving weights")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Collect LoRA weights
        lora_state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            lora_state_dict[f"{name}.lora_A.weight"] = lora_layer.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = lora_layer.lora_B.weight
        
        # Save weights and config
        torch.save({
            'lora_state_dict': lora_state_dict,
            'config': {
                'rank': self.rank,
                'alpha': self.alpha,
                'dropout': self.dropout,
                'target_modules': self.target_modules
            }
        }, save_path)
        
        self.logger.info(f"LoRA weights saved to {save_path}")
    
    def load_peft_weights(self, load_path: str) -> None:
        """
        Load LoRA weights.
        
        Args:
            load_path: Path to load the weights from
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"LoRA weights file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        if not self.is_applied:
            self.logger.warning("LoRA not applied yet, applying first...")
            self.apply_peft()
        
        # Load weights
        for name, lora_layer in self.lora_layers.items():
            if f"{name}.lora_A.weight" in lora_state_dict:
                lora_layer.lora_A.weight.data = lora_state_dict[f"{name}.lora_A.weight"]
            if f"{name}.lora_B.weight" in lora_state_dict:
                lora_layer.lora_B.weight.data = lora_state_dict[f"{name}.lora_B.weight"]
        
        self.logger.info(f"LoRA weights loaded from {load_path}")
    
    def validate_config(self) -> None:
        """
        Validate LoRA configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        super().validate_config()
        
        # LoRA-specific validation
        if not hasattr(self.config, 'rank') or self.config.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        
        if not hasattr(self.config, 'alpha') or self.config.alpha <= 0:
            raise ValueError("LoRA alpha must be a positive number")
        
        if hasattr(self.config, 'dropout') and (self.config.dropout < 0 or self.config.dropout >= 1):
            raise ValueError("LoRA dropout must be in range [0, 1)")


# Alias for backward compatibility
LoRAAdapter = LoRAPEFT