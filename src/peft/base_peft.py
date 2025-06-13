#!/usr/bin/env python3
"""
Base PEFT method abstract class for STRATO-PEFT experimental framework.

This module defines the common interface for all PEFT methods.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import PreTrainedModel


@dataclass
class PEFTMetrics:
    """
    Metrics for PEFT methods.
    """
    trainable_params: int
    total_params: int
    trainable_ratio: float
    memory_usage_mb: float
    flops_per_forward: Optional[int] = None
    rank_distribution: Optional[Dict[str, int]] = None
    adaptation_efficiency: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.trainable_ratio is None and self.total_params > 0:
            self.trainable_ratio = self.trainable_params / self.total_params


class BasePEFT(ABC):
    """
    Abstract base class for all PEFT methods.
    """
    
    def __init__(self, config: DictConfig, model: PreTrainedModel, logger: Optional[logging.Logger] = None):
        """
        Initialize the PEFT method.
        
        Args:
            config: PEFT configuration
            model: Pre-trained model to adapt
            logger: Logger instance
        """
        self.config = config
        self.model = model
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.adapted_model: Optional[nn.Module] = None
        self.is_applied = False
        
        # Initialize method-specific components
        self._initialize_peft()
    
    @abstractmethod
    def _initialize_peft(self) -> None:
        """
        Initialize PEFT-specific components.
        
        This method should be implemented by each PEFT method to set up
        method-specific parameters and components.
        """
        pass
    
    @abstractmethod
    def apply_peft(self) -> nn.Module:
        """
        Apply the PEFT method to the model.
        
        Returns:
            The adapted model with PEFT applied
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get the list of trainable parameters after PEFT application.
        
        Returns:
            List of trainable parameters
        """
        pass
    
    @abstractmethod
    def get_peft_metrics(self) -> PEFTMetrics:
        """
        Get metrics specific to this PEFT method.
        
        Returns:
            PEFT metrics including parameter counts, memory usage, etc.
        """
        pass
    
    @abstractmethod
    def save_peft_weights(self, save_path: str) -> None:
        """
        Save PEFT-specific weights.
        
        Args:
            save_path: Path to save the weights
        """
        pass
    
    @abstractmethod
    def load_peft_weights(self, load_path: str) -> None:
        """
        Load PEFT-specific weights.
        
        Args:
            load_path: Path to load the weights from
        """
        pass
    
    def get_parameter_count(self) -> Tuple[int, int, float]:
        """
        Get parameter count information.
        
        Returns:
            Tuple of (trainable_params, total_params, trainable_ratio)
        """
        if not self.is_applied:
            # Before PEFT application, use base model
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            # After PEFT application, use adapted model
            total_params = sum(p.numel() for p in self.adapted_model.parameters())
            trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
        
        return trainable_params, total_params, trainable_ratio
    
    def get_memory_usage(self) -> float:
        """
        Get memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        if not self.is_applied:
            return 0.0
        
        memory_bytes = sum(
            p.numel() * p.element_size() 
            for p in self.adapted_model.parameters()
        )
        return memory_bytes / (1024 * 1024)  # Convert to MB
    
    def freeze_base_model(self) -> None:
        """
        Freeze all parameters in the base model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self.logger.info("Base model parameters frozen")
    
    def unfreeze_base_model(self) -> None:
        """
        Unfreeze all parameters in the base model.
        """
        for param in self.model.parameters():
            param.requires_grad = True
        self.logger.info("Base model parameters unfrozen")
    
    def log_peft_info(self) -> None:
        """
        Log information about the PEFT method.
        """
        if not self.is_applied:
            self.logger.warning("PEFT method not applied yet")
            return
        
        metrics = self.get_peft_metrics()
        
        self.logger.info(f"PEFT Method: {self.__class__.__name__}")
        self.logger.info(f"Trainable parameters: {metrics.trainable_params:,}")
        self.logger.info(f"Total parameters: {metrics.total_params:,}")
        self.logger.info(f"Trainable ratio: {metrics.trainable_ratio:.4f} ({metrics.trainable_ratio*100:.2f}%)")
        self.logger.info(f"Memory usage: {metrics.memory_usage_mb:.2f} MB")
        
        if metrics.rank_distribution:
            self.logger.info(f"Rank distribution: {metrics.rank_distribution}")
        
        if metrics.adaptation_efficiency is not None:
            self.logger.info(f"Adaptation efficiency: {metrics.adaptation_efficiency:.4f}")
    
    def validate_config(self) -> None:
        """
        Validate the PEFT configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Basic validation - can be overridden by subclasses
        if not hasattr(self.config, 'target_modules'):
            raise ValueError("PEFT config must specify target_modules")
        
        if not self.config.target_modules:
            raise ValueError("target_modules cannot be empty")
    
    def __repr__(self) -> str:
        """String representation of the PEFT method."""
        status = "applied" if self.is_applied else "not applied"
        return f"{self.__class__.__name__}(status={status})"