#!/usr/bin/env python3
"""
Base task class for STRATO-PEFT experimental framework.

This module defines the abstract base class for all tasks in the framework,
providing a unified interface for data loading, preprocessing, and evaluation.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer


@dataclass
class TaskMetrics:
    """
    Container for task evaluation metrics.
    """
    primary_metric: str
    metrics: Dict[str, float]
    detailed_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.primary_metric not in self.metrics:
            raise ValueError(f"Primary metric '{self.primary_metric}' not found in metrics")
    
    @property
    def primary_score(self) -> float:
        """Get the primary metric score."""
        return self.metrics[self.primary_metric]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "metrics": self.metrics,
            "detailed_results": self.detailed_results
        }


class BaseTask(ABC):
    """
    Abstract base class for all tasks in the STRATO-PEFT framework.
    
    This class defines the interface that all task implementations must follow,
    ensuring consistency across different tasks and enabling the framework
    to work with any task implementation.
    """
    
    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizer):
        """
        Initialize the base task.
        
        Args:
            config: Task configuration
            tokenizer: Pre-trained tokenizer
        """
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Task metadata
        self.task_name = config.get("name", "unknown")
        self.task_type = config.get("type", "unknown")
        
        # Data splits
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        # Evaluation metrics
        self.metric_names = self._get_metric_names()
        self.primary_metric = self._get_primary_metric()
        
        # Initialize task-specific components
        self._initialize_task()
    
    @abstractmethod
    def _initialize_task(self) -> None:
        """
        Initialize task-specific components.
        
        This method should be implemented by subclasses to set up
        task-specific configurations, data processors, etc.
        """
        pass
    
    @abstractmethod
    def _get_metric_names(self) -> List[str]:
        """
        Get the list of metric names for this task.
        
        Returns:
            List[str]: List of metric names
        """
        pass
    
    @abstractmethod
    def _get_primary_metric(self) -> str:
        """
        Get the primary metric name for this task.
        
        Returns:
            str: Primary metric name
        """
        pass
    
    @abstractmethod
    def load_data(self) -> None:
        """
        Load and preprocess the task data.
        
        This method should load the raw data, apply preprocessing,
        and create train/validation/test datasets.
        """
        pass
    
    @abstractmethod
    def create_dataloaders(self, 
                          batch_size: int,
                          num_workers: int = 0,
                          pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            
        Returns:
            Tuple[DataLoader, DataLoader, Optional[DataLoader]]: 
                Train, validation, and test data loaders
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                model: torch.nn.Module,
                dataloader: DataLoader,
                device: torch.device) -> TaskMetrics:
        """
        Evaluate the model on the given data loader.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            TaskMetrics: Evaluation results
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of data for model input.
        
        Args:
            batch: Raw batch data
            
        Returns:
            Dict[str, torch.Tensor]: Preprocessed batch
        """
        pass
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task information dictionary.
        
        Returns:
            Dict[str, Any]: Task information
        """
        return {
            "name": self.task_name,
            "type": self.task_type,
            "metric_names": self.metric_names,
            "primary_metric": self.primary_metric,
            "config": self.config,
            "data_info": self._get_data_info()
        }
    
    def _get_data_info(self) -> Dict[str, Any]:
        """
        Get data information dictionary.
        
        Returns:
            Dict[str, Any]: Data information
        """
        info = {}
        
        if self.train_dataset is not None:
            info["train_size"] = len(self.train_dataset)
        
        if self.val_dataset is not None:
            info["val_size"] = len(self.val_dataset)
        
        if self.test_dataset is not None:
            info["test_size"] = len(self.test_dataset)
        
        return info
    
    def validate_config(self) -> None:
        """
        Validate the task configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["name", "type", "dataset"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required task config field: {field}")
        
        # Validate dataset configuration
        dataset_config = self.config.dataset
        if "name" not in dataset_config:
            raise ValueError("Missing dataset name in task config")
    
    def log_task_info(self) -> None:
        """
        Log task information.
        """
        self.logger.info(f"Task: {self.task_name} ({self.task_type})")
        self.logger.info(f"Primary metric: {self.primary_metric}")
        self.logger.info(f"All metrics: {self.metric_names}")
        
        # Log data information
        data_info = self._get_data_info()
        for split, size in data_info.items():
            self.logger.info(f"{split.capitalize()}: {size:,} samples")
    
    def get_sample_input(self) -> Dict[str, torch.Tensor]:
        """
        Get a sample input for model testing.
        
        Returns:
            Dict[str, torch.Tensor]: Sample input
        """
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not loaded. Call load_data() first.")
        
        # Get first sample from training dataset
        sample = self.train_dataset[0]
        
        # Convert to batch format
        batch = {key: [value] if not isinstance(value, list) else value 
                for key, value in sample.items()}
        
        # Preprocess batch
        return self.preprocess_batch(batch)
    
    def compute_loss(self, 
                    model_outputs: Dict[str, torch.Tensor],
                    labels: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Args:
            model_outputs: Model outputs
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Default implementation uses model's loss if available
        if "loss" in model_outputs:
            return model_outputs["loss"]
        
        # Otherwise, compute cross-entropy loss
        if "logits" in model_outputs:
            logits = model_outputs["logits"]
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        raise ValueError("Cannot compute loss: no 'loss' or 'logits' in model outputs")
    
    def prepare_inputs_for_generation(self, 
                                    input_ids: torch.Tensor,
                                    attention_mask: Optional[torch.Tensor] = None,
                                    **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for text generation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, torch.Tensor]: Prepared inputs
        """
        inputs = {"input_ids": input_ids}
        
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        
        return inputs
    
    def postprocess_generation(self, 
                             generated_ids: torch.Tensor,
                             input_ids: torch.Tensor) -> List[str]:
        """
        Postprocess generated text.
        
        Args:
            generated_ids: Generated token IDs
            input_ids: Original input token IDs
            
        Returns:
            List[str]: Generated text strings
        """
        # Remove input tokens from generated sequence
        if generated_ids.size(1) > input_ids.size(1):
            generated_ids = generated_ids[:, input_ids.size(1):]
        
        # Decode generated tokens
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_texts
    
    def __repr__(self) -> str:
        """String representation of the task."""
        return f"{self.__class__.__name__}(name='{self.task_name}', type='{self.task_type}')"