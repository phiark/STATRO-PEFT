#!/usr/bin/env python3
"""
Base trainer abstract class for STRATO-PEFT experimental framework.

This module defines the abstract base class for all trainers, providing
a common interface and shared functionality for training different types
of models with various PEFT methods.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.reproducibility_utils import ReproducibilityManager
from ..tasks.base_task import BaseTask, TaskMetrics
from ..peft.base_peft import BasePEFT, PEFTMetrics


@dataclass
class TrainingMetrics:
    """Data structure for tracking training metrics."""
    
    # Basic training metrics
    epoch: int = 0
    step: int = 0
    learning_rate: float = 0.0
    
    # Loss metrics
    train_loss: float = 0.0
    eval_loss: float = 0.0
    best_eval_loss: float = float('inf')
    
    # Task-specific metrics
    task_metrics: Optional[TaskMetrics] = None
    
    # PEFT metrics
    peft_metrics: Optional[PEFTMetrics] = None
    
    # Performance metrics
    training_time: float = 0.0
    evaluation_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Convergence metrics
    gradient_norm: float = 0.0
    parameter_norm: float = 0.0
    
    # History tracking
    loss_history: List[float] = field(default_factory=list)
    eval_history: List[Dict[str, float]] = field(default_factory=list)
    
    def update_loss_history(self, loss: float) -> None:
        """Update loss history."""
        self.loss_history.append(loss)
        
    def update_eval_history(self, metrics: Dict[str, float]) -> None:
        """Update evaluation history."""
        self.eval_history.append(metrics.copy())
        
    def get_recent_loss_trend(self, window: int = 10) -> float:
        """Get recent loss trend (negative means decreasing)."""
        if len(self.loss_history) < window:
            return 0.0
        
        recent_losses = self.loss_history[-window:]
        if len(recent_losses) < 2:
            return 0.0
            
        # Simple linear trend
        x = list(range(len(recent_losses)))
        y = recent_losses
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
        
    def is_improving(self, patience: int = 5) -> bool:
        """Check if model is improving based on evaluation history."""
        if len(self.eval_history) < patience:
            return True
            
        # Check if recent evaluations show improvement
        recent_evals = self.eval_history[-patience:]
        if not recent_evals:
            return True
            
        # Use primary metric if available
        metric_key = 'eval_loss'  # Default metric
        if recent_evals[0] and len(recent_evals[0]) > 0:
            # Try to find a primary metric
            for key in ['accuracy', 'f1', 'exact_match']:
                if key in recent_evals[0]:
                    metric_key = key
                    break
        
        if metric_key not in recent_evals[0]:
            return True
            
        # Check for improvement
        best_value = recent_evals[0][metric_key]
        for eval_metrics in recent_evals[1:]:
            if metric_key in eval_metrics:
                if metric_key == 'eval_loss':
                    # Lower is better for loss
                    if eval_metrics[metric_key] < best_value:
                        return True
                else:
                    # Higher is better for other metrics
                    if eval_metrics[metric_key] > best_value:
                        return True
                        
        return False


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: BaseTask,
        peft_method: Optional[BasePEFT],
        config: DictConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize base trainer.
        
        Args:
            model: The pre-trained model to train
            tokenizer: The tokenizer for the model
            task: The task to train on
            peft_method: Optional PEFT method to apply
            config: Training configuration
            logger: Optional logger instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.peft_method = peft_method
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Training state
        self.metrics = TrainingMetrics()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_training = False
        
        # Reproducibility
        self.reproducibility_manager = ReproducibilityManager(
            seed=config.get('seed', 42),
            logger=self.logger
        )
        
        # Training components (to be initialized)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None
        
        # Callbacks
        self.callbacks: List[Any] = []
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        
    @abstractmethod
    def setup_training(self) -> None:
        """
        Setup training components (optimizer, scheduler, data loaders, etc.).
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics for the epoch
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Optional dataloader for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def train(
        self,
        num_epochs: int,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> TrainingMetrics:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            eval_steps: Steps between evaluations (if None, evaluate each epoch)
            save_steps: Steps between checkpoints (if None, save each epoch)
            output_dir: Directory to save checkpoints
            
        Returns:
            Final training metrics
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        # Setup training
        self.setup_training()
        
        # Apply PEFT if specified
        if self.peft_method:
            self.logger.info("Applying PEFT method...")
            self.model = self.peft_method.apply_peft()
            
        # Move model to device
        self.model.to(self.device)
        
        # Set training flag
        self.is_training = True
        
        # Training loop
        start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                self.metrics.epoch = epoch
                
                # Train epoch
                epoch_metrics = self.train_epoch()
                self.metrics.train_loss = epoch_metrics.get('train_loss', 0.0)
                self.metrics.update_loss_history(self.metrics.train_loss)
                
                # Evaluate if needed
                if eval_steps is None or (epoch + 1) % eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.metrics.eval_loss = eval_metrics.get('eval_loss', 0.0)
                    self.metrics.update_eval_history(eval_metrics)
                    
                    # Update best metrics
                    if self.metrics.eval_loss < self.metrics.best_eval_loss:
                        self.metrics.best_eval_loss = self.metrics.eval_loss
                        
                        # Save best model
                        if output_dir:
                            self.save_checkpoint(os.path.join(output_dir, 'best_model'))
                
                # Save checkpoint if needed
                if save_steps and (epoch + 1) % save_steps == 0 and output_dir:
                    checkpoint_path = os.path.join(output_dir, f'checkpoint-epoch-{epoch+1}')
                    self.save_checkpoint(checkpoint_path)
                
                # Log progress
                self._log_epoch_progress(epoch, epoch_metrics)
                
                # Check for early stopping
                if self._should_early_stop():
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.is_training = False
            
        # Calculate total training time
        self.metrics.training_time = time.time() - start_time
        
        # Final evaluation
        if self.eval_dataloader:
            final_eval = self.evaluate()
            self.metrics.update_eval_history(final_eval)
            
        # Update PEFT metrics if applicable
        if self.peft_method:
            self.metrics.peft_metrics = self.peft_method.get_peft_metrics()
            
        self.logger.info(f"Training completed in {self.metrics.training_time:.2f} seconds")
        return self.metrics
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model state
        if self.peft_method and self.peft_method.is_applied:
            # Save PEFT weights
            peft_path = os.path.join(checkpoint_path, 'peft_weights.pt')
            self.peft_method.save_peft_weights(peft_path)
        else:
            # Save full model
            model_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
            torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer state
        if self.optimizer:
            optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
            torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        if self.scheduler:
            scheduler_path = os.path.join(checkpoint_path, 'scheduler.pt')
            torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save training metrics
        metrics_path = os.path.join(checkpoint_path, 'training_metrics.pt')
        torch.save(self.metrics, metrics_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to load checkpoint from
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load PEFT weights if available
        peft_path = os.path.join(checkpoint_path, 'peft_weights.pt')
        if os.path.exists(peft_path) and self.peft_method:
            self.peft_method.load_peft_weights(peft_path)
        else:
            # Load full model
            model_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
        
        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        if os.path.exists(optimizer_path) and self.optimizer:
            optimizer_state = torch.load(optimizer_path, map_location='cpu')
            self.optimizer.load_state_dict(optimizer_state)
        
        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_path, 'scheduler.pt')
        if os.path.exists(scheduler_path) and self.scheduler:
            scheduler_state = torch.load(scheduler_path, map_location='cpu')
            self.scheduler.load_state_dict(scheduler_state)
        
        # Load training metrics
        metrics_path = os.path.join(checkpoint_path, 'training_metrics.pt')
        if os.path.exists(metrics_path):
            self.metrics = torch.load(metrics_path, map_location='cpu')
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_parameter_count(self) -> Tuple[int, int, float]:
        """
        Get parameter count statistics.
        
        Returns:
            Tuple of (trainable_params, total_params, trainable_ratio)
        """
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
        
        return trainable_params, total_params, trainable_ratio
    
    def _log_epoch_progress(self, epoch: int, epoch_metrics: Dict[str, float]) -> None:
        """
        Log progress for an epoch.
        
        Args:
            epoch: Current epoch number
            epoch_metrics: Metrics for the epoch
        """
        # Basic metrics
        log_msg = f"Epoch {epoch+1}: "
        log_msg += f"train_loss={epoch_metrics.get('train_loss', 0.0):.4f}"
        
        if 'eval_loss' in epoch_metrics:
            log_msg += f", eval_loss={epoch_metrics['eval_loss']:.4f}"
        
        # Add task-specific metrics
        for key, value in epoch_metrics.items():
            if key not in ['train_loss', 'eval_loss'] and isinstance(value, (int, float)):
                log_msg += f", {key}={value:.4f}"
        
        # Add memory usage
        memory_usage = self.get_memory_usage()
        if memory_usage > 0:
            log_msg += f", memory={memory_usage:.1f}MB"
        
        self.logger.info(log_msg)
    
    def _should_early_stop(self) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if training should stop early
        """
        patience = self.config.get('early_stopping_patience', 0)
        if patience <= 0:
            return False
            
        return not self.metrics.is_improving(patience=patience)
    
    def add_callback(self, callback: Any) -> None:
        """
        Add a training callback.
        
        Args:
            callback: Callback object
        """
        self.callbacks.append(callback)
    
    def _call_callbacks(self, event: str, **kwargs) -> None:
        """
        Call all registered callbacks for an event.
        
        Args:
            event: Event name
            **kwargs: Event arguments
        """
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)