#!/usr/bin/env python3
"""
PEFT trainer implementation for STRATO-PEFT experimental framework.

This module provides a specialized trainer for Parameter-Efficient Fine-Tuning (PEFT)
methods, including optimized training loops, gradient accumulation, and PEFT-specific
monitoring and evaluation.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import time
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..tasks.base_task import BaseTask
from ..peft.base_peft import BasePEFT


class PEFTTrainer(BaseTrainer):
    """
    Specialized trainer for PEFT methods.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: BaseTask,
        peft_method: BasePEFT,
        config: DictConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PEFT trainer.
        
        Args:
            model: The pre-trained model to train
            tokenizer: The tokenizer for the model
            task: The task to train on
            peft_method: PEFT method to apply
            config: Training configuration
            logger: Optional logger instance
        """
        super().__init__(model, tokenizer, task, peft_method, config, logger)
        
        # PEFT-specific configuration
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.warmup_ratio = config.get('warmup_ratio', 0.1)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.learning_rate = config.get('learning_rate', 5e-4)
        
        # Training optimization
        self.fp16 = config.get('fp16', False)
        self.dataloader_num_workers = config.get('dataloader_num_workers', 0)
        self.dataloader_pin_memory = config.get('dataloader_pin_memory', True)
        
        # Evaluation configuration
        self.eval_accumulation_steps = config.get('eval_accumulation_steps', None)
        self.metric_for_best_model = config.get('metric_for_best_model', 'eval_loss')
        self.greater_is_better = config.get('greater_is_better', False)
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        self.logger.info(f"PEFT Trainer initialized with {peft_method.__class__.__name__}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        self.logger.info(f"Mixed precision (FP16): {self.fp16}")
    
    def setup_training(self) -> None:
        """
        Setup training components for PEFT training.
        """
        self.logger.info("Setting up PEFT training components...")
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Apply PEFT method
        if self.peft_method and not self.peft_method.is_applied:
            self.logger.info("Applying PEFT method...")
            self.model = self.peft_method.apply_peft()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Log training setup
        self._log_training_setup()
    
    def _setup_data_loaders(self) -> None:
        """
        Setup training and evaluation data loaders.
        """
        batch_size = self.config.get('batch_size', 8)
        eval_batch_size = self.config.get('eval_batch_size', batch_size)
        
        # Create data loaders
        self.train_dataloader = self.task.create_dataloader(
            split='train',
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory
        )
        
        self.eval_dataloader = self.task.create_dataloader(
            split='validation',
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory
        )
        
        self.logger.info(f"Train dataloader: {len(self.train_dataloader)} batches")
        self.logger.info(f"Eval dataloader: {len(self.eval_dataloader)} batches")
    
    def _setup_optimizer(self) -> None:
        """
        Setup optimizer for PEFT training.
        """
        # Get trainable parameters from PEFT method
        if self.peft_method:
            trainable_params = self.peft_method.get_trainable_parameters()
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise ValueError("No trainable parameters found")
        
        # Create optimizer
        optimizer_type = self.config.get('optimizer', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.config.get('adam_betas', (0.9, 0.999)),
                eps=self.config.get('adam_epsilon', 1e-8)
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.config.get('sgd_momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        num_params = sum(p.numel() for p in trainable_params)
        self.logger.info(f"Optimizer: {optimizer_type}, Trainable parameters: {num_params:,}")
    
    def _setup_scheduler(self) -> None:
        """
        Setup learning rate scheduler.
        """
        if not self.train_dataloader:
            return
        
        num_epochs = self.config.get('num_epochs', 3)
        total_steps = len(self.train_dataloader) * num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        scheduler_type = self.config.get('scheduler', 'linear')
        
        if scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None
        
        if self.scheduler:
            self.logger.info(
                f"Scheduler: {scheduler_type}, Total steps: {total_steps}, "
                f"Warmup steps: {warmup_steps}"
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics for the epoch
        """
        if not self.train_dataloader:
            raise RuntimeError("Training dataloader not initialized")
        
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        gradient_norm_sum = 0.0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.metrics.epoch + 1}",
            disable=not self.config.get('show_progress', True)
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss = self._compute_loss(batch)
                
                # Scale loss for gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update metrics
                gradient_norm_sum += grad_norm.item()
                self.metrics.step += 1
            
            # Accumulate loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Update learning rate in metrics
            self.metrics.learning_rate = current_lr
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = gradient_norm_sum / max(num_batches // self.gradient_accumulation_steps, 1)
        
        # Update metrics
        self.metrics.gradient_norm = avg_grad_norm
        self.metrics.memory_usage_mb = self.get_memory_usage()
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.metrics.learning_rate,
            'gradient_norm': avg_grad_norm,
            'memory_usage_mb': self.metrics.memory_usage_mb
        }
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Optional dataloader for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        eval_dataloader = dataloader or self.eval_dataloader
        if not eval_dataloader:
            self.logger.warning("No evaluation dataloader available")
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(
                eval_dataloader,
                desc="Evaluating",
                disable=not self.config.get('show_progress', True)
            ):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    loss = self._compute_loss(batch)
                    outputs = self._get_model_outputs(batch)
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels for task evaluation
                predictions, labels = self._extract_predictions_and_labels(batch, outputs)
                if predictions is not None:
                    all_predictions.extend(predictions)
                if labels is not None:
                    all_labels.extend(labels)
        
        # Calculate evaluation metrics
        eval_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {'eval_loss': avg_loss}
        
        # Task-specific evaluation
        if all_predictions and all_labels:
            task_metrics = self.task.evaluate(all_predictions, all_labels)
            if task_metrics:
                metrics.update(task_metrics.to_dict())
        
        # Update metrics
        self.metrics.evaluation_time = eval_time
        
        return metrics
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss tensor
        """
        outputs = self.model(**batch)
        
        if hasattr(outputs, 'loss'):
            return outputs.loss
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        else:
            raise ValueError("Could not extract loss from model outputs")
    
    def _get_model_outputs(self, batch: Dict[str, Any]) -> Any:
        """
        Get model outputs for a batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Model outputs
        """
        return self.model(**batch)
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch to the appropriate device.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch moved to device
        """
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _extract_predictions_and_labels(self, batch: Dict[str, Any], outputs: Any) -> tuple:
        """
        Extract predictions and labels from batch and outputs.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            
        Returns:
            Tuple of (predictions, labels)
        """
        # This is task-specific and should be implemented based on the task
        # For now, return None to indicate no extraction
        return None, None
    
    def _log_training_setup(self) -> None:
        """
        Log training setup information.
        """
        if self.peft_method:
            peft_metrics = self.peft_method.get_peft_metrics()
            self.logger.info(f"PEFT Method: {self.peft_method.__class__.__name__}")
            self.logger.info(f"Trainable parameters: {peft_metrics.trainable_params:,}")
            self.logger.info(f"Total parameters: {peft_metrics.total_params:,}")
            self.logger.info(f"Trainable ratio: {peft_metrics.trainable_ratio:.4f}")
        
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        self.logger.info(f"Max gradient norm: {self.max_grad_norm}")