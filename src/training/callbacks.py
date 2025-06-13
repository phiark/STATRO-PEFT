#!/usr/bin/env python3
"""
Training callbacks for STRATO-PEFT experimental framework.

This module provides a flexible callback system for monitoring and controlling
the training process, including early stopping, learning rate scheduling,
model checkpointing, and custom logging.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import time
import os
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

import torch
import numpy as np
from omegaconf import DictConfig


class CallbackEvent(Enum):
    """
    Events that can trigger callbacks during training.
    """
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    EVALUATION_START = "evaluation_start"
    EVALUATION_END = "evaluation_end"
    CHECKPOINT_SAVE = "checkpoint_save"
    EARLY_STOP = "early_stop"
    RL_UPDATE = "rl_update"  # STRATO-specific
    CONFIGURATION_CHANGE = "configuration_change"  # STRATO-specific


@dataclass
class CallbackContext:
    """
    Context information passed to callbacks.
    """
    event: CallbackEvent
    trainer: Any  # The trainer instance
    epoch: int = 0
    batch_idx: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: Dict[str, Any] = field(default_factory=dict)
    model_state: Optional[Dict[str, Any]] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


class BaseCallback(ABC):
    """
    Base class for training callbacks.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize callback.
        
        Args:
            name: Optional name for the callback
        """
        self.name = name or self.__class__.__name__
        self.enabled = True
        self.logger = logging.getLogger(f"callback.{self.name}")
    
    def on_event(self, context: CallbackContext) -> bool:
        """
        Handle a callback event.
        
        Args:
            context: Callback context with event information
            
        Returns:
            True to continue training, False to stop
        """
        if not self.enabled:
            return True
            
        try:
            return self._handle_event(context)
        except Exception as e:
            self.logger.error(f"Error in callback {self.name}: {e}")
            return True  # Continue training by default
    
    @abstractmethod
    def _handle_event(self, context: CallbackContext) -> bool:
        """
        Handle the specific event. Must be implemented by subclasses.
        
        Args:
            context: Callback context
            
        Returns:
            True to continue training, False to stop
        """
        pass
    
    def enable(self) -> None:
        """Enable the callback."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the callback."""
        self.enabled = False


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on validation metrics.
    """
    
    def __init__(
        self,
        monitor: str = 'eval_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        name: str = None
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            restore_best_weights: Whether to restore best weights when stopping
            name: Optional callback name
        """
        super().__init__(name)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None
        
        self.logger.info(
            f"EarlyStoppingCallback initialized: monitor={monitor}, "
            f"patience={patience}, mode={mode}"
        )
    
    def _handle_event(self, context: CallbackContext) -> bool:
        """
        Handle callback events for early stopping.
        
        Args:
            context: Callback context
            
        Returns:
            True to continue training, False to stop
        """
        if context.event == CallbackEvent.EVALUATION_END:
            return self._check_early_stopping(context)
        elif context.event == CallbackEvent.EARLY_STOP and self.restore_best_weights:
            self._restore_best_weights(context)
        
        return True
    
    def _check_early_stopping(self, context: CallbackContext) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            context: Callback context
            
        Returns:
            True to continue training, False to stop
        """
        if self.monitor not in context.metrics:
            self.logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return True
        
        current_value = context.metrics[self.monitor]
        
        # Check if this is an improvement
        if self.mode == 'min':
            is_improvement = current_value < (self.best_value - self.min_delta)
        else:
            is_improvement = current_value > (self.best_value + self.min_delta)
        
        if is_improvement:
            self.best_value = current_value
            self.best_epoch = context.epoch
            self.wait = 0
            
            # Save best weights if requested
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in context.trainer.model.state_dict().items()}
            
            self.logger.info(
                f"New best {self.monitor}: {current_value:.6f} at epoch {context.epoch}"
            )
        else:
            self.wait += 1
            self.logger.debug(
                f"No improvement in {self.monitor}: {current_value:.6f} "
                f"(best: {self.best_value:.6f}, wait: {self.wait}/{self.patience})"
            )
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.logger.info(
                f"Early stopping triggered after {self.patience} epochs without improvement"
            )
            return False
        
        return True
    
    def _restore_best_weights(self, context: CallbackContext) -> None:
        """
        Restore the best weights.
        
        Args:
            context: Callback context
        """
        if self.best_weights is not None:
            context.trainer.model.load_state_dict(self.best_weights)
            self.logger.info(f"Restored best weights from epoch {self.best_epoch}")


class ModelCheckpointCallback(BaseCallback):
    """
    Model checkpointing callback.
    """
    
    def __init__(
        self,
        output_dir: str,
        monitor: str = 'eval_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_frequency: int = 1,
        max_checkpoints: int = 5,
        name: str = None
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            output_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for the monitor metric
            save_best_only: Whether to only save the best model
            save_frequency: Frequency of saving (in epochs)
            max_checkpoints: Maximum number of checkpoints to keep
            name: Optional callback name
        """
        super().__init__(name)
        self.output_dir = output_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_files = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(
            f"ModelCheckpointCallback initialized: output_dir={output_dir}, "
            f"monitor={monitor}, save_best_only={save_best_only}"
        )
    
    def _handle_event(self, context: CallbackContext) -> bool:
        """
        Handle callback events for model checkpointing.
        
        Args:
            context: Callback context
            
        Returns:
            True to continue training
        """
        if context.event == CallbackEvent.EPOCH_END:
            self._maybe_save_checkpoint(context)
        elif context.event == CallbackEvent.TRAINING_END:
            self._save_final_checkpoint(context)
        
        return True
    
    def _maybe_save_checkpoint(self, context: CallbackContext) -> None:
        """
        Maybe save a checkpoint based on criteria.
        
        Args:
            context: Callback context
        """
        # Check save frequency
        if (context.epoch + 1) % self.save_frequency != 0:
            return
        
        should_save = True
        
        if self.save_best_only and self.monitor in context.metrics:
            current_value = context.metrics[self.monitor]
            
            if self.mode == 'min':
                is_best = current_value < self.best_value
            else:
                is_best = current_value > self.best_value
            
            if is_best:
                self.best_value = current_value
                should_save = True
                self.logger.info(f"New best {self.monitor}: {current_value:.6f}")
            else:
                should_save = False
        
        if should_save:
            self._save_checkpoint(context)
    
    def _save_checkpoint(self, context: CallbackContext) -> None:
        """
        Save a checkpoint.
        
        Args:
            context: Callback context
        """
        checkpoint_name = f"checkpoint-epoch-{context.epoch+1}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        # Save using trainer's save method
        context.trainer.save_checkpoint(checkpoint_path)
        
        # Track checkpoint files
        self.checkpoint_files.append(checkpoint_path)
        
        # Remove old checkpoints if needed
        if len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if os.path.exists(old_checkpoint):
                import shutil
                shutil.rmtree(old_checkpoint, ignore_errors=True)
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_checkpoint(self, context: CallbackContext) -> None:
        """
        Save final checkpoint at the end of training.
        
        Args:
            context: Callback context
        """
        final_path = os.path.join(self.output_dir, "final-checkpoint")
        context.trainer.save_checkpoint(final_path)
        self.logger.info(f"Final checkpoint saved: {final_path}")


class MetricsLoggerCallback(BaseCallback):
    """
    Callback for logging training metrics.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_frequency: int = 1,
        include_system_metrics: bool = True,
        name: str = None
    ):
        """
        Initialize metrics logger callback.
        
        Args:
            log_file: Optional file to log metrics to
            log_frequency: Frequency of logging (in epochs)
            include_system_metrics: Whether to include system metrics
            name: Optional callback name
        """
        super().__init__(name)
        self.log_file = log_file
        self.log_frequency = log_frequency
        self.include_system_metrics = include_system_metrics
        
        self.metrics_history = []
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        self.logger.info(f"MetricsLoggerCallback initialized: log_file={log_file}")
    
    def _handle_event(self, context: CallbackContext) -> bool:
        """
        Handle callback events for metrics logging.
        
        Args:
            context: Callback context
            
        Returns:
            True to continue training
        """
        if context.event == CallbackEvent.EPOCH_END:
            self._log_epoch_metrics(context)
        elif context.event == CallbackEvent.TRAINING_END:
            self._save_metrics_history()
        
        return True
    
    def _log_epoch_metrics(self, context: CallbackContext) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            context: Callback context
        """
        if (context.epoch + 1) % self.log_frequency != 0:
            return
        
        # Collect metrics
        epoch_metrics = {
            'epoch': context.epoch,
            'timestamp': time.time(),
            **context.metrics
        }
        
        # Add system metrics if requested
        if self.include_system_metrics:
            epoch_metrics.update(self._get_system_metrics(context))
        
        # Store in history
        self.metrics_history.append(epoch_metrics)
        
        # Log to file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(epoch_metrics) + '\n')
        
        # Log key metrics
        key_metrics = {k: v for k, v in epoch_metrics.items() 
                      if k in ['train_loss', 'eval_loss', 'accuracy', 'f1']}
        self.logger.info(f"Epoch {context.epoch}: {key_metrics}")
    
    def _get_system_metrics(self, context: CallbackContext) -> Dict[str, float]:
        """
        Get system metrics like memory usage.
        
        Args:
            context: Callback context
            
        Returns:
            System metrics dictionary
        """
        metrics = {}
        
        # GPU memory if available
        if torch.cuda.is_available():
            metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # Model parameters
        if hasattr(context.trainer, 'model'):
            total_params = sum(p.numel() for p in context.trainer.model.parameters())
            trainable_params = sum(p.numel() for p in context.trainer.model.parameters() if p.requires_grad)
            metrics['total_parameters'] = total_params
            metrics['trainable_parameters'] = trainable_params
            metrics['trainable_ratio'] = trainable_params / total_params if total_params > 0 else 0.0
        
        return metrics
    
    def _save_metrics_history(self) -> None:
        """
        Save complete metrics history.
        """
        if self.log_file and self.metrics_history:
            history_file = self.log_file.replace('.jsonl', '_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            self.logger.info(f"Metrics history saved: {history_file}")


class StratoCallback(BaseCallback):
    """
    STRATO-specific callback for monitoring RL training.
    """
    
    def __init__(
        self,
        log_configurations: bool = True,
        log_rewards: bool = True,
        log_policy_updates: bool = True,
        name: str = None
    ):
        """
        Initialize STRATO callback.
        
        Args:
            log_configurations: Whether to log configuration changes
            log_rewards: Whether to log RL rewards
            log_policy_updates: Whether to log policy updates
            name: Optional callback name
        """
        super().__init__(name)
        self.log_configurations = log_configurations
        self.log_rewards = log_rewards
        self.log_policy_updates = log_policy_updates
        
        self.configuration_history = []
        self.reward_history = []
        
        self.logger.info("StratoCallback initialized")
    
    def _handle_event(self, context: CallbackContext) -> bool:
        """
        Handle STRATO-specific events.
        
        Args:
            context: Callback context
            
        Returns:
            True to continue training
        """
        if context.event == CallbackEvent.CONFIGURATION_CHANGE and self.log_configurations:
            self._log_configuration_change(context)
        elif context.event == CallbackEvent.RL_UPDATE and self.log_policy_updates:
            self._log_rl_update(context)
        elif context.event == CallbackEvent.EPOCH_END and self.log_rewards:
            self._log_rewards(context)
        
        return True
    
    def _log_configuration_change(self, context: CallbackContext) -> None:
        """
        Log configuration changes.
        
        Args:
            context: Callback context
        """
        if 'configuration' in context.extra_data:
            config = context.extra_data['configuration']
            self.configuration_history.append({
                'epoch': context.epoch,
                'configuration': config,
                'timestamp': time.time()
            })
            
            self.logger.info(f"Configuration changed: {config}")
    
    def _log_rl_update(self, context: CallbackContext) -> None:
        """
        Log RL policy updates.
        
        Args:
            context: Callback context
        """
        if 'policy_metrics' in context.extra_data:
            metrics = context.extra_data['policy_metrics']
            self.logger.info(f"RL policy updated: {metrics}")
    
    def _log_rewards(self, context: CallbackContext) -> None:
        """
        Log RL rewards.
        
        Args:
            context: Callback context
        """
        if 'rl_reward' in context.metrics:
            reward = context.metrics['rl_reward']
            self.reward_history.append({
                'epoch': context.epoch,
                'reward': reward,
                'timestamp': time.time()
            })
            
            # Log reward statistics
            if len(self.reward_history) >= 5:
                recent_rewards = [r['reward'] for r in self.reward_history[-5:]]
                avg_reward = np.mean(recent_rewards)
                self.logger.info(f"Recent average reward: {avg_reward:.4f}")


class TrainingCallbacks:
    """
    Manager for training callbacks.
    """
    
    def __init__(self):
        """
        Initialize callback manager.
        """
        self.callbacks: List[BaseCallback] = []
        self.logger = logging.getLogger("callbacks")
    
    def add_callback(self, callback: BaseCallback) -> None:
        """
        Add a callback.
        
        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)
        self.logger.info(f"Added callback: {callback.name}")
    
    def remove_callback(self, callback_name: str) -> bool:
        """
        Remove a callback by name.
        
        Args:
            callback_name: Name of callback to remove
            
        Returns:
            True if callback was removed, False if not found
        """
        for i, callback in enumerate(self.callbacks):
            if callback.name == callback_name:
                removed = self.callbacks.pop(i)
                self.logger.info(f"Removed callback: {removed.name}")
                return True
        return False
    
    def trigger_event(self, context: CallbackContext) -> bool:
        """
        Trigger an event for all callbacks.
        
        Args:
            context: Callback context
            
        Returns:
            True to continue training, False to stop
        """
        for callback in self.callbacks:
            if not callback.on_event(context):
                self.logger.info(f"Training stopped by callback: {callback.name}")
                return False
        return True
    
    def enable_callback(self, callback_name: str) -> bool:
        """
        Enable a callback by name.
        
        Args:
            callback_name: Name of callback to enable
            
        Returns:
            True if callback was found and enabled
        """
        for callback in self.callbacks:
            if callback.name == callback_name:
                callback.enable()
                return True
        return False
    
    def disable_callback(self, callback_name: str) -> bool:
        """
        Disable a callback by name.
        
        Args:
            callback_name: Name of callback to disable
            
        Returns:
            True if callback was found and disabled
        """
        for callback in self.callbacks:
            if callback.name == callback_name:
                callback.disable()
                return True
        return False
    
    def get_callback(self, callback_name: str) -> Optional[BaseCallback]:
        """
        Get a callback by name.
        
        Args:
            callback_name: Name of callback to get
            
        Returns:
            Callback instance or None if not found
        """
        for callback in self.callbacks:
            if callback.name == callback_name:
                return callback
        return None
    
    def list_callbacks(self) -> List[str]:
        """
        List all callback names.
        
        Returns:
            List of callback names
        """
        return [callback.name for callback in self.callbacks]