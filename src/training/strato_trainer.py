#!/usr/bin/env python3
"""
STRATO trainer implementation for STRATO-PEFT experimental framework.

This module provides a specialized trainer for STRATO-PEFT that integrates
reinforcement learning for strategic parameter allocation with traditional
fine-tuning, implementing the dual-loop optimization described in the paper.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import numpy as np

from .peft_trainer import PEFTTrainer
from ..tasks.base_task import BaseTask
from ..peft.strato_peft import (
    StratoPEFT, 
    StratoState, 
    StratoAction, 
    EpisodeMemory,
    LayerSensitivity
)


class StratoTrainer(PEFTTrainer):
    """
    Specialized trainer for STRATO-PEFT with RL integration.
    
    Implements the dual-loop optimization:
    1. Inner loop: Traditional SGD for model parameters
    2. Outer loop: RL policy updates for strategic parameter allocation
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: BaseTask,
        strato_peft: StratoPEFT,
        config: DictConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize STRATO trainer.
        
        Args:
            model: The pre-trained model to train
            tokenizer: The tokenizer for the model
            task: The task to train on
            strato_peft: STRATO-PEFT method instance
            config: Training configuration
            logger: Optional logger instance
        """
        super().__init__(model, tokenizer, task, strato_peft, config, logger)
        
        # STRATO-specific configuration
        self.strato_config = config.get('strato', {})
        self.rl_update_frequency = self.strato_config.get('rl_update_frequency', 5)
        self.exploration_episodes = self.strato_config.get('exploration_episodes', 10)
        self.sensitivity_analysis_samples = self.strato_config.get('sensitivity_analysis_samples', 100)
        self.budget_constraint = self.strato_config.get('budget_constraint', 0.1)  # 10% of full model
        
        # RL training state
        self.episode_buffer: List[Dict[str, Any]] = []
        self.current_episode_id = 0
        self.exploration_phase = True
        self.best_configuration: Optional[Dict[str, int]] = None
        self.best_reward = float('-inf')
        
        # Performance tracking
        self.configuration_history: List[Dict[str, Any]] = []
        self.reward_history: List[float] = []
        self.cost_history: List[float] = []
        
        # Sensitivity analysis cache
        self.layer_sensitivities: Dict[str, LayerSensitivity] = {}
        self.sensitivity_computed = False
        
        self.logger.info("STRATO Trainer initialized")
        self.logger.info(f"RL update frequency: {self.rl_update_frequency}")
        self.logger.info(f"Exploration episodes: {self.exploration_episodes}")
        self.logger.info(f"Budget constraint: {self.budget_constraint}")
    
    def setup_training(self) -> None:
        """
        Setup training components for STRATO training.
        """
        super().setup_training()
        
        # Perform initial sensitivity analysis
        self._perform_sensitivity_analysis()
        
        # Initialize RL components
        self._initialize_rl_components()
        
        self.logger.info("STRATO training setup completed")
    
    def train(
        self,
        num_epochs: int,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main STRATO training loop with dual optimization.
        
        Args:
            num_epochs: Number of epochs to train
            eval_steps: Steps between evaluations
            save_steps: Steps between checkpoints
            output_dir: Directory to save checkpoints
            
        Returns:
            Training results including RL metrics
        """
        self.logger.info(f"Starting STRATO training for {num_epochs} epochs")
        
        # Setup training
        self.setup_training()
        
        # Training loop with RL exploration
        start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                self.metrics.epoch = epoch
                
                # Exploration phase: try different configurations
                if self.exploration_phase:
                    epoch_results = self._exploration_epoch()
                else:
                    # Exploitation phase: use best configuration
                    epoch_results = self._exploitation_epoch()
                
                # Update RL policy if needed
                if (epoch + 1) % self.rl_update_frequency == 0:
                    self._update_rl_policy()
                
                # Evaluate current configuration
                if eval_steps is None or (epoch + 1) % eval_steps == 0:
                    eval_metrics = self.evaluate()
                    epoch_results.update(eval_metrics)
                    
                    # Update best configuration
                    self._update_best_configuration(eval_metrics)
                
                # Save checkpoint if needed
                if save_steps and (epoch + 1) % save_steps == 0 and output_dir:
                    self._save_strato_checkpoint(output_dir, epoch)
                
                # Log progress
                self._log_strato_progress(epoch, epoch_results)
                
                # Check phase transition
                if self.exploration_phase and epoch >= self.exploration_episodes:
                    self._transition_to_exploitation()
                
                # Early stopping check
                if self._should_early_stop():
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("STRATO training interrupted by user")
        except Exception as e:
            self.logger.error(f"STRATO training failed: {e}")
            raise
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Final evaluation with best configuration
        if self.best_configuration:
            self._apply_configuration(self.best_configuration)
            final_eval = self.evaluate()
        else:
            final_eval = {}
        
        # Compile results
        results = {
            'training_time': total_time,
            'best_configuration': self.best_configuration,
            'best_reward': self.best_reward,
            'final_evaluation': final_eval,
            'configuration_history': self.configuration_history,
            'reward_history': self.reward_history,
            'cost_history': self.cost_history,
            'layer_sensitivities': {k: v.__dict__ for k, v in self.layer_sensitivities.items()}
        }
        
        self.logger.info(f"STRATO training completed in {total_time:.2f} seconds")
        return results
    
    def _perform_sensitivity_analysis(self) -> None:
        """
        Perform layer sensitivity analysis using the mapping builder.
        """
        if self.sensitivity_computed:
            return
            
        self.logger.info("Performing layer sensitivity analysis...")
        
        # Get a small sample of data for analysis
        analysis_dataloader = self.task.create_dataloader(
            split='train',
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=0
        )
        
        # Perform sensitivity analysis
        self.layer_sensitivities = self.peft_method.mapping_builder.analyze_layer_sensitivity(
            dataloader=analysis_dataloader,
            target_modules=self.peft_method.target_modules,
            num_samples=self.sensitivity_analysis_samples
        )
        
        self.sensitivity_computed = True
        
        # Log sensitivity results
        self.logger.info(f"Analyzed {len(self.layer_sensitivities)} layers")
        top_layers = sorted(
            self.layer_sensitivities.items(),
            key=lambda x: x[1].marginal_utility,
            reverse=True
        )[:5]
        
        self.logger.info("Top 5 layers by marginal utility:")
        for name, sensitivity in top_layers:
            self.logger.info(
                f"  {name}: utility={sensitivity.marginal_utility:.4f}, "
                f"sensitivity={sensitivity.sensitivity_score:.4f}"
            )
    
    def _initialize_rl_components(self) -> None:
        """
        Initialize RL components for STRATO training.
        """
        # RL components are already initialized in StratoPEFT
        # Here we just set up the training state
        self.current_episode_id = 0
        self.episode_buffer = []
        
        self.logger.info("RL components initialized")
    
    def _exploration_epoch(self) -> Dict[str, float]:
        """
        Run one exploration epoch with RL policy.
        
        Returns:
            Epoch metrics
        """
        # Generate new configuration using RL policy
        configuration = self._generate_rl_configuration()
        
        # Apply configuration
        self._apply_configuration(configuration)
        
        # Train with this configuration
        epoch_metrics = self.train_epoch()
        
        # Evaluate configuration
        eval_metrics = self.evaluate()
        reward = self._calculate_reward(eval_metrics, configuration)
        cost = self._calculate_cost(configuration)
        
        # Store episode in buffer
        episode = {
            'episode_id': self.current_episode_id,
            'configuration': configuration.copy(),
            'reward': reward,
            'cost': cost,
            'metrics': eval_metrics.copy(),
            'epoch': self.metrics.epoch
        }
        
        self.episode_buffer.append(episode)
        self.configuration_history.append(episode)
        self.reward_history.append(reward)
        self.cost_history.append(cost)
        
        # Add to memory cache
        memory_episode = EpisodeMemory(
            configuration=configuration,
            reward=reward,
            cost=cost,
            validation_score=eval_metrics.get('eval_loss', 0.0),
            episode_id=str(self.current_episode_id),
            timestamp=time.time()
        )
        self.peft_method.memory_cache.add_episode(memory_episode)
        
        self.current_episode_id += 1
        
        # Combine metrics
        combined_metrics = epoch_metrics.copy()
        combined_metrics.update(eval_metrics)
        combined_metrics.update({
            'rl_reward': reward,
            'rl_cost': cost,
            'episode_id': self.current_episode_id - 1
        })
        
        return combined_metrics
    
    def _exploitation_epoch(self) -> Dict[str, float]:
        """
        Run one exploitation epoch with best configuration.
        
        Returns:
            Epoch metrics
        """
        if self.best_configuration:
            self._apply_configuration(self.best_configuration)
        
        # Standard training epoch
        epoch_metrics = self.train_epoch()
        
        return epoch_metrics
    
    def _generate_rl_configuration(self) -> Dict[str, int]:
        """
        Generate a new configuration using RL policy.
        
        Returns:
            Configuration dictionary mapping layer names to ranks
        """
        # Create current state
        state = StratoState(
            current_layer=0,
            remaining_budget=self.budget_constraint,
            validation_reward_estimate=np.mean(self.reward_history[-5:]) if self.reward_history else 0.0,
            layer_sensitivities=[s.marginal_utility for s in self.layer_sensitivities.values()],
            current_configuration={}
        )
        
        configuration = {}
        remaining_budget = self.budget_constraint
        
        # Generate actions for each layer
        for layer_name, sensitivity in self.layer_sensitivities.items():
            if remaining_budget <= 0:
                break
                
            # Update state
            state.current_layer += 1
            state.remaining_budget = remaining_budget
            
            # Get action from policy
            action = self.peft_method.policy_agent.select_action(state)
            
            # Apply action
            if action.action_type == 'insert' and action.rank:
                # Check budget constraint
                estimated_cost = self._estimate_rank_cost(layer_name, action.rank)
                if estimated_cost <= remaining_budget:
                    configuration[layer_name] = action.rank
                    remaining_budget -= estimated_cost
            # Skip 'freeze' and 'skip' actions for now
        
        # Ensure at least some layers are adapted
        if not configuration and self.layer_sensitivities:
            # Add the most sensitive layer with minimum rank
            best_layer = max(self.layer_sensitivities.items(), key=lambda x: x[1].marginal_utility)
            configuration[best_layer[0]] = 4  # Minimum rank
        
        return configuration
    
    def _apply_configuration(self, configuration: Dict[str, int]) -> None:
        """
        Apply a configuration to the model.
        
        Args:
            configuration: Configuration to apply
        """
        # For now, this is a simplified implementation
        # In practice, this would dynamically modify the model architecture
        self.peft_method.current_configuration = configuration.copy()
        
        self.logger.debug(f"Applied configuration: {configuration}")
    
    def _calculate_reward(self, eval_metrics: Dict[str, float], configuration: Dict[str, int]) -> float:
        """
        Calculate reward for a configuration.
        
        Args:
            eval_metrics: Evaluation metrics
            configuration: Configuration used
            
        Returns:
            Reward value
        """
        # Primary reward: task performance (higher is better)
        primary_metric = eval_metrics.get('accuracy', eval_metrics.get('f1', 0.0))
        if primary_metric == 0.0:
            # Use negative loss if no positive metric available
            primary_metric = -eval_metrics.get('eval_loss', 1.0)
        
        # Cost penalty
        cost = self._calculate_cost(configuration)
        cost_penalty = cost * self.strato_config.get('cost_penalty_weight', 0.1)
        
        # Efficiency bonus
        efficiency_bonus = 0.0
        if cost > 0:
            efficiency_bonus = (primary_metric / cost) * self.strato_config.get('efficiency_bonus_weight', 0.05)
        
        reward = primary_metric - cost_penalty + efficiency_bonus
        
        return reward
    
    def _calculate_cost(self, configuration: Dict[str, int]) -> float:
        """
        Calculate cost for a configuration.
        
        Args:
            configuration: Configuration to evaluate
            
        Returns:
            Normalized cost value
        """
        total_cost = 0.0
        
        for layer_name, rank in configuration.items():
            if layer_name in self.layer_sensitivities:
                sensitivity = self.layer_sensitivities[layer_name]
                # Combine different cost components
                layer_cost = (
                    sensitivity.parameter_cost * self.peft_method.cost_weights['parameter'] +
                    sensitivity.flops_cost * self.peft_method.cost_weights['flops'] +
                    sensitivity.memory_cost * self.peft_method.cost_weights['memory']
                ) * (rank / 8.0)  # Normalize by default rank
                
                total_cost += layer_cost
        
        # Normalize by total model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        normalized_cost = total_cost / total_params
        
        return normalized_cost
    
    def _estimate_rank_cost(self, layer_name: str, rank: int) -> float:
        """
        Estimate cost for a specific rank assignment.
        
        Args:
            layer_name: Name of the layer
            rank: Rank to assign
            
        Returns:
            Estimated cost
        """
        if layer_name not in self.layer_sensitivities:
            return 0.0
        
        sensitivity = self.layer_sensitivities[layer_name]
        base_cost = (
            sensitivity.parameter_cost * self.peft_method.cost_weights['parameter'] +
            sensitivity.flops_cost * self.peft_method.cost_weights['flops'] +
            sensitivity.memory_cost * self.peft_method.cost_weights['memory']
        )
        
        # Scale by rank
        rank_cost = base_cost * (rank / 8.0)  # Normalize by default rank
        
        # Normalize by total model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        return rank_cost / total_params
    
    def _update_rl_policy(self) -> None:
        """
        Update RL policy using collected episodes.
        """
        if len(self.episode_buffer) < 2:
            return
        
        self.logger.info(f"Updating RL policy with {len(self.episode_buffer)} episodes")
        
        # Update policy using episode buffer
        policy_metrics = self.peft_method.policy_agent.update_policy(self.episode_buffer)
        
        # Log policy update metrics
        if policy_metrics:
            self.logger.info(f"Policy update metrics: {policy_metrics}")
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def _update_best_configuration(self, eval_metrics: Dict[str, float]) -> None:
        """
        Update best configuration based on evaluation metrics.
        
        Args:
            eval_metrics: Current evaluation metrics
        """
        current_reward = self._calculate_reward(eval_metrics, self.peft_method.current_configuration)
        
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_configuration = self.peft_method.current_configuration.copy()
            
            self.logger.info(
                f"New best configuration found! Reward: {current_reward:.4f}, "
                f"Config: {self.best_configuration}"
            )
    
    def _transition_to_exploitation(self) -> None:
        """
        Transition from exploration to exploitation phase.
        """
        self.exploration_phase = False
        
        self.logger.info("Transitioning to exploitation phase")
        self.logger.info(f"Best configuration: {self.best_configuration}")
        self.logger.info(f"Best reward: {self.best_reward:.4f}")
        
        # Apply best configuration
        if self.best_configuration:
            self._apply_configuration(self.best_configuration)
    
    def _save_strato_checkpoint(self, output_dir: str, epoch: int) -> None:
        """
        Save STRATO-specific checkpoint.
        
        Args:
            output_dir: Output directory
            epoch: Current epoch
        """
        checkpoint_dir = os.path.join(output_dir, f'strato-checkpoint-epoch-{epoch+1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save standard checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Save STRATO-specific data
        strato_data = {
            'configuration_history': self.configuration_history,
            'reward_history': self.reward_history,
            'cost_history': self.cost_history,
            'best_configuration': self.best_configuration,
            'best_reward': self.best_reward,
            'current_episode_id': self.current_episode_id,
            'exploration_phase': self.exploration_phase,
            'layer_sensitivities': {k: v.__dict__ for k, v in self.layer_sensitivities.items()}
        }
        
        strato_path = os.path.join(checkpoint_dir, 'strato_data.json')
        with open(strato_path, 'w') as f:
            json.dump(strato_data, f, indent=2)
        
        self.logger.info(f"STRATO checkpoint saved to {checkpoint_dir}")
    
    def _log_strato_progress(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log STRATO training progress.
        
        Args:
            epoch: Current epoch
            metrics: Epoch metrics
        """
        # Basic progress logging
        self._log_epoch_progress(epoch, metrics)
        
        # STRATO-specific logging
        if 'rl_reward' in metrics:
            self.logger.info(
                f"  RL - Reward: {metrics['rl_reward']:.4f}, "
                f"Cost: {metrics['rl_cost']:.4f}, "
                f"Episode: {metrics.get('episode_id', 'N/A')}"
            )
        
        # Log current configuration
        if self.peft_method.current_configuration:
            config_summary = {k: v for k, v in self.peft_method.current_configuration.items()}
            self.logger.debug(f"Current config: {config_summary}")
        
        # Log phase
        phase = "Exploration" if self.exploration_phase else "Exploitation"
        self.logger.info(f"  Phase: {phase}")