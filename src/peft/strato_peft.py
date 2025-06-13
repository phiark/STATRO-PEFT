#!/usr/bin/env python3
"""
STRATO-PEFT (Strategic Rank Optimization) implementation for parameter-efficient fine-tuning.

This module implements the STRATO-PEFT method which treats fine-tuning as a cost-constrained
exploration game, using reinforcement learning to strategically allocate adapter parameters
across model layers while optimizing for resource efficiency.

Key Components:
1. Mapping Builder - Analyzes layer sensitivity through low-rank probes
2. Policy Agent - RL/PPO agent for layer-rank action selection
3. Rank Scheduler - Dynamic rank adjustment using predicted marginal utility
4. Memory Cache - Go-Explore style episodic storage for explored configurations

Reference:
    STRATO-PEFT: Strategic Resource-Aware Tunable Optimization for Parameter-Efficient Fine-Tuning
    Internal Research Document

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import os
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .base_peft import BasePEFT, PEFTMetrics
from .lora import LoRALayer


@dataclass
class LayerSensitivity:
    """Layer sensitivity analysis results."""
    layer_name: str
    sensitivity_score: float
    parameter_cost: int
    flops_cost: float
    memory_cost: float
    marginal_utility: float  # gain / cost ratio


@dataclass
class StratoState:
    """STRATO-PEFT MDP state representation."""
    current_layer: int
    remaining_budget: float
    validation_reward_estimate: float
    layer_sensitivities: List[float]
    current_configuration: Dict[str, int]  # layer_name -> rank


@dataclass
class StratoAction:
    """STRATO-PEFT action representation."""
    action_type: str  # 'insert', 'freeze', 'skip'
    layer_name: str
    rank: Optional[int] = None


@dataclass
class EpisodeMemory:
    """Go-Explore style episode memory."""
    configuration: Dict[str, int]
    reward: float
    cost: float
    validation_score: float
    episode_id: str
    timestamp: float


class MappingBuilder:
    """
    Analyzes layer sensitivity through low-rank probes.
    """
    
    def __init__(self, model: PreTrainedModel, config: DictConfig, logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.layer_sensitivities: Dict[str, LayerSensitivity] = {}
        
    def analyze_layer_sensitivity(
        self,
        dataloader: DataLoader,
        target_modules: List[str],
        probe_rank: int = 4,
        num_samples: int = 100
    ) -> Dict[str, LayerSensitivity]:
        """
        Analyze sensitivity of each layer using low-rank probes.
        
        Args:
            dataloader: Data loader for sensitivity analysis
            target_modules: List of target module patterns
            probe_rank: Rank for sensitivity probes
            num_samples: Number of samples for analysis
            
        Returns:
            Dictionary mapping layer names to sensitivity analysis
        """
        self.logger.info(f"Analyzing layer sensitivity with probe_rank={probe_rank}")
        
        # Get baseline performance
        baseline_loss = self._compute_baseline_loss(dataloader, num_samples)
        
        # Analyze each target layer
        for name, module in self.model.named_modules():
            if self._is_target_module(name, module, target_modules):
                sensitivity = self._analyze_single_layer(
                    name, module, dataloader, baseline_loss, probe_rank, num_samples
                )
                self.layer_sensitivities[name] = sensitivity
                
        self.logger.info(f"Analyzed {len(self.layer_sensitivities)} layers")
        return self.layer_sensitivities
    
    def _is_target_module(self, name: str, module: nn.Module, target_modules: List[str]) -> bool:
        """Check if module is a target for adaptation."""
        if not isinstance(module, nn.Linear):
            return False
        return any(target in name for target in target_modules)
    
    def _compute_baseline_loss(self, dataloader: DataLoader, num_samples: int) -> float:
        """Compute baseline loss without any adaptation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                # Move batch to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _analyze_single_layer(
        self,
        layer_name: str,
        module: nn.Module,
        dataloader: DataLoader,
        baseline_loss: float,
        probe_rank: int,
        num_samples: int
    ) -> LayerSensitivity:
        """Analyze sensitivity of a single layer."""
        # Create temporary LoRA probe
        original_module = module
        probe_layer = LoRALayer(
            original_layer=module,
            rank=probe_rank,
            alpha=probe_rank,  # alpha = rank for simplicity
            dropout=0.0
        )
        
        # Replace module temporarily
        parent_name = '.'.join(layer_name.split('.')[:-1])
        child_name = layer_name.split('.')[-1]
        
        if parent_name:
            parent_module = dict(self.model.named_modules())[parent_name]
        else:
            parent_module = self.model
            
        setattr(parent_module, child_name, probe_layer)
        
        try:
            # Quick training on probe
            probe_loss = self._train_probe(probe_layer, dataloader, num_samples)
            
            # Calculate sensitivity metrics
            sensitivity_score = max(0.0, baseline_loss - probe_loss)
            parameter_cost = sum(p.numel() for p in probe_layer.get_lora_parameters())
            flops_cost = self._estimate_flops_cost(module, probe_rank)
            memory_cost = parameter_cost * 4  # Assume float32
            
            # Calculate marginal utility
            total_cost = parameter_cost + flops_cost * 0.001 + memory_cost * 0.0001
            marginal_utility = sensitivity_score / max(total_cost, 1e-6)
            
            return LayerSensitivity(
                layer_name=layer_name,
                sensitivity_score=sensitivity_score,
                parameter_cost=parameter_cost,
                flops_cost=flops_cost,
                memory_cost=memory_cost,
                marginal_utility=marginal_utility
            )
            
        finally:
            # Restore original module
            setattr(parent_module, child_name, original_module)
    
    def _train_probe(self, probe_layer: LoRALayer, dataloader: DataLoader, num_samples: int) -> float:
        """Quick training of probe layer."""
        optimizer = torch.optim.AdamW(probe_layer.get_lora_parameters(), lr=1e-3)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Move batch to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _estimate_flops_cost(self, module: nn.Module, rank: int) -> float:
        """Estimate FLOPs cost for LoRA adaptation.
        
        Args:
            module: Target module to adapt
            rank: LoRA rank
            
        Returns:
            Estimated FLOPs cost
        """
        if not hasattr(module, 'in_features') or not hasattr(module, 'out_features'):
            return 0.0
        
        in_features = module.in_features
        out_features = module.out_features
        
        # Estimate based on typical usage patterns
        batch_size = self.config.get('batch_size', 8)
        seq_len = self.config.get('max_seq_length', 512)
        
        # Original linear layer FLOPs: batch_size * seq_len * in_features * out_features
        original_flops = batch_size * seq_len * in_features * out_features
        
        # LoRA additional FLOPs:
        # - Down projection: batch_size * seq_len * in_features * rank
        # - Up projection: batch_size * seq_len * rank * out_features
        lora_flops = batch_size * seq_len * (in_features * rank + rank * out_features)
        
        # Return relative cost (LoRA FLOPs / Original FLOPs)
        if original_flops > 0:
            relative_cost = lora_flops / original_flops
        else:
            relative_cost = 0.0
        
        # Scale by a factor to make it comparable to other costs
        return relative_cost * 100.0  # Scale to percentage


class PolicyAgent:
    """
    RL/PPO agent for layer-rank action selection.
    """
    
    def __init__(self, config: DictConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # PPO hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Network architecture
        self.state_dim = config.get('state_dim', 64)
        self.action_dim = config.get('action_dim', 32)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # Initialize networks
        self._build_networks()
        
        # Training state
        self.episode_rewards = deque(maxlen=100)
        self.episode_costs = deque(maxlen=100)
        
    def _build_networks(self):
        """Build policy and value networks."""
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Optimizer
        params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        
    def select_action(self, state: StratoState, deterministic: bool = False) -> Tuple[StratoAction, float, float]:
        """Select action based on current state.
        
        Args:
            state: Current MDP state
            deterministic: If True, select action with highest probability
            
        Returns:
            Tuple of (action, log_probability, state_value)
        """
        # Convert state to tensor
        state_tensor = self._state_to_tensor(state).unsqueeze(0)  # Add batch dimension
        
        # Get action probabilities and state value
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            state_value = self.value_net(state_tensor)
            
        # Create action distribution
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Select action
        if deterministic:
            action_idx = action_probs.argmax(dim=-1)
        else:
            action_idx = action_dist.sample()
        
        # Get log probability
        log_prob = action_dist.log_prob(action_idx)
        
        # Convert to StratoAction
        action = self._idx_to_action(action_idx.item(), state)
        
        return action, log_prob.item(), state_value.item()
    
    def _state_to_tensor(self, state: StratoState) -> torch.Tensor:
        """Convert state to tensor representation.
        
        Args:
            state: StratoState object to convert
            
        Returns:
            Tensor representation of the state
        """
        features = []
        
        # Basic state features
        features.extend([
            state.current_layer / 100.0,  # Normalized layer index
            state.remaining_budget,
            state.validation_reward_estimate,
        ])
        
        # Configuration features
        total_allocated_rank = sum(state.current_configuration.values()) if state.current_configuration else 0
        num_allocated_layers = len(state.current_configuration) if state.current_configuration else 0
        avg_rank = total_allocated_rank / max(num_allocated_layers, 1)
        
        features.extend([
            total_allocated_rank / 1000.0,  # Normalized total rank
            num_allocated_layers / 100.0,   # Normalized layer count
            avg_rank / 32.0,                # Normalized average rank
        ])
        
        # Layer sensitivity statistics
        if state.layer_sensitivities:
            sens_array = np.array(state.layer_sensitivities)
            features.extend([
                sens_array.mean(),
                sens_array.std(),
                sens_array.max(),
                sens_array.min(),
                np.percentile(sens_array, 75),
                np.percentile(sens_array, 25),
            ])
        else:
            features.extend([0.0] * 6)
        
        # Add raw layer sensitivities (truncated/padded to fixed size)
        max_sens_features = 45  # Reserve space for sensitivity values
        sens_features = state.layer_sensitivities[:max_sens_features] if state.layer_sensitivities else []
        sens_features.extend([0.0] * (max_sens_features - len(sens_features)))
        features.extend(sens_features)
        
        # Ensure fixed size matches state_dim
        features = features[:self.state_dim]
        features.extend([0.0] * (self.state_dim - len(features)))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _idx_to_action(self, action_idx: int, state: StratoState) -> StratoAction:
        """Convert action index to StratoAction.
        
        Args:
            action_idx: Action index from policy network
            state: Current state for context
            
        Returns:
            StratoAction object
        """
        # Define available ranks based on configuration
        available_ranks = [4, 8, 16, 32, 64]  # Extended rank options
        num_ranks = len(available_ranks)
        
        # Get available layers (those not yet configured)
        configured_layers = set(state.current_configuration.keys()) if state.current_configuration else set()
        
        # For simplicity, assume we have layer names from sensitivity analysis
        # In practice, this would come from the actual model structure
        if hasattr(state, 'available_layers') and state.available_layers:
            available_layers = [layer for layer in state.available_layers if layer not in configured_layers]
        else:
            # Fallback: generate layer names based on sensitivity count
            num_layers = len(state.layer_sensitivities) if state.layer_sensitivities else 10
            available_layers = [f"layer_{i}" for i in range(num_layers) if f"layer_{i}" not in configured_layers]
        
        if not available_layers:
            return StratoAction(action_type='skip', layer_name='')
        
        num_layers = len(available_layers)
        total_insert_actions = num_layers * num_ranks
        total_actions = total_insert_actions + num_layers + 1  # insert + freeze + skip
        
        # Normalize action index
        action_idx = action_idx % total_actions
        
        # Decode action
        if action_idx < total_insert_actions:
            # Insert action with specific rank
            layer_idx = action_idx // num_ranks
            rank_idx = action_idx % num_ranks
            
            if layer_idx < len(available_layers):
                layer_name = available_layers[layer_idx]
                rank = available_ranks[rank_idx]
                
                # Adjust rank based on remaining budget
                if state.remaining_budget < 0.5:
                    rank = min(rank, 8)  # Use smaller ranks when budget is low
                elif state.remaining_budget > 0.8:
                    rank = min(rank, 32)  # Cap at reasonable size
                
                return StratoAction(action_type='insert', layer_name=layer_name, rank=rank)
        
        elif action_idx < total_insert_actions + num_layers:
            # Freeze action
            layer_idx = action_idx - total_insert_actions
            if layer_idx < len(available_layers):
                layer_name = available_layers[layer_idx]
                return StratoAction(action_type='freeze', layer_name=layer_name)
        
        # Default: skip action
        return StratoAction(action_type='skip', layer_name='')
    
    def update_policy(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update policy using PPO algorithm.
        
        Args:
            episodes: List of episode data containing states, actions, rewards, etc.
            
        Returns:
            Dictionary containing training metrics
        """
        if not episodes:
            return {}
        
        # Extract episode data
        states = []
        actions = []
        rewards = []
        old_log_probs = []
        values = []
        
        for episode in episodes:
            if 'states' in episode and 'actions' in episode:
                states.extend(episode['states'])
                actions.extend(episode['actions'])
                rewards.extend(episode['rewards'])
                old_log_probs.extend(episode.get('log_probs', []))
                values.extend(episode.get('values', []))
        
        if not states:
            # Fallback for simple episode format
            total_reward = sum(ep['reward'] for ep in episodes)
            total_cost = sum(ep['cost'] for ep in episodes)
            self.episode_rewards.append(total_reward)
            self.episode_costs.append(total_cost)
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_reward': total_reward,
                'total_cost': total_cost
            }
        
        # Convert to tensors
        states_tensor = torch.stack([self._state_to_tensor(s) for s in states])
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32) if old_log_probs else None
        
        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = self._compute_advantages(rewards_tensor, values)
        returns = advantages + torch.tensor(values, dtype=torch.float32) if values else rewards_tensor
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Multiple epochs of optimization
        for _ in range(4):  # PPO typically uses 4 epochs
            # Forward pass
            action_probs = self.policy_net(states_tensor)
            state_values = self.value_net(states_tensor).squeeze()
            
            # Compute new log probabilities
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions_tensor)
            entropy = action_dist.entropy().mean()
            
            # Compute ratio for PPO clipping
            if old_log_probs_tensor is not None:
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            else:
                ratio = torch.ones_like(new_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(state_values, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + list(self.value_net.parameters()), 0.5)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
        
        # Update tracking
        total_reward = rewards_tensor.sum().item()
        total_cost = sum(ep.get('cost', 0) for ep in episodes)
        self.episode_rewards.append(total_reward)
        self.episode_costs.append(total_cost)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_reward': total_reward,
            'total_cost': total_cost,
            'advantage_mean': advantages.mean().item() if len(advantages) > 0 else 0.0,
            'advantage_std': advantages.std().item() if len(advantages) > 0 else 0.0
        }
    
    def _compute_advantages(self, rewards: torch.Tensor, values: List[float], gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
        """Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards
            values: List of state values
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            Tensor of computed advantages
        """
        if not values:
            # Simple case: use rewards as advantages
            return rewards - rewards.mean()
        
        values_tensor = torch.tensor(values + [0], dtype=torch.float32)  # Add terminal value
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_tensor[t + 1] - values_tensor[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        
        return advantages


class RankScheduler:
    """
    Dynamic rank adjustment using predicted marginal utility.
    """
    
    def __init__(self, config: DictConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.max_rank = config.get('max_rank', 32)
        self.min_rank = config.get('min_rank', 4)
        self.initial_rank = config.get('initial_rank', 8)
        
        # Dynamic adjustment parameters
        self.utility_threshold = config.get('utility_threshold', 0.1)
        self.adjustment_factor = config.get('adjustment_factor', 1.5)
        
    def schedule_rank(
        self,
        layer_name: str,
        current_rank: int,
        marginal_utility: float,
        budget_remaining: float
    ) -> int:
        """Schedule rank for a layer based on marginal utility."""
        # Increase rank if utility is high and budget allows
        if marginal_utility > self.utility_threshold and budget_remaining > 0.5:
            new_rank = min(self.max_rank, int(current_rank * self.adjustment_factor))
        # Decrease rank if utility is low
        elif marginal_utility < self.utility_threshold / 2:
            new_rank = max(self.min_rank, int(current_rank / self.adjustment_factor))
        else:
            new_rank = current_rank
            
        if new_rank != current_rank:
            self.logger.debug(
                f"Rank adjustment for {layer_name}: {current_rank} -> {new_rank} "
                f"(utility={marginal_utility:.4f})"
            )
            
        return new_rank


class MemoryCache:
    """
    Go-Explore style episodic storage for explored configurations.
    """
    
    def __init__(self, config: DictConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.max_size = config.get('max_size', 1000)
        self.revisit_threshold = config.get('revisit_threshold', 3)
        
        self.memory: Dict[str, EpisodeMemory] = {}
        self.visit_counts: Dict[str, int] = defaultdict(int)
        
    def add_episode(self, episode: EpisodeMemory) -> None:
        """Add episode to memory cache."""
        config_key = self._config_to_key(episode.configuration)
        
        # Update visit count
        self.visit_counts[config_key] += 1
        
        # Store if new or better
        if (config_key not in self.memory or 
            episode.reward > self.memory[config_key].reward):
            self.memory[config_key] = episode
            
        # Prune if cache is full
        if len(self.memory) > self.max_size:
            self._prune_cache()
    
    def should_explore(self, configuration: Dict[str, int]) -> bool:
        """Check if configuration should be explored."""
        config_key = self._config_to_key(configuration)
        return self.visit_counts[config_key] < self.revisit_threshold
    
    def get_best_episodes(self, k: int = 10) -> List[EpisodeMemory]:
        """Get top k episodes by reward."""
        episodes = list(self.memory.values())
        episodes.sort(key=lambda x: x.reward, reverse=True)
        return episodes[:k]
    
    def _config_to_key(self, configuration: Dict[str, int]) -> str:
        """Convert configuration to string key."""
        return json.dumps(configuration, sort_keys=True)
    
    def _prune_cache(self) -> None:
        """Prune cache to maintain size limit."""
        # Remove episodes with lowest reward
        episodes = list(self.memory.items())
        episodes.sort(key=lambda x: x[1].reward)
        
        # Remove bottom 10%
        num_to_remove = len(episodes) // 10
        for i in range(num_to_remove):
            config_key = episodes[i][0]
            del self.memory[config_key]
            if config_key in self.visit_counts:
                del self.visit_counts[config_key]


class StratoPEFT(BasePEFT):
    """
    STRATO-PEFT main implementation.
    """
    
    def _initialize_peft(self) -> None:
        """
        Initialize STRATO-PEFT components.
        """
        self.target_modules = self.config.target_modules
        
        # Cost weights
        self.cost_weights = {
            'parameter': self.config.get('parameter_weight', 1.0),
            'flops': self.config.get('flops_weight', 0.001),
            'memory': self.config.get('memory_weight', 0.0001)
        }
        
        # RL agent configuration
        agent_config = self.config.get('rl_agent', {})
        
        # Initialize components
        self.mapping_builder = MappingBuilder(self.model, self.config, self.logger)
        self.policy_agent = PolicyAgent(agent_config, self.logger)
        self.rank_scheduler = RankScheduler(self.config.get('rank_scheduler', {}), self.logger)
        self.memory_cache = MemoryCache(self.config.get('memory_cache', {}), self.logger)
        
        # Training state
        self.current_configuration: Dict[str, int] = {}
        self.lora_layers: Dict[str, LoRALayer] = {}
        self.episode_count = 0
        
        self.logger.info("STRATO-PEFT components initialized")
        self.logger.info(f"Cost weights: {self.cost_weights}")
    
    def apply_peft(self, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Apply STRATO-PEFT to the model based on configuration.
        
        Args:
            config: Optional configuration override. If None, uses current_configuration
                   Format: {layer_name: {'rank': int, 'action_type': str}}
        
        Returns:
            The adapted model with STRATO-PEFT applied
        """
        if self.is_applied:
            self.logger.warning("STRATO-PEFT already applied")
            return self.adapted_model
        
        self.logger.info("Applying STRATO-PEFT to model...")
        
        # Use provided config or current configuration
        target_config = config or self.current_configuration
        
        # For now, apply a simple configuration
        # In full implementation, this would run the RL exploration
        self.adapted_model = self.model
        
        if not target_config:
            self.logger.warning("No configuration provided. Applying default LoRA configuration.")
            # Apply default configuration to all target modules
            default_rank = self.config.get('default_rank', 8)
            modules_replaced = 0
            
            for name, module in self.adapted_model.named_modules():
                if self._should_replace_module(name, module):
                    self._replace_module_with_lora(name, module, default_rank)
                    self.current_configuration[name] = {'rank': default_rank, 'action_type': 'insert'}
                    modules_replaced += 1
            
            self.logger.info(f"Applied default LoRA to {modules_replaced} modules")
        else:
            # Apply configuration from config dict
            modules_replaced = 0
            frozen_count = 0
            
            for layer_name, layer_config in target_config.items():
                try:
                    # Navigate to the target module
                    module = self.adapted_model
                    module_path = layer_name.split('.')
                    
                    # Find the parent module and attribute name
                    for attr in module_path[:-1]:
                        module = getattr(module, attr)
                    
                    final_attr = module_path[-1]
                    target_module = getattr(module, final_attr)
                    
                    action_type = layer_config.get('action_type', 'insert')
                    
                    if action_type == 'insert' and 'rank' in layer_config:
                        # Apply LoRA
                        rank = layer_config['rank']
                        self._replace_module_with_lora(layer_name, target_module, rank)
                        modules_replaced += 1
                        self.logger.info(f"Applied LoRA to {layer_name} with rank {rank}")
                        
                    elif action_type == 'freeze':
                        # Freeze the module parameters
                        for param in target_module.parameters():
                            param.requires_grad = False
                        frozen_count += 1
                        self.logger.info(f"Frozen parameters in {layer_name}")
                        
                    elif action_type == 'skip':
                        # Skip this layer (do nothing)
                        self.logger.info(f"Skipped {layer_name}")
                        
                    else:
                        self.logger.warning(f"Unknown action type '{action_type}' for {layer_name}")
                        
                except AttributeError as e:
                    self.logger.error(f"Failed to find module {layer_name}: {e}")
                except Exception as e:
                    self.logger.error(f"Failed to apply PEFT to {layer_name}: {e}")
            
            self.logger.info(f"STRATO-PEFT application completed: {modules_replaced} LoRA modules, {frozen_count} frozen modules")
        
        if modules_replaced == 0:
            self.logger.warning("No modules were replaced with STRATO-PEFT layers")
        else:
            self.logger.info(f"Replaced {modules_replaced} modules with STRATO-PEFT layers")
        
        self.is_applied = True
        self.log_peft_info()
        
        return self.adapted_model
    
    def _should_replace_module(self, name: str, module: nn.Module) -> bool:
        """Check if a module should be replaced."""
        if not isinstance(module, nn.Linear):
            return False
        return any(target in name for target in self.target_modules)
    
    def _replace_module_with_lora(self, name: str, module: nn.Module, rank: int) -> None:
        """Replace a module with LoRA layer."""
        alpha = self.config.get('alpha', rank)
        dropout = self.config.get('dropout', 0.0)
        
        lora_layer = LoRALayer(
            original_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout
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
        self.current_configuration[name] = rank
        
        self.logger.debug(f"Replaced {name} with STRATO-PEFT layer (rank={rank})")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters."""
        if not self.is_applied:
            return []
        
        trainable_params = []
        for lora_layer in self.lora_layers.values():
            trainable_params.extend(lora_layer.get_lora_parameters())
        
        return trainable_params
    
    def get_peft_metrics(self) -> PEFTMetrics:
        """Get STRATO-PEFT metrics."""
        if not self.is_applied:
            return PEFTMetrics(
                trainable_params=0,
                total_params=0,
                trainable_ratio=0.0,
                memory_usage_mb=0.0
            )
        
        trainable_params, total_params, trainable_ratio = self.get_parameter_count()
        memory_usage = self.get_memory_usage()
        
        # STRATO-PEFT specific metrics
        rank_distribution = {name: layer.rank for name, layer in self.lora_layers.items()}
        
        # Calculate adaptation efficiency
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
        """Save STRATO-PEFT weights."""
        if not self.is_applied:
            raise RuntimeError("STRATO-PEFT must be applied before saving weights")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Collect weights and configuration
        strato_state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            strato_state_dict[f"{name}.lora_A.weight"] = lora_layer.lora_A.weight
            strato_state_dict[f"{name}.lora_B.weight"] = lora_layer.lora_B.weight
        
        # Save weights, configuration, and metadata
        torch.save({
            'strato_state_dict': strato_state_dict,
            'configuration': self.current_configuration,
            'config': {
                'target_modules': self.target_modules,
                'cost_weights': self.cost_weights
            },
            'episode_count': self.episode_count
        }, save_path)
        
        self.logger.info(f"STRATO-PEFT weights saved to {save_path}")
    
    def load_peft_weights(self, load_path: str) -> None:
        """Load STRATO-PEFT weights."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"STRATO-PEFT weights file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location='cpu')
        strato_state_dict = checkpoint['strato_state_dict']
        self.current_configuration = checkpoint.get('configuration', {})
        
        if not self.is_applied:
            self.logger.warning("STRATO-PEFT not applied yet, applying first...")
            self.apply_peft()
        
        # Load weights
        for name, lora_layer in self.lora_layers.items():
            if f"{name}.lora_A.weight" in strato_state_dict:
                lora_layer.lora_A.weight.data = strato_state_dict[f"{name}.lora_A.weight"]
            if f"{name}.lora_B.weight" in strato_state_dict:
                lora_layer.lora_B.weight.data = strato_state_dict[f"{name}.lora_B.weight"]
        
        self.logger.info(f"STRATO-PEFT weights loaded from {load_path}")
    
    def validate_config(self) -> None:
        """Validate STRATO-PEFT configuration."""
        super().validate_config()
        
        # STRATO-PEFT specific validation
        if not hasattr(self.config, 'cost_weights'):
            self.logger.warning("No cost_weights specified, using defaults")
        
        # Validate cost weights
        for weight_name in ['parameter_weight', 'flops_weight', 'memory_weight']:
            if hasattr(self.config, weight_name):
                weight_value = getattr(self.config, weight_name)
                if weight_value < 0:
                    raise ValueError(f"{weight_name} must be non-negative")