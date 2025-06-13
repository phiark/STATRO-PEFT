#!/usr/bin/env python3
"""
Training module for STRATO-PEFT experimental framework.

This module provides training components including:
- BaseTrainer: Abstract base class for all trainers
- PEFTTrainer: Specialized trainer for PEFT methods
- StratoTrainer: Advanced trainer for STRATO-PEFT with RL integration
- TrainingMetrics: Data structures for tracking training progress
- TrainingCallbacks: Callback system for training events

Author: STRATO-PEFT Research Team
Date: 2024
"""

from .base_trainer import BaseTrainer, TrainingMetrics
from .peft_trainer import PEFTTrainer
from .strato_trainer import StratoTrainer
from .callbacks import TrainingCallbacks, CallbackEvent

__all__ = [
    'BaseTrainer',
    'PEFTTrainer', 
    'StratoTrainer',
    'TrainingMetrics',
    'TrainingCallbacks',
    'CallbackEvent'
]