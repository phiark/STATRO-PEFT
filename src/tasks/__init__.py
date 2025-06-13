#!/usr/bin/env python3
"""
Task modules for STRATO-PEFT experimental framework.

This package contains task-specific implementations for various NLP tasks
including language modeling, question answering, text classification, etc.

Author: STRATO-PEFT Research Team
Date: 2024
"""

from .task_factory import TaskFactory
from .base_task import BaseTask

__all__ = [
    "TaskFactory",
    "BaseTask",
]