#!/usr/bin/env python3
"""
Task factory for STRATO-PEFT experimental framework.

This module provides a unified interface for creating and configuring
various NLP tasks for PEFT experiments.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
from typing import Dict, Any, Type

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from .base_task import BaseTask
from .language_modeling import LanguageModelingTask
from .question_answering import QuestionAnsweringTask
from .text_classification import TextClassificationTask
from .multiple_choice import MultipleChoiceTask
from .summarization import SummarizationTask
from .translation import TranslationTask


class TaskFactory:
    """
    Factory class for creating task instances.
    """
    
    # Registry of supported tasks
    TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
        # Language modeling tasks
        "language_modeling": LanguageModelingTask,
        "causal_lm": LanguageModelingTask,
        "clm": LanguageModelingTask,
        
        # Question answering tasks
        "question_answering": QuestionAnsweringTask,
        "qa": QuestionAnsweringTask,
        "reading_comprehension": QuestionAnsweringTask,
        
        # Text classification tasks
        "text_classification": TextClassificationTask,
        "classification": TextClassificationTask,
        "sentiment_analysis": TextClassificationTask,
        "nli": TextClassificationTask,
        "glue": TextClassificationTask,
        
        # Multiple choice tasks
        "multiple_choice": MultipleChoiceTask,
        "mcqa": MultipleChoiceTask,
        "commonsense_qa": MultipleChoiceTask,
        "mmlu": MultipleChoiceTask,
        "arc": MultipleChoiceTask,
        "hellaswag": MultipleChoiceTask,
        "winogrande": MultipleChoiceTask,
        
        # Summarization tasks
        "summarization": SummarizationTask,
        "sum": SummarizationTask,
        
        # Translation tasks
        "translation": TranslationTask,
        "mt": TranslationTask,
    }
    
    # Dataset to task type mapping
    DATASET_TASK_MAPPING = {
        # Multiple choice datasets
        "mmlu": "multiple_choice",
        "arc_easy": "multiple_choice",
        "arc_challenge": "multiple_choice",
        "hellaswag": "multiple_choice",
        "winogrande": "multiple_choice",
        "piqa": "multiple_choice",
        "siqa": "multiple_choice",
        "commonsense_qa": "multiple_choice",
        "openbookqa": "multiple_choice",
        
        # Question answering datasets
        "squad": "question_answering",
        "squad_v2": "question_answering",
        "natural_questions": "question_answering",
        "ms_marco": "question_answering",
        "quac": "question_answering",
        "coqa": "question_answering",
        
        # Text classification datasets
        "sst2": "text_classification",
        "cola": "text_classification",
        "mnli": "text_classification",
        "qnli": "text_classification",
        "rte": "text_classification",
        "wnli": "text_classification",
        "mrpc": "text_classification",
        "qqp": "text_classification",
        "stsb": "text_classification",
        "imdb": "text_classification",
        "ag_news": "text_classification",
        
        # Language modeling datasets
        "wikitext": "language_modeling",
        "ptb": "language_modeling",
        "lambada": "language_modeling",
        "pile": "language_modeling",
        "c4": "language_modeling",
        
        # Summarization datasets
        "cnn_dailymail": "summarization",
        "xsum": "summarization",
        "newsroom": "summarization",
        "multi_news": "summarization",
        
        # Translation datasets
        "wmt14": "translation",
        "wmt16": "translation",
        "opus": "translation",
    }
    
    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizer):
        """
        Initialize the task factory.
        
        Args:
            config: Task configuration
            tokenizer: Pre-trained tokenizer
        """
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate the task configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["type", "dataset"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required task config field: {field}")
        
        # Validate dataset configuration
        dataset_config = self.config.dataset
        if "name" not in dataset_config:
            raise ValueError("Missing dataset name in task config")
    
    def create_task(self) -> BaseTask:
        """
        Create a task instance based on configuration.
        
        Returns:
            BaseTask: Task instance
        """
        task_type = self._determine_task_type()
        
        self.logger.info(f"Creating task: {task_type}")
        self.logger.info(f"Dataset: {self.config.dataset.name}")
        
        # Get task class
        if task_type not in self.TASK_REGISTRY:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                f"Supported types: {list(self.TASK_REGISTRY.keys())}"
            )
        
        task_class = self.TASK_REGISTRY[task_type]
        
        # Create task instance
        task = task_class(self.config, self.tokenizer)
        
        # Validate task configuration
        task.validate_config()
        
        # Load data
        task.load_data()
        
        # Log task information
        task.log_task_info()
        
        return task
    
    def _determine_task_type(self) -> str:
        """
        Determine the task type based on configuration.
        
        Returns:
            str: Task type
        """
        # Check if task type is explicitly specified
        if "type" in self.config and self.config.type in self.TASK_REGISTRY:
            return self.config.type
        
        # Try to infer from dataset name
        dataset_name = self.config.dataset.name.lower()
        
        # Direct mapping
        if dataset_name in self.DATASET_TASK_MAPPING:
            return self.DATASET_TASK_MAPPING[dataset_name]
        
        # Pattern matching
        for pattern, task_type in self._get_dataset_patterns().items():
            if pattern in dataset_name:
                return task_type
        
        # Fallback to configured type
        if "type" in self.config:
            return self.config.type
        
        # Default fallback
        self.logger.warning(
            f"Could not determine task type for dataset: {dataset_name}. "
            "Falling back to language_modeling."
        )
        return "language_modeling"
    
    def _get_dataset_patterns(self) -> Dict[str, str]:
        """
        Get dataset name patterns for task type inference.
        
        Returns:
            Dict[str, str]: Pattern to task type mapping
        """
        return {
            "glue": "text_classification",
            "superglue": "text_classification",
            "squad": "question_answering",
            "qa": "question_answering",
            "wmt": "translation",
            "opus": "translation",
            "cnn": "summarization",
            "xsum": "summarization",
            "wiki": "language_modeling",
            "pile": "language_modeling",
            "c4": "language_modeling",
            "mmlu": "multiple_choice",
            "arc": "multiple_choice",
            "hella": "multiple_choice",
            "wino": "multiple_choice",
            "piqa": "multiple_choice",
            "siqa": "multiple_choice",
            "commonsense": "multiple_choice",
        }
    
    def get_supported_tasks(self) -> Dict[str, Any]:
        """
        Get information about supported tasks.
        
        Returns:
            Dict[str, Any]: Supported tasks information
        """
        return {
            "task_types": list(self.TASK_REGISTRY.keys()),
            "datasets": list(self.DATASET_TASK_MAPPING.keys()),
            "task_classes": {
                task_type: task_class.__name__ 
                for task_type, task_class in self.TASK_REGISTRY.items()
            }
        }
    
    def register_task(self, task_type: str, task_class: Type[BaseTask]) -> None:
        """
        Register a new task type.
        
        Args:
            task_type: Task type identifier
            task_class: Task class
        """
        if not issubclass(task_class, BaseTask):
            raise ValueError(f"Task class must inherit from BaseTask")
        
        self.TASK_REGISTRY[task_type] = task_class
        self.logger.info(f"Registered task type: {task_type}")
    
    def register_dataset(self, dataset_name: str, task_type: str) -> None:
        """
        Register a dataset to task type mapping.
        
        Args:
            dataset_name: Dataset name
            task_type: Task type
        """
        if task_type not in self.TASK_REGISTRY:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.DATASET_TASK_MAPPING[dataset_name] = task_type
        self.logger.info(f"Registered dataset mapping: {dataset_name} -> {task_type}")
    
    @classmethod
    def list_supported_datasets(cls) -> Dict[str, List[str]]:
        """
        List supported datasets grouped by task type.
        
        Returns:
            Dict[str, List[str]]: Datasets grouped by task type
        """
        grouped = {}
        for dataset, task_type in cls.DATASET_TASK_MAPPING.items():
            if task_type not in grouped:
                grouped[task_type] = []
            grouped[task_type].append(dataset)
        
        # Sort datasets within each group
        for task_type in grouped:
            grouped[task_type].sort()
        
        return grouped
    
    @classmethod
    def get_task_type_for_dataset(cls, dataset_name: str) -> str:
        """
        Get the task type for a given dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            str: Task type
        """
        dataset_name = dataset_name.lower()
        
        # Direct mapping
        if dataset_name in cls.DATASET_TASK_MAPPING:
            return cls.DATASET_TASK_MAPPING[dataset_name]
        
        # Pattern matching
        patterns = {
            "glue": "text_classification",
            "superglue": "text_classification",
            "squad": "question_answering",
            "qa": "question_answering",
            "wmt": "translation",
            "opus": "translation",
            "cnn": "summarization",
            "xsum": "summarization",
            "wiki": "language_modeling",
            "pile": "language_modeling",
            "c4": "language_modeling",
            "mmlu": "multiple_choice",
            "arc": "multiple_choice",
            "hella": "multiple_choice",
            "wino": "multiple_choice",
            "piqa": "multiple_choice",
            "siqa": "multiple_choice",
            "commonsense": "multiple_choice",
        }
        
        for pattern, task_type in patterns.items():
            if pattern in dataset_name:
                return task_type
        
        # Default fallback
        return "language_modeling"
    
    def __repr__(self) -> str:
        """String representation of the factory."""
        return f"TaskFactory(task_type='{self.config.get('type', 'auto')}', dataset='{self.config.dataset.name}')"