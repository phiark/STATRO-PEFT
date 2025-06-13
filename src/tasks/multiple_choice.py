#!/usr/bin/env python3
"""
Multiple choice task implementation for STRATO-PEFT experimental framework.

This module implements multiple choice tasks such as MMLU, ARC, HellaSwag, etc.
These tasks involve selecting the correct answer from multiple options.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import os
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from .base_task import BaseTask, TaskMetrics


class MultipleChoiceDataset(Dataset):
    """
    Dataset class for multiple choice tasks.
    """
    
    def __init__(self, 
                 data: List[Dict[str, Any]], 
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512,
                 prompt_template: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            data: List of data samples
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            prompt_template: Template for formatting prompts
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or "{question}\n{choices}"
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Format the prompt
        formatted_prompt = self._format_prompt(item)
        
        # Tokenize choices
        choice_inputs = []
        for choice in item["choices"]:
            # Combine prompt with choice
            full_text = formatted_prompt + " " + choice
            
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            choice_inputs.append({
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            })
        
        return {
            "choice_inputs": choice_inputs,
            "label": item["answer"],
            "question": item["question"],
            "choices": item["choices"],
            "subject": item.get("subject", ""),
            "id": item.get("id", idx)
        }
    
    def _format_prompt(self, item: Dict[str, Any]) -> str:
        """
        Format the prompt for a multiple choice question.
        
        Args:
            item: Data item
            
        Returns:
            str: Formatted prompt
        """
        question = item["question"]
        choices = item["choices"]
        
        # Format choices with labels (A, B, C, D, ...)
        choice_labels = [chr(65 + i) for i in range(len(choices))]  # A, B, C, D, ...
        formatted_choices = "\n".join([
            f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
        ])
        
        # Apply template
        prompt = self.prompt_template.format(
            question=question,
            choices=formatted_choices
        )
        
        return prompt


class MultipleChoiceTask(BaseTask):
    """
    Multiple choice task implementation.
    """
    
    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizer):
        """
        Initialize the multiple choice task.
        
        Args:
            config: Task configuration
            tokenizer: Pre-trained tokenizer
        """
        super().__init__(config, tokenizer)
        
        # Task-specific configuration
        self.max_length = config.get("max_length", 512)
        self.prompt_template = config.get("prompt_template", None)
        self.num_choices = config.get("num_choices", 4)
        
        # Dataset-specific configurations
        self.dataset_configs = {
            "mmlu": {
                "question_column": "question",
                "choices_column": "choices",
                "answer_column": "answer",
                "subject_column": "subject",
                "num_choices": 4,
                "prompt_template": "Question: {question}\n{choices}\nAnswer:"
            },
            "arc_easy": {
                "question_column": "question",
                "choices_column": "choices",
                "answer_column": "answerKey",
                "num_choices": 4,
                "prompt_template": "{question}\n{choices}\nAnswer:"
            },
            "arc_challenge": {
                "question_column": "question",
                "choices_column": "choices",
                "answer_column": "answerKey",
                "num_choices": 4,
                "prompt_template": "{question}\n{choices}\nAnswer:"
            },
            "hellaswag": {
                "question_column": "ctx",
                "choices_column": "endings",
                "answer_column": "label",
                "num_choices": 4,
                "prompt_template": "{question}\n{choices}\nAnswer:"
            },
            "winogrande": {
                "question_column": "sentence",
                "choices_column": "option1",  # Special handling needed
                "answer_column": "answer",
                "num_choices": 2,
                "prompt_template": "{question}\n{choices}\nAnswer:"
            },
            "piqa": {
                "question_column": "goal",
                "choices_column": "sol1",  # Special handling needed
                "answer_column": "label",
                "num_choices": 2,
                "prompt_template": "Goal: {question}\n{choices}\nAnswer:"
            }
        }
    
    def _initialize_task(self) -> None:
        """
        Initialize task-specific components.
        """
        dataset_name = self.config.dataset.name.lower()
        
        # Get dataset-specific configuration
        if dataset_name in self.dataset_configs:
            dataset_config = self.dataset_configs[dataset_name]
            self.question_column = dataset_config["question_column"]
            self.choices_column = dataset_config["choices_column"]
            self.answer_column = dataset_config["answer_column"]
            self.num_choices = dataset_config["num_choices"]
            
            # Use dataset-specific prompt template if not specified
            if self.prompt_template is None:
                self.prompt_template = dataset_config["prompt_template"]
        else:
            # Default configuration
            self.question_column = "question"
            self.choices_column = "choices"
            self.answer_column = "answer"
            
            if self.prompt_template is None:
                self.prompt_template = "{question}\n{choices}\nAnswer:"
    
    def _get_metric_names(self) -> List[str]:
        """
        Get the list of metric names for this task.
        
        Returns:
            List[str]: List of metric names
        """
        return ["accuracy", "f1_macro", "f1_micro"]
    
    def _get_primary_metric(self) -> str:
        """
        Get the primary metric name for this task.
        
        Returns:
            str: Primary metric name
        """
        return "accuracy"
    
    def load_data(self) -> None:
        """
        Load and preprocess the task data.
        """
        dataset_config = self.config.dataset
        dataset_name = dataset_config.name
        
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Load dataset from Hugging Face
            if "path" in dataset_config:
                # Load from local path
                dataset = load_dataset(
                    dataset_config.path,
                    name=dataset_config.get("subset", None),
                    cache_dir=dataset_config.get("cache_dir", None)
                )
            else:
                # Load from Hugging Face Hub
                dataset = load_dataset(
                    dataset_name,
                    name=dataset_config.get("subset", None),
                    cache_dir=dataset_config.get("cache_dir", None)
                )
            
            # Process splits
            self.train_dataset = self._process_split(dataset.get("train", None))
            self.val_dataset = self._process_split(dataset.get("validation", None))
            self.test_dataset = self._process_split(dataset.get("test", None))
            
            # If no validation set, create one from training set
            if self.val_dataset is None and self.train_dataset is not None:
                self.logger.info("Creating validation set from training set")
                val_size = min(1000, len(self.train_dataset.data) // 10)
                
                # Split training data
                train_data = self.train_dataset.data[:-val_size]
                val_data = self.train_dataset.data[-val_size:]
                
                self.train_dataset = MultipleChoiceDataset(
                    train_data, self.tokenizer, self.max_length, self.prompt_template
                )
                self.val_dataset = MultipleChoiceDataset(
                    val_data, self.tokenizer, self.max_length, self.prompt_template
                )
            
            self.logger.info(f"Data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _process_split(self, split_data: Optional[HFDataset]) -> Optional[MultipleChoiceDataset]:
        """
        Process a data split.
        
        Args:
            split_data: Raw split data
            
        Returns:
            MultipleChoiceDataset or None
        """
        if split_data is None:
            return None
        
        processed_data = []
        
        for item in split_data:
            processed_item = self._process_item(item)
            if processed_item is not None:
                processed_data.append(processed_item)
        
        if not processed_data:
            return None
        
        return MultipleChoiceDataset(
            processed_data, 
            self.tokenizer, 
            self.max_length, 
            self.prompt_template
        )
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single data item.
        
        Args:
            item: Raw data item
            
        Returns:
            Processed item or None if invalid
        """
        try:
            # Extract question
            question = item[self.question_column]
            
            # Extract choices
            choices = self._extract_choices(item)
            
            # Extract answer
            answer = self._extract_answer(item)
            
            # Validate
            if not question or not choices or answer is None:
                return None
            
            if answer >= len(choices):
                return None
            
            processed_item = {
                "question": question,
                "choices": choices,
                "answer": answer,
                "id": item.get("id", None)
            }
            
            # Add subject if available (for MMLU)
            if "subject" in item:
                processed_item["subject"] = item["subject"]
            
            return processed_item
            
        except Exception as e:
            self.logger.warning(f"Failed to process item: {e}")
            return None
    
    def _extract_choices(self, item: Dict[str, Any]) -> List[str]:
        """
        Extract choices from data item.
        
        Args:
            item: Data item
            
        Returns:
            List[str]: List of choices
        """
        dataset_name = self.config.dataset.name.lower()
        
        if dataset_name == "winogrande":
            # Winogrande has option1 and option2
            return [item["option1"], item["option2"]]
        
        elif dataset_name == "piqa":
            # PIQA has sol1 and sol2
            return [item["sol1"], item["sol2"]]
        
        elif dataset_name == "hellaswag":
            # HellaSwag has endings list
            return item["endings"]
        
        elif dataset_name in ["arc_easy", "arc_challenge"]:
            # ARC has choices with text and label
            choices_data = item["choices"]
            if isinstance(choices_data, dict) and "text" in choices_data:
                return choices_data["text"]
            elif isinstance(choices_data, list):
                return [choice["text"] if isinstance(choice, dict) else str(choice) 
                       for choice in choices_data]
        
        else:
            # Default: assume choices is a list
            choices = item.get(self.choices_column, [])
            if isinstance(choices, list):
                return [str(choice) for choice in choices]
            else:
                return [str(choices)]
    
    def _extract_answer(self, item: Dict[str, Any]) -> Optional[int]:
        """
        Extract answer index from data item.
        
        Args:
            item: Data item
            
        Returns:
            Answer index or None if invalid
        """
        answer = item.get(self.answer_column)
        
        if answer is None:
            return None
        
        # Handle different answer formats
        if isinstance(answer, int):
            return answer
        
        elif isinstance(answer, str):
            # Convert letter to index (A=0, B=1, C=2, D=3)
            if len(answer) == 1 and answer.upper() in "ABCDEFGHIJ":
                return ord(answer.upper()) - ord("A")
            
            # Try to convert to int
            try:
                return int(answer)
            except ValueError:
                return None
        
        return None
    
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
        def collate_fn(batch):
            """Custom collate function for multiple choice data."""
            # Stack choice inputs
            batch_size = len(batch)
            num_choices = len(batch[0]["choice_inputs"])
            
            # Initialize tensors
            input_ids = torch.zeros(batch_size, num_choices, self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, num_choices, self.max_length, dtype=torch.long)
            labels = torch.zeros(batch_size, dtype=torch.long)
            
            # Fill tensors
            for i, item in enumerate(batch):
                labels[i] = item["label"]
                for j, choice_input in enumerate(item["choice_inputs"]):
                    input_ids[i, j] = choice_input["input_ids"]
                    attention_mask[i, j] = choice_input["attention_mask"]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "questions": [item["question"] for item in batch],
                "choices": [item["choices"] for item in batch],
                "subjects": [item["subject"] for item in batch],
                "ids": [item["id"] for item in batch]
            }
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        ) if self.train_dataset else None
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        ) if self.val_dataset else None
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        ) if self.test_dataset else None
        
        return train_loader, val_loader, test_loader
    
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
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        subject_results = {}
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Reshape for model input
                batch_size, num_choices, seq_len = input_ids.shape
                input_ids = input_ids.view(-1, seq_len)
                attention_mask = attention_mask.view(-1, seq_len)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Compute choice scores
                # For multiple choice, we typically use the log probability of the next token
                # or the average log probability of the sequence
                choice_scores = self._compute_choice_scores(logits, input_ids, batch_size, num_choices)
                
                # Get predictions
                predictions = torch.argmax(choice_scores, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(choice_scores.cpu().numpy())
                
                # Track subject-wise results (for MMLU)
                if "subjects" in batch and batch["subjects"][0]:  # Check if subjects are provided
                    for pred, label, subject in zip(predictions.cpu().numpy(), 
                                                   labels.cpu().numpy(), 
                                                   batch["subjects"]):
                        if subject not in subject_results:
                            subject_results[subject] = {"correct": 0, "total": 0}
                        
                        subject_results[subject]["total"] += 1
                        if pred == label:
                            subject_results[subject]["correct"] += 1
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average="macro")
        f1_micro = f1_score(all_labels, all_predictions, average="micro")
        
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }
        
        # Add subject-wise metrics
        if subject_results:
            subject_accuracies = {}
            for subject, results in subject_results.items():
                subject_accuracies[f"accuracy_{subject}"] = results["correct"] / results["total"]
            
            metrics.update(subject_accuracies)
            
            # Compute average subject accuracy
            avg_subject_accuracy = np.mean(list(subject_accuracies.values()))
            metrics["avg_subject_accuracy"] = avg_subject_accuracy
        
        # Detailed results
        detailed_results = {
            "predictions": all_predictions,
            "labels": all_labels,
            "logits": all_logits,
            "subject_results": subject_results
        }
        
        return TaskMetrics(
            primary_metric=self.primary_metric,
            metrics=metrics,
            detailed_results=detailed_results
        )
    
    def _compute_choice_scores(self, 
                              logits: torch.Tensor, 
                              input_ids: torch.Tensor,
                              batch_size: int, 
                              num_choices: int) -> torch.Tensor:
        """
        Compute scores for each choice.
        
        Args:
            logits: Model logits [batch_size * num_choices, seq_len, vocab_size]
            input_ids: Input token IDs [batch_size * num_choices, seq_len]
            batch_size: Batch size
            num_choices: Number of choices
            
        Returns:
            torch.Tensor: Choice scores [batch_size, num_choices]
        """
        # Reshape logits
        seq_len, vocab_size = logits.shape[1], logits.shape[2]
        logits = logits.view(batch_size, num_choices, seq_len, vocab_size)
        input_ids = input_ids.view(batch_size, num_choices, seq_len)
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Method 1: Average log probability of the sequence
        # Shift logits and input_ids for next token prediction
        shift_logits = log_probs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Gather log probabilities for actual tokens
        token_log_probs = torch.gather(
            shift_logits.view(-1, vocab_size),
            1,
            shift_labels.view(-1, 1)
        ).view(batch_size, num_choices, -1)
        
        # Mask padding tokens
        attention_mask = (shift_labels != self.tokenizer.pad_token_id).float()
        
        # Compute average log probability per choice
        choice_scores = (token_log_probs * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
        
        return choice_scores
    
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of data for model input.
        
        Args:
            batch: Raw batch data
            
        Returns:
            Dict[str, torch.Tensor]: Preprocessed batch
        """
        # This method is called by the base class for sample input generation
        # For multiple choice, we need to handle the choice structure
        
        if "choice_inputs" in batch:
            # Already preprocessed by dataset
            return batch
        
        # If raw batch, process it
        processed_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                processed_batch[key] = value
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    processed_batch[key] = torch.stack(value)
                else:
                    processed_batch[key] = value
            else:
                processed_batch[key] = value
        
        return processed_batch