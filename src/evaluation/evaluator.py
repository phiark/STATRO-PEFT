#!/usr/bin/env python3
"""
Core evaluator for STRATO-PEFT experimental framework.

This module provides the main evaluation interface for assessing PEFT methods
across multiple dimensions including performance, efficiency, and robustness.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from ..tasks.base_task import BaseTask
from ..peft.base_peft import BasePEFT
from .metrics import MetricCalculator, PerformanceMetrics, EfficiencyMetrics


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation.
    """
    # Basic evaluation settings
    batch_size: int = 16
    max_samples: Optional[int] = None
    device: str = 'auto'
    
    # Performance evaluation
    compute_performance: bool = True
    performance_splits: List[str] = field(default_factory=lambda: ['test', 'validation'])
    
    # Efficiency evaluation
    compute_efficiency: bool = True
    measure_inference_time: bool = True
    measure_memory_usage: bool = True
    measure_flops: bool = True
    warmup_steps: int = 10
    timing_steps: int = 100
    
    # Robustness evaluation
    compute_robustness: bool = False
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    
    # Output settings
    save_predictions: bool = False
    save_detailed_metrics: bool = True
    output_dir: Optional[str] = None


@dataclass
class EvaluationResult:
    """
    Results from model evaluation.
    """
    # Basic information
    model_name: str
    peft_method: str
    task_name: str
    timestamp: float = field(default_factory=time.time)
    
    # Performance metrics
    performance: Optional[PerformanceMetrics] = None
    
    # Efficiency metrics
    efficiency: Optional[EfficiencyMetrics] = None
    
    # Robustness metrics
    robustness: Optional[Dict[str, Any]] = None
    
    # Raw predictions and labels
    predictions: Optional[List[Any]] = None
    labels: Optional[List[Any]] = None
    
    # Additional metadata
    config: Optional[Dict[str, Any]] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        result = {
            'model_name': self.model_name,
            'peft_method': self.peft_method,
            'task_name': self.task_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'extra_metrics': self.extra_metrics
        }
        
        if self.performance:
            result['performance'] = self.performance.__dict__
        
        if self.efficiency:
            result['efficiency'] = self.efficiency.__dict__
        
        if self.robustness:
            result['robustness'] = self.robustness
        
        # Only include predictions/labels if they exist and are serializable
        if self.predictions and len(self.predictions) < 10000:  # Avoid huge files
            result['predictions'] = self.predictions
        if self.labels and len(self.labels) < 10000:
            result['labels'] = self.labels
        
        return result
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save evaluation result to file.
        
        Args:
            filepath: Path to save the result
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'EvaluationResult':
        """
        Load evaluation result from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded evaluation result
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct dataclass instances
        result = cls(
            model_name=data['model_name'],
            peft_method=data['peft_method'],
            task_name=data['task_name'],
            timestamp=data.get('timestamp', time.time()),
            config=data.get('config'),
            extra_metrics=data.get('extra_metrics', {}),
            predictions=data.get('predictions'),
            labels=data.get('labels')
        )
        
        # Reconstruct performance metrics
        if 'performance' in data:
            perf_data = data['performance']
            result.performance = PerformanceMetrics(**perf_data)
        
        # Reconstruct efficiency metrics
        if 'efficiency' in data:
            eff_data = data['efficiency']
            result.efficiency = EfficiencyMetrics(**eff_data)
        
        # Robustness metrics
        if 'robustness' in data:
            result.robustness = data['robustness']
        
        return result


class Evaluator:
    """
    Main evaluator for PEFT methods.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: BaseTask,
        peft_method: Optional[BasePEFT] = None,
        config: Optional[EvaluationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            task: Task to evaluate on
            peft_method: Optional PEFT method being evaluated
            config: Evaluation configuration
            logger: Optional logger
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.peft_method = peft_method
        self.config = config or EvaluationConfig()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Setup device
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metric calculator
        self.metric_calculator = MetricCalculator()
        
        self.logger.info(f"Evaluator initialized on device: {self.device}")
        self.logger.info(f"Task: {self.task.__class__.__name__}")
        if self.peft_method:
            self.logger.info(f"PEFT method: {self.peft_method.__class__.__name__}")
    
    def evaluate(
        self,
        splits: Optional[List[str]] = None,
        return_predictions: bool = False
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation.
        
        Args:
            splits: Data splits to evaluate on
            return_predictions: Whether to return predictions
            
        Returns:
            Evaluation results
        """
        self.logger.info("Starting comprehensive evaluation")
        
        # Use default splits if not provided
        if splits is None:
            splits = self.config.performance_splits
        
        # Initialize result
        result = EvaluationResult(
            model_name=getattr(self.model, 'name_or_path', 'unknown'),
            peft_method=self.peft_method.__class__.__name__ if self.peft_method else 'none',
            task_name=self.task.__class__.__name__,
            config=self.config.__dict__.copy()
        )
        
        try:
            # Performance evaluation
            if self.config.compute_performance:
                self.logger.info("Computing performance metrics")
                performance_result = self._evaluate_performance(splits, return_predictions)
                result.performance = performance_result['metrics']
                if return_predictions:
                    result.predictions = performance_result.get('predictions')
                    result.labels = performance_result.get('labels')
            
            # Efficiency evaluation
            if self.config.compute_efficiency:
                self.logger.info("Computing efficiency metrics")
                result.efficiency = self._evaluate_efficiency()
            
            # Robustness evaluation
            if self.config.compute_robustness:
                self.logger.info("Computing robustness metrics")
                result.robustness = self._evaluate_robustness()
            
            # Save results if output directory is specified
            if self.config.output_dir:
                self._save_evaluation_result(result)
            
            self.logger.info("Evaluation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
        
        return result
    
    def _evaluate_performance(
        self,
        splits: List[str],
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on specified splits.
        
        Args:
            splits: Data splits to evaluate
            return_predictions: Whether to return predictions
            
        Returns:
            Performance evaluation results
        """
        all_predictions = []
        all_labels = []
        split_metrics = {}
        
        for split in splits:
            self.logger.info(f"Evaluating on {split} split")
            
            # Create dataloader
            dataloader = self.task.create_dataloader(
                split=split,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Evaluate on this split
            split_result = self._evaluate_on_dataloader(dataloader, return_predictions)
            split_metrics[split] = split_result['metrics']
            
            if return_predictions:
                all_predictions.extend(split_result['predictions'])
                all_labels.extend(split_result['labels'])
        
        # Compute overall metrics
        if len(splits) == 1:
            overall_metrics = split_metrics[splits[0]]
        else:
            # Average metrics across splits
            overall_metrics = self._average_metrics(split_metrics)
        
        # Create performance metrics object
        performance = PerformanceMetrics(
            accuracy=overall_metrics.get('accuracy', 0.0),
            f1_score=overall_metrics.get('f1', 0.0),
            precision=overall_metrics.get('precision', 0.0),
            recall=overall_metrics.get('recall', 0.0),
            loss=overall_metrics.get('loss', 0.0),
            perplexity=overall_metrics.get('perplexity'),
            bleu_score=overall_metrics.get('bleu'),
            rouge_scores=overall_metrics.get('rouge'),
            task_specific_metrics=overall_metrics.get('task_specific', {}),
            split_metrics=split_metrics
        )
        
        result = {'metrics': performance}
        if return_predictions:
            result['predictions'] = all_predictions
            result['labels'] = all_labels
        
        return result
    
    def _evaluate_on_dataloader(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
            return_predictions: Whether to return predictions
            
        Returns:
            Evaluation results for this dataloader
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # Limit samples if specified
                if (self.config.max_samples and 
                    batch_idx * self.config.batch_size >= self.config.max_samples):
                    break
                
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Extract loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                
                # Extract predictions and labels
                predictions = self._extract_predictions(outputs, batch)
                labels = self._extract_labels(batch)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        task_metrics = self.task.evaluate(all_predictions, all_labels)
        
        # Combine metrics
        metrics = {'loss': avg_loss, **task_metrics}
        
        result = {'metrics': metrics}
        if return_predictions:
            result['predictions'] = all_predictions
            result['labels'] = all_labels
        
        return result
    
    def _evaluate_efficiency(self) -> EfficiencyMetrics:
        """
        Evaluate model efficiency.
        
        Returns:
            Efficiency metrics
        """
        efficiency_metrics = EfficiencyMetrics()
        
        # Model size metrics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        efficiency_metrics.total_parameters = total_params
        efficiency_metrics.trainable_parameters = trainable_params
        efficiency_metrics.parameter_efficiency = trainable_params / total_params if total_params > 0 else 0.0
        
        # Memory usage
        if self.config.measure_memory_usage and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run a forward pass to measure memory
            self._measure_memory_usage(efficiency_metrics)
        
        # Inference time
        if self.config.measure_inference_time:
            self._measure_inference_time(efficiency_metrics)
        
        # FLOPs
        if self.config.measure_flops:
            self._measure_flops(efficiency_metrics)
        
        # PEFT-specific metrics
        if self.peft_method:
            peft_metrics = self.peft_method.get_peft_metrics()
            efficiency_metrics.peft_overhead = peft_metrics.memory_overhead
            efficiency_metrics.adaptation_efficiency = peft_metrics.adaptation_efficiency
        
        return efficiency_metrics
    
    def _evaluate_robustness(self) -> Dict[str, Any]:
        """
        Evaluate model robustness.
        
        Returns:
            Robustness metrics
        """
        robustness_metrics = {}
        
        # Get clean performance baseline
        clean_dataloader = self.task.create_dataloader(
            split='test',
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        clean_result = self._evaluate_on_dataloader(clean_dataloader)
        clean_accuracy = clean_result['metrics'].get('accuracy', 0.0)
        
        # Test with different noise levels
        noise_results = {}
        for noise_level in self.config.noise_levels:
            self.logger.info(f"Testing robustness with noise level: {noise_level}")
            
            # Add noise to inputs and evaluate
            noisy_accuracy = self._evaluate_with_noise(clean_dataloader, noise_level)
            noise_results[f'noise_{noise_level}'] = {
                'accuracy': noisy_accuracy,
                'degradation': clean_accuracy - noisy_accuracy
            }
        
        robustness_metrics['noise_robustness'] = noise_results
        robustness_metrics['clean_accuracy'] = clean_accuracy
        
        return robustness_metrics
    
    def _measure_memory_usage(self, efficiency_metrics: EfficiencyMetrics) -> None:
        """
        Measure memory usage during inference.
        
        Args:
            efficiency_metrics: Metrics object to update
        """
        if not torch.cuda.is_available():
            return
        
        # Create a sample batch
        sample_dataloader = self.task.create_dataloader(
            split='test',
            batch_size=1,
            shuffle=False
        )
        
        sample_batch = next(iter(sample_dataloader))
        sample_batch = self._move_to_device(sample_batch)
        
        # Measure memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = self.model(**sample_batch)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        efficiency_metrics.peak_memory_usage = peak_memory
    
    def _measure_inference_time(self, efficiency_metrics: EfficiencyMetrics) -> None:
        """
        Measure inference time.
        
        Args:
            efficiency_metrics: Metrics object to update
        """
        # Create a sample batch
        sample_dataloader = self.task.create_dataloader(
            split='test',
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        sample_batch = next(iter(sample_dataloader))
        sample_batch = self._move_to_device(sample_batch)
        
        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.config.warmup_steps):
                _ = self.model(**sample_batch)
        
        # Measure timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(self.config.timing_steps):
                _ = self.model(**sample_batch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / self.config.timing_steps
        throughput = self.config.batch_size / avg_time
        
        efficiency_metrics.inference_time = avg_time
        efficiency_metrics.throughput = throughput
    
    def _measure_flops(self, efficiency_metrics: EfficiencyMetrics) -> None:
        """
        Measure FLOPs using a basic estimation approach.
        
        Args:
            efficiency_metrics: Metrics object to update
        """
        try:
            # Try to use fvcore if available
            try:
                from fvcore.nn import FlopCountMode, flop_count
                
                # Create a sample input
                sample_input = self._create_sample_input()
                
                # Count FLOPs
                flop_dict, _ = flop_count(
                    self.model,
                    sample_input,
                    supported_ops=None
                )
                
                total_flops = sum(flop_dict.values())
                efficiency_metrics.flops = total_flops
                
            except ImportError:
                # Fallback to manual estimation
                total_flops = self._estimate_flops_manual()
                efficiency_metrics.flops = total_flops
                self.logger.info("Using manual FLOPs estimation (install fvcore for accurate measurement)")
                
        except Exception as e:
            self.logger.warning(f"FLOPs measurement failed: {e}")
            efficiency_metrics.flops = None
    
    def _create_sample_input(self) -> Dict[str, torch.Tensor]:
        """
        Create a sample input for FLOPs measurement.
        
        Returns:
            Sample input dictionary
        """
        batch_size = 1
        seq_length = 512
        
        # Create dummy input
        sample_input = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=self.device),
            'attention_mask': torch.ones((batch_size, seq_length), device=self.device)
        }
        
        return sample_input
    
    def _estimate_flops_manual(self) -> int:
        """
        Manual FLOPs estimation for transformer models.
        
        Returns:
            Estimated FLOPs count
        """
        # Basic estimation for transformer models
        # This is a rough approximation
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Rough estimation: 2 FLOPs per parameter for forward pass
        # This is a very basic approximation
        estimated_flops = total_params * 2
        
        # Scale by sequence length and batch size
        seq_length = 512  # Assume default
        batch_size = 1
        
        # Attention mechanism scaling (quadratic in sequence length)
        if hasattr(self.model.config, 'num_attention_heads'):
            num_heads = self.model.config.num_attention_heads
            hidden_size = self.model.config.hidden_size
            head_dim = hidden_size // num_heads
            
            # Attention FLOPs: O(seq_len^2 * hidden_size)
            attention_flops = seq_length * seq_length * hidden_size * batch_size
            estimated_flops += attention_flops
        
        return int(estimated_flops)
    
    def _evaluate_with_noise(self, dataloader: DataLoader, noise_level: float) -> float:
        """
        Evaluate model with noisy inputs.
        
        Args:
            dataloader: DataLoader to evaluate on
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Accuracy with noisy inputs
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                
                # Add noise to input embeddings
                if 'input_ids' in batch:
                    # Get embeddings
                    embeddings = self.model.get_input_embeddings()(batch['input_ids'])
                    
                    # Add Gaussian noise
                    noise = torch.randn_like(embeddings) * noise_level
                    noisy_embeddings = embeddings + noise
                    
                    # Forward pass with noisy embeddings
                    # This is a simplified approach - in practice, you'd need to
                    # modify the model to accept embeddings directly
                    outputs = self.model(inputs_embeds=noisy_embeddings, 
                                       attention_mask=batch.get('attention_mask'))
                else:
                    # Fallback to normal forward pass
                    outputs = self.model(**batch)
                
                predictions = self._extract_predictions(outputs, batch)
                labels = self._extract_labels(batch)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Compute accuracy
        task_metrics = self.task.evaluate(all_predictions, all_labels)
        return task_metrics.get('accuracy', 0.0)
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch to the appropriate device.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch moved to device
        """
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    
    def _extract_predictions(self, outputs: Any, batch: Dict[str, Any]) -> List[Any]:
        """
        Extract predictions from model outputs.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            List of predictions
        """
        # Use task-specific prediction extraction
        return self.task.extract_predictions(outputs, batch)
    
    def _extract_labels(self, batch: Dict[str, Any]) -> List[Any]:
        """
        Extract labels from batch.
        
        Args:
            batch: Input batch
            
        Returns:
            List of labels
        """
        # Use task-specific label extraction
        return self.task.extract_labels(batch)
    
    def _average_metrics(self, split_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Average metrics across splits.
        
        Args:
            split_metrics: Metrics for each split
            
        Returns:
            Averaged metrics
        """
        if not split_metrics:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for metrics in split_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Average each metric
        averaged = {}
        for metric in all_metrics:
            values = [metrics.get(metric, 0.0) for metrics in split_metrics.values() 
                     if metric in metrics]
            if values:
                averaged[metric] = np.mean(values)
        
        return averaged
    
    def _save_evaluation_result(self, result: EvaluationResult) -> None:
        """
        Save evaluation result to output directory.
        
        Args:
            result: Evaluation result to save
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{result.peft_method}_{result.task_name}_{timestamp}.json"
        filepath = output_dir / filename
        
        # Save result
        result.save(filepath)
        self.logger.info(f"Evaluation result saved to: {filepath}")