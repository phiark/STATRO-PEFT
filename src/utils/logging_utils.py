#!/usr/bin/env python3
"""
Logging utilities for STRATO-PEFT experimental framework.

This module provides comprehensive logging setup and utilities for tracking
experiment progress, system information, and debugging.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import psutil
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    log_dir: str,
    log_level: str = "INFO",
    experiment_name: str = "strato_peft",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging for the experiment.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        experiment_name: Name of the experiment for log file naming
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler with Rich
    if console_output:
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(rich_handler)
    
    # File handlers
    if file_output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Main log file
        main_log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        file_handler = logging.FileHandler(main_log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_dir / f"{experiment_name}_errors_{timestamp}.log"
        error_handler = logging.FileHandler(error_log_file, mode='w', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Debug log file (if debug level)
        if log_level.upper() == "DEBUG":
            debug_log_file = log_dir / f"{experiment_name}_debug_{timestamp}.log"
            debug_handler = logging.FileHandler(debug_log_file, mode='w', encoding='utf-8')
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(detailed_formatter)
            logger.addHandler(debug_handler)
    
    # Log initial setup information
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir}")
    
    return logger


def log_system_info(logger: logging.Logger) -> Dict[str, Any]:
    """
    Log comprehensive system information.
    
    Args:
        logger: Logger instance
        
    Returns:
        Dict[str, Any]: System information dictionary
    """
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    
    # Basic system info
    system_info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "python_executable": sys.executable,
    }
    
    logger.info(f"Platform: {system_info['platform']}")
    logger.info(f"Python Version: {system_info['python_version']}")
    logger.info(f"Python Executable: {system_info['python_executable']}")
    
    # CPU information
    try:
        cpu_info = {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }
        system_info.update(cpu_info)
        
        logger.info(f"CPU Count (Logical): {cpu_info['cpu_count']}")
        logger.info(f"CPU Count (Physical): {cpu_info['cpu_count_physical']}")
        if cpu_info['cpu_freq']:
            logger.info(f"CPU Frequency: {cpu_info['cpu_freq']}")
    except Exception as e:
        logger.warning(f"Could not get CPU info: {e}")
    
    # Memory information
    try:
        memory = psutil.virtual_memory()
        memory_info = {
            "total_memory_gb": round(memory.total / (1024**3), 2),
            "available_memory_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent,
        }
        system_info.update(memory_info)
        
        logger.info(f"Total Memory: {memory_info['total_memory_gb']} GB")
        logger.info(f"Available Memory: {memory_info['available_memory_gb']} GB")
        logger.info(f"Memory Usage: {memory_info['memory_percent']}%")
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
    
    # GPU information
    gpu_info = log_gpu_info(logger)
    system_info.update(gpu_info)
    
    # PyTorch information
    torch_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    system_info.update(torch_info)
    
    logger.info(f"PyTorch Version: {torch_info['torch_version']}")
    logger.info(f"CUDA Available: {torch_info['cuda_available']}")
    if torch_info['cuda_available']:
        logger.info(f"CUDA Version: {torch_info['cuda_version']}")
        logger.info(f"cuDNN Version: {torch_info['cudnn_version']}")
    
    # Environment variables
    important_env_vars = [
        "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HOME", "TRANSFORMERS_CACHE",
        "WANDB_PROJECT", "WANDB_ENTITY", "OMP_NUM_THREADS"
    ]
    
    env_info = {}
    logger.info("Important Environment Variables:")
    for var in important_env_vars:
        value = os.environ.get(var)
        env_info[var] = value
        logger.info(f"  {var}: {value}")
    
    system_info["environment"] = env_info
    
    logger.info("=" * 60)
    
    return system_info


def log_gpu_info(logger: logging.Logger) -> Dict[str, Any]:
    """
    Log detailed GPU information.
    
    Args:
        logger: Logger instance
        
    Returns:
        Dict[str, Any]: GPU information dictionary
    """
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpus": []
    }
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available - running on CPU")
        return gpu_info
    
    gpu_count = torch.cuda.device_count()
    gpu_info["gpu_count"] = gpu_count
    
    logger.info(f"GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        try:
            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            
            # Get memory info
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = props.total_memory
            
            gpu_data = {
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(memory_total / (1024**3), 2),
                "allocated_memory_gb": round(memory_allocated / (1024**3), 2),
                "reserved_memory_gb": round(memory_reserved / (1024**3), 2),
                "free_memory_gb": round((memory_total - memory_reserved) / (1024**3), 2),
                "multiprocessor_count": props.multi_processor_count,
            }
            
            gpu_info["gpus"].append(gpu_data)
            
            logger.info(f"GPU {i}: {gpu_data['name']}")
            logger.info(f"  Compute Capability: {gpu_data['compute_capability']}")
            logger.info(f"  Total Memory: {gpu_data['total_memory_gb']} GB")
            logger.info(f"  Free Memory: {gpu_data['free_memory_gb']} GB")
            logger.info(f"  Multiprocessors: {gpu_data['multiprocessor_count']}")
            
        except Exception as e:
            logger.warning(f"Could not get info for GPU {i}: {e}")
    
    return gpu_info


def log_model_info(logger: logging.Logger, model: torch.nn.Module, model_name: str) -> Dict[str, Any]:
    """
    Log detailed model information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        model_name: Name of the model
        
    Returns:
        Dict[str, Any]: Model information dictionary
    """
    logger.info("=" * 60)
    logger.info(f"MODEL INFORMATION: {model_name}")
    logger.info("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    model_info = {
        "name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
    }
    
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Frozen Parameters: {frozen_params:,}")
    logger.info(f"Trainable Percentage: {model_info['trainable_percentage']:.2f}%")
    
    # Model size estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    model_info.update({
        "parameter_size_mb": round(param_size / (1024**2), 2),
        "buffer_size_mb": round(buffer_size / (1024**2), 2),
        "total_size_mb": round(total_size / (1024**2), 2),
    })
    
    logger.info(f"Parameter Size: {model_info['parameter_size_mb']} MB")
    logger.info(f"Buffer Size: {model_info['buffer_size_mb']} MB")
    logger.info(f"Total Model Size: {model_info['total_size_mb']} MB")
    
    # Device information
    devices = set()
    for param in model.parameters():
        devices.add(str(param.device))
    
    model_info["devices"] = list(devices)
    logger.info(f"Model Devices: {list(devices)}")
    
    logger.info("=" * 60)
    
    return model_info


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    metrics: Optional[Dict[str, float]] = None,
    gpu_memory: bool = True
) -> None:
    """
    Log training progress information.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        step: Current step
        total_steps: Total number of steps
        loss: Current loss value
        lr: Current learning rate
        metrics: Additional metrics to log
        gpu_memory: Whether to log GPU memory usage
    """
    progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
    
    log_msg = f"Epoch {epoch} | Step {step}/{total_steps} ({progress_pct:.1f}%) | "
    log_msg += f"Loss: {loss:.4f} | LR: {lr:.2e}"
    
    if metrics:
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        log_msg += f" | {' | '.join(metric_strs)}"
    
    if gpu_memory and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        log_msg += f" | GPU Mem: {memory_allocated:.1f}/{memory_reserved:.1f} GB"
    
    logger.info(log_msg)


def create_experiment_summary_table(config: Dict[str, Any], results: Dict[str, Any]) -> Table:
    """
    Create a Rich table summarizing the experiment.
    
    Args:
        config: Experiment configuration
        results: Experiment results
        
    Returns:
        Table: Rich table with experiment summary
    """
    table = Table(title="🧪 Experiment Summary")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Parameter", style="blue")
    table.add_column("Value", style="green")
    
    # Configuration
    table.add_row("Config", "Experiment Name", str(config.get("experiment_name", "N/A")))
    table.add_row("Config", "Model", str(config.get("model", "N/A")))
    table.add_row("Config", "Task", str(config.get("task", "N/A")))
    table.add_row("Config", "PEFT Method", str(config.get("peft_method", "N/A")))
    table.add_row("Config", "Seed", str(config.get("seed", "N/A")))
    
    # Results
    for key, value in results.items():
        if isinstance(value, float):
            table.add_row("Results", key, f"{value:.4f}")
        else:
            table.add_row("Results", key, str(value))
    
    return table


class ExperimentLogger:
    """
    Comprehensive experiment logger with context management.
    """
    
    def __init__(self, logger: logging.Logger, experiment_name: str):
        self.logger = logger
        self.experiment_name = experiment_name
        self.start_time = None
        self.phase_times = {}
    
    def start_experiment(self):
        """Start experiment timing."""
        self.start_time = time.time()
        self.logger.info(f"🚀 Starting experiment: {self.experiment_name}")
    
    def start_phase(self, phase_name: str):
        """Start a new phase."""
        self.phase_times[phase_name] = time.time()
        self.logger.info(f"📍 Starting phase: {phase_name}")
    
    def end_phase(self, phase_name: str):
        """End a phase and log duration."""
        if phase_name in self.phase_times:
            duration = time.time() - self.phase_times[phase_name]
            self.logger.info(f"✅ Completed phase: {phase_name} (Duration: {duration:.2f}s)")
            return duration
        return None
    
    def end_experiment(self):
        """End experiment and log total duration."""
        if self.start_time:
            total_duration = time.time() - self.start_time
            self.logger.info(f"🏁 Experiment completed: {self.experiment_name}")
            self.logger.info(f"⏱️  Total duration: {total_duration:.2f}s ({total_duration/60:.2f}m)")
            return total_duration
        return None
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """Log checkpoint save."""
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        error_msg = f"❌ Error in {context}: {str(error)}" if context else f"❌ Error: {str(error)}"
        self.logger.error(error_msg, exc_info=True)