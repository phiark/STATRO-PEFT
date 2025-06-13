g#!/usr/bin/env python3
"""
Configuration utilities for STRATO-PEFT experimental framework.

This module provides utilities for loading, validating, and managing configurations
across different training environments (CPU, GPU, ROCm, CUDA, Apple Silicon).
Designed to be Docker-compatible and platform-agnostic.

Key Features:
1. Multi-platform device detection and configuration
2. Docker environment detection and adaptation
3. Configuration validation and merging
4. Environment-specific optimizations
5. Resource constraint handling

Author: STRATO-PEFT Research Team
Date: 2024
"""

import os
import sys
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import yaml
import json
from dataclasses import dataclass, asdict

from omegaconf import DictConfig, OmegaConf
import torch
import psutil


@dataclass
class DeviceInfo:
    """Device information for training configuration."""
    device_type: str  # 'cpu', 'cuda', 'mps', 'rocm'
    device_count: int
    memory_per_device: float  # GB
    total_memory: float  # GB
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    is_docker: bool = False
    platform_info: str = ""


@dataclass
class TrainingEnvironment:
    """Training environment configuration."""
    device_info: DeviceInfo
    batch_size: int
    gradient_accumulation_steps: int
    mixed_precision: bool
    dataloader_num_workers: int
    pin_memory: bool
    persistent_workers: bool
    compile_model: bool
    optimization_level: str  # 'O0', 'O1', 'O2', 'O3'


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class PlatformDetector:
    """
    Detects platform capabilities and constraints.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._device_info = None
        
    def detect_device_info(self) -> DeviceInfo:
        """
        Detect available compute devices and their capabilities.
        
        Returns:
            DeviceInfo object with detected device information
        """
        if self._device_info is not None:
            return self._device_info
            
        self.logger.info("Detecting compute devices...")
        
        # Detect Docker environment
        is_docker = self._is_running_in_docker()
        platform_info = f"{platform.system()} {platform.release()} {platform.machine()}"
        
        # Try different device types in order of preference
        device_info = None
        
        # 1. Try CUDA
        if torch.cuda.is_available():
            device_info = self._detect_cuda_info()
            
        # 2. Try ROCm (AMD GPU)
        elif self._is_rocm_available():
            device_info = self._detect_rocm_info()
            
        # 3. Try Apple Silicon (MPS)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info = self._detect_mps_info()
            
        # 4. Fallback to CPU
        else:
            device_info = self._detect_cpu_info()
        
        device_info.is_docker = is_docker
        device_info.platform_info = platform_info
        
        self._device_info = device_info
        self.logger.info(f"Detected device: {device_info.device_type} with {device_info.device_count} device(s)")
        self.logger.info(f"Total memory: {device_info.total_memory:.2f} GB")
        
        return device_info
    
    def _is_running_in_docker(self) -> bool:
        """
        Check if running inside Docker container.
        
        Returns:
            True if running in Docker, False otherwise
        """
        # Check for .dockerenv file
        if Path('/.dockerenv').exists():
            return True
            
        # Check cgroup for docker
        try:
            with open('/proc/1/cgroup', 'r') as f:
                return 'docker' in f.read()
        except (FileNotFoundError, PermissionError):
            pass
            
        # Check environment variables
        return any(key in os.environ for key in ['DOCKER_CONTAINER', 'CONTAINER_ID'])
    
    def _detect_cuda_info(self) -> DeviceInfo:
        """Detect CUDA device information."""
        device_count = torch.cuda.device_count()
        total_memory = 0.0
        memory_per_device = 0.0
        
        if device_count > 0:
            # Get memory info from first device
            memory_per_device = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            total_memory = memory_per_device * device_count
            
            # Get compute capability
            props = torch.cuda.get_device_properties(0)
            compute_capability = f"{props.major}.{props.minor}"
            
            # Get driver version
            try:
                driver_version = torch.version.cuda
            except:
                driver_version = "unknown"
        else:
            compute_capability = None
            driver_version = None
        
        return DeviceInfo(
            device_type='cuda',
            device_count=device_count,
            memory_per_device=memory_per_device,
            total_memory=total_memory,
            compute_capability=compute_capability,
            driver_version=driver_version
        )
    
    def _is_rocm_available(self) -> bool:
        """Check if ROCm is available."""
        try:
            # Check for ROCm installation
            result = subprocess.run(['rocm-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _detect_rocm_info(self) -> DeviceInfo:
        """Detect ROCm device information."""
        try:
            # Get device count and memory info using rocm-smi
            result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse rocm-smi output (simplified)
                lines = result.stdout.strip().split('\n')
                device_count = len([line for line in lines if 'GPU' in line and 'MB' in line])
                
                # Estimate memory (this is a simplified approach)
                memory_per_device = 8.0  # Default assumption
                total_memory = memory_per_device * device_count
            else:
                device_count = 1
                memory_per_device = 8.0
                total_memory = 8.0
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            device_count = 1
            memory_per_device = 8.0
            total_memory = 8.0
        
        return DeviceInfo(
            device_type='rocm',
            device_count=device_count,
            memory_per_device=memory_per_device,
            total_memory=total_memory,
            compute_capability="gfx",  # ROCm uses gfx notation
            driver_version="rocm"
        )
    
    def _detect_mps_info(self) -> DeviceInfo:
        """Detect Apple Silicon MPS information."""
        # Apple Silicon unified memory
        total_memory = psutil.virtual_memory().total / (1024**3)
        
        # MPS uses unified memory, so memory per device is total memory
        return DeviceInfo(
            device_type='mps',
            device_count=1,
            memory_per_device=total_memory,
            total_memory=total_memory,
            compute_capability="apple_silicon",
            driver_version="mps"
        )
    
    def _detect_cpu_info(self) -> DeviceInfo:
        """Detect CPU information."""
        cpu_count = os.cpu_count() or 1
        total_memory = psutil.virtual_memory().total / (1024**3)
        
        return DeviceInfo(
            device_type='cpu',
            device_count=cpu_count,
            memory_per_device=total_memory / cpu_count,
            total_memory=total_memory,
            compute_capability=platform.processor(),
            driver_version="cpu"
        )


def validate_config(config: DictConfig) -> None:
    """
    Validate experiment configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # 验证必需的顶级字段
    required_fields = [
        "experiment", "model", "task", "peft", "training", 
        "evaluation", "system", "logging", "output"
    ]
    
    for field in required_fields:
        if field not in config:
            raise ConfigValidationError(f"Missing required field: {field}")
    
    # 验证实验配置
    _validate_experiment_config(config.experiment)
    
    # 验证模型配置
    _validate_model_config(config.model)
    
    # 验证任务配置
    _validate_task_config(config.task)
    
    # 验证PEFT配置
    _validate_peft_config(config.peft)
    
    # 验证训练配置
    _validate_training_config(config.training)
    
    # 验证评估配置
    _validate_evaluation_config(config.evaluation)
    
    # 验证系统配置
    _validate_system_config(config.system)
    
    # 验证日志配置
    _validate_logging_config(config.logging)
    
    # 验证输出配置
    _validate_output_config(config.output)


def _validate_experiment_config(exp_config: DictConfig) -> None:
    """Validate experiment configuration."""
    required_fields = ["name", "phase", "description"]
    for field in required_fields:
        if field not in exp_config:
            raise ConfigValidationError(f"Missing experiment.{field}")
    
    # 验证阶段
    valid_phases = ["P-0", "P-1", "P-2"]
    if exp_config.phase not in valid_phases:
        raise ConfigValidationError(
            f"Invalid experiment phase: {exp_config.phase}. "
            f"Must be one of {valid_phases}"
        )


def _validate_model_config(model_config: DictConfig) -> None:
    """Validate model configuration."""
    required_fields = ["name", "path_or_name"]
    for field in required_fields:
        if field not in model_config:
            raise ConfigValidationError(f"Missing model.{field}")
    
    # 验证支持的模型
    supported_models = [
        "llama2-7b", "llama2-13b", "llama2-70b",
        "llama3-8b", "llama3-70b",
        "mistral-7b", "mixtral-8x7b",
        "qwen2-7b", "qwen2-72b"
    ]
    
    if model_config.name not in supported_models:
        warnings.warn(
            f"Model {model_config.name} not in tested models: {supported_models}"
        )


def _validate_task_config(task_config: DictConfig) -> None:
    """Validate task configuration."""
    required_fields = ["name", "type"]
    for field in required_fields:
        if field not in task_config:
            raise ConfigValidationError(f"Missing task.{field}")
    
    # 验证任务类型
    valid_task_types = ["classification", "generation", "qa", "reasoning"]
    if task_config.type not in valid_task_types:
        raise ConfigValidationError(
            f"Invalid task type: {task_config.type}. "
            f"Must be one of {valid_task_types}"
        )
    
    # 验证数据集配置
    if "dataset" in task_config:
        if "name" not in task_config.dataset:
            raise ConfigValidationError("Missing task.dataset.name")


def _validate_peft_config(peft_config: DictConfig) -> None:
    """Validate PEFT configuration."""
    if "method" not in peft_config:
        raise ConfigValidationError("Missing peft.method")
    
    method = peft_config.method
    valid_methods = ["lora", "adalora", "dora", "strato_peft"]
    
    if method not in valid_methods:
        raise ConfigValidationError(
            f"Invalid PEFT method: {method}. "
            f"Must be one of {valid_methods}"
        )
    
    # 验证方法特定配置
    if method == "lora" and "lora" in peft_config:
        _validate_lora_config(peft_config.lora)
    elif method == "adalora" and "adalora" in peft_config:
        _validate_adalora_config(peft_config.adalora)
    elif method == "dora" and "dora" in peft_config:
        _validate_dora_config(peft_config.dora)
    elif method == "strato_peft" and "strato_peft" in peft_config:
        _validate_strato_peft_config(peft_config.strato_peft)


def _validate_lora_config(lora_config: DictConfig) -> None:
    """Validate LoRA configuration."""
    required_fields = ["r", "alpha", "dropout"]
    for field in required_fields:
        if field not in lora_config:
            raise ConfigValidationError(f"Missing peft.lora.{field}")
    
    # 验证数值范围
    if lora_config.r <= 0:
        raise ConfigValidationError("LoRA rank must be positive")
    
    if lora_config.alpha <= 0:
        raise ConfigValidationError("LoRA alpha must be positive")
    
    if not (0 <= lora_config.dropout <= 1):
        raise ConfigValidationError("LoRA dropout must be between 0 and 1")


def _validate_adalora_config(adalora_config: DictConfig) -> None:
    """Validate AdaLoRA configuration."""
    required_fields = ["init_r", "target_r", "beta1", "beta2"]
    for field in required_fields:
        if field not in adalora_config:
            raise ConfigValidationError(f"Missing peft.adalora.{field}")
    
    if adalora_config.init_r <= adalora_config.target_r:
        raise ConfigValidationError(
            "AdaLoRA init_r must be greater than target_r"
        )


def _validate_dora_config(dora_config: DictConfig) -> None:
    """Validate DoRA configuration."""
    required_fields = ["r", "alpha", "dropout"]
    for field in required_fields:
        if field not in dora_config:
            raise ConfigValidationError(f"Missing peft.dora.{field}")


def _validate_strato_peft_config(strato_config: DictConfig) -> None:
    """Validate STRATO-PEFT configuration."""
    required_fields = ["lambda_cost", "scheduler", "agent", "inner_loop"]
    for field in required_fields:
        if field not in strato_config:
            raise ConfigValidationError(f"Missing peft.strato_peft.{field}")
    
    # 验证成本权重
    if strato_config.lambda_cost < 0:
        raise ConfigValidationError("STRATO-PEFT lambda_cost must be non-negative")
    
    # 验证调度器配置
    scheduler = strato_config.scheduler
    required_scheduler_fields = ["rank_max", "rank_min", "rank_init"]
    for field in required_scheduler_fields:
        if field not in scheduler:
            raise ConfigValidationError(f"Missing peft.strato_peft.scheduler.{field}")
    
    if not (scheduler.rank_min <= scheduler.rank_init <= scheduler.rank_max):
        raise ConfigValidationError(
            "STRATO-PEFT rank constraints: rank_min <= rank_init <= rank_max"
        )
    
    # 验证智能体配置
    agent = strato_config.agent
    if "ppo" in agent:
        ppo = agent.ppo
        required_ppo_fields = ["lr", "eps", "value_loss_coef", "entropy_coef"]
        for field in required_ppo_fields:
            if field not in ppo:
                raise ConfigValidationError(f"Missing peft.strato_peft.agent.ppo.{field}")


def _validate_training_config(training_config: DictConfig) -> None:
    """Validate training configuration."""
    required_fields = [
        "num_epochs", "per_device_train_batch_size", 
        "gradient_accumulation_steps", "optimizer"
    ]
    for field in required_fields:
        if field not in training_config:
            raise ConfigValidationError(f"Missing training.{field}")
    
    # 验证优化器配置
    optimizer = training_config.optimizer
    if "name" not in optimizer:
        raise ConfigValidationError("Missing training.optimizer.name")
    
    if "lr" not in optimizer:
        raise ConfigValidationError("Missing training.optimizer.lr")
    
    if optimizer.lr <= 0:
        raise ConfigValidationError("Learning rate must be positive")


def _validate_evaluation_config(eval_config: DictConfig) -> None:
    """Validate evaluation configuration."""
    if "metrics" not in eval_config:
        raise ConfigValidationError("Missing evaluation.metrics")
    
    if not eval_config.metrics:
        raise ConfigValidationError("At least one evaluation metric must be specified")


def _validate_system_config(system_config: DictConfig) -> None:
    """Validate system configuration."""
    required_fields = ["seed", "gpu"]
    for field in required_fields:
        if field not in system_config:
            raise ConfigValidationError(f"Missing system.{field}")
    
    # 验证GPU配置
    gpu_config = system_config.gpu
    if "enabled" not in gpu_config:
        raise ConfigValidationError("Missing system.gpu.enabled")
    
    if gpu_config.enabled and not torch.cuda.is_available():
        warnings.warn("GPU enabled but CUDA not available")


def _validate_logging_config(logging_config: DictConfig) -> None:
    """Validate logging configuration."""
    required_fields = ["local", "wandb"]
    for field in required_fields:
        if field not in logging_config:
            raise ConfigValidationError(f"Missing logging.{field}")
    
    # 验证本地日志配置
    local = logging_config.local
    if "log_level" not in local:
        raise ConfigValidationError("Missing logging.local.log_level")
    
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if local.log_level not in valid_log_levels:
        raise ConfigValidationError(
            f"Invalid log level: {local.log_level}. "
            f"Must be one of {valid_log_levels}"
        )


def _validate_output_config(output_config: DictConfig) -> None:
    """Validate output configuration."""
    if "base_dir" not in output_config:
        raise ConfigValidationError("Missing output.base_dir")


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        DictConfig: Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def resolve_config_paths(config: DictConfig, config_dir: Optional[str] = None) -> DictConfig:
    """
    Resolve relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration with potentially relative paths
        config_dir: Directory containing the config file
        
    Returns:
        DictConfig: Configuration with resolved paths
    """
    if config_dir is None:
        config_dir = os.getcwd()
    
    config_dir = Path(config_dir)
    
    # 解析模型路径
    if "model" in config and "path_or_name" in config.model:
        model_path = config.model.path_or_name
        if not model_path.startswith("/") and not model_path.startswith("~"):
            # 相对路径，解析为绝对路径
            if os.path.exists(config_dir / model_path):
                config.model.path_or_name = str(config_dir / model_path)
    
    # 解析数据集路径
    if "task" in config and "dataset" in config.task:
        dataset_config = config.task.dataset
        for path_field in ["data_dir", "train_file", "validation_file", "test_file"]:
            if path_field in dataset_config:
                path_value = dataset_config[path_field]
                if isinstance(path_value, str) and not path_value.startswith("/"):
                    resolved_path = config_dir / path_value
                    if resolved_path.exists():
                        dataset_config[path_field] = str(resolved_path)
    
    # 解析输出路径
    if "output" in config and "base_dir" in config.output:
        output_dir = config.output.base_dir
        if not output_dir.startswith("/") and not output_dir.startswith("~"):
            config.output.base_dir = str(config_dir / output_dir)
    
    return config


def get_config_summary(config: DictConfig) -> Dict[str, Any]:
    """
    Get a summary of key configuration parameters.
    
    Args:
        config: Configuration to summarize
        
    Returns:
        Dict[str, Any]: Configuration summary
    """
    summary = {
        "experiment_name": config.experiment.name,
        "phase": config.experiment.phase,
        "model": config.model.name,
        "task": config.task.name,
        "peft_method": config.peft.method,
        "seed": config.system.seed,
        "epochs": config.training.num_epochs,
        "learning_rate": config.training.optimizer.lr,
        "batch_size": config.training.per_device_train_batch_size,
    }
    
    # 添加PEFT特定参数
    if config.peft.method == "lora" and "lora" in config.peft:
        summary["lora_rank"] = config.peft.lora.r
        summary["lora_alpha"] = config.peft.lora.alpha
    elif config.peft.method == "strato_peft" and "strato_peft" in config.peft:
        summary["lambda_cost"] = config.peft.strato_peft.lambda_cost
        summary["max_rank"] = config.peft.strato_peft.scheduler.rank_max
    
    return summary


def save_config_with_metadata(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration with additional metadata.
    
    Args:
        config: Configuration to save
        save_path: Path to save the configuration
    """
    import time
    import getpass
    import socket
    
    # 添加元数据
    metadata = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "saved_by": getpass.getuser(),
        "hostname": socket.gethostname(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": torch.__version__,
    }
    
    # 创建带元数据的配置
    config_with_metadata = OmegaConf.create({
        "metadata": metadata,
        "config": config
    })
    
    # 保存配置
    OmegaConf.save(config_with_metadata, save_path)


class ConfigManager:
    """
    Manages configuration loading, validation, and environment adaptation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.platform_detector = PlatformDetector(logger)
        
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Loaded configuration as DictConfig
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.logger.info(f"Loading configuration from {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Convert to OmegaConf
        config = OmegaConf.create(config_dict)
        
        # Validate and adapt for current environment
        config = self.adapt_config_for_environment(config)
        
        return config
    
    def adapt_config_for_environment(self, config: DictConfig) -> DictConfig:
        """
        Adapt configuration for current training environment.
        
        Args:
            config: Base configuration
            
        Returns:
            Environment-adapted configuration
        """
        device_info = self.platform_detector.detect_device_info()
        
        # Create training environment configuration
        training_env = self._create_training_environment(config, device_info)
        
        # Update config with environment-specific settings
        config = self._update_config_for_device(config, device_info, training_env)
        
        # Add device info to config
        config.device_info = OmegaConf.create(asdict(device_info))
        config.training_environment = OmegaConf.create(asdict(training_env))
        
        self.logger.info(f"Configuration adapted for {device_info.device_type} environment")
        
        return config
    
    def _create_training_environment(self, config: DictConfig, device_info: DeviceInfo) -> TrainingEnvironment:
        """
        Create training environment configuration based on device capabilities.
        
        Args:
            config: Base configuration
            device_info: Detected device information
            
        Returns:
            Training environment configuration
        """
        # Base settings from config
        base_batch_size = config.get('training', {}).get('per_device_train_batch_size', 8)
        base_grad_accum = config.get('training', {}).get('gradient_accumulation_steps', 1)
        
        # Adapt based on device type and memory
        if device_info.device_type == 'cuda':
            # CUDA optimizations
            batch_size, grad_accum = self._optimize_batch_size_cuda(base_batch_size, base_grad_accum, device_info)
            mixed_precision = config.get('training', {}).get('mixed_precision', True)
            compile_model = config.get('training', {}).get('compile_model', True)
            optimization_level = config.get('training', {}).get('optimization_level', 'O1')
            num_workers = min(8, os.cpu_count() or 1)
            
        elif device_info.device_type == 'rocm':
            # ROCm optimizations
            batch_size, grad_accum = self._optimize_batch_size_rocm(base_batch_size, base_grad_accum, device_info)
            mixed_precision = config.get('training', {}).get('mixed_precision', True)
            compile_model = config.get('training', {}).get('compile_model', False)  # May have issues with ROCm
            optimization_level = config.get('training', {}).get('optimization_level', 'O1')
            num_workers = min(4, os.cpu_count() or 1)
            
        elif device_info.device_type == 'mps':
            # Apple Silicon optimizations
            batch_size, grad_accum = self._optimize_batch_size_mps(base_batch_size, base_grad_accum, device_info)
            mixed_precision = config.get('training', {}).get('mixed_precision', False)  # MPS has limited FP16 support
            compile_model = config.get('training', {}).get('compile_model', False)  # torch.compile may not work well with MPS
            optimization_level = 'O0'
            num_workers = min(4, os.cpu_count() or 1)
            
        else:  # CPU
            # CPU optimizations
            batch_size, grad_accum = self._optimize_batch_size_cpu(base_batch_size, base_grad_accum, device_info)
            mixed_precision = False  # No mixed precision on CPU
            compile_model = config.get('training', {}).get('compile_model', True)
            optimization_level = 'O0'
            num_workers = min(os.cpu_count() or 1, 8)
        
        # Docker-specific adjustments
        if device_info.is_docker:
            num_workers = min(num_workers, 2)  # Reduce workers in Docker
            persistent_workers = False
        else:
            persistent_workers = num_workers > 0
        
        return TrainingEnvironment(
            device_info=device_info,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            mixed_precision=mixed_precision,
            dataloader_num_workers=num_workers,
            pin_memory=device_info.device_type in ['cuda', 'rocm'],
            persistent_workers=persistent_workers,
            compile_model=compile_model,
            optimization_level=optimization_level
        )
    
    def _optimize_batch_size_cuda(self, base_batch_size: int, base_grad_accum: int, device_info: DeviceInfo) -> Tuple[int, int]:
        """Optimize batch size for CUDA devices."""
        # Estimate memory usage per sample (rough heuristic)
        memory_per_sample_gb = 0.5  # Assume 0.5GB per sample for 7B model
        
        # Calculate maximum batch size per device
        max_batch_per_device = max(1, int(device_info.memory_per_device * 0.8 / memory_per_sample_gb))
        
        # Adjust batch size
        if base_batch_size > max_batch_per_device:
            # Increase gradient accumulation to maintain effective batch size
            effective_batch_size = base_batch_size * base_grad_accum
            new_batch_size = max_batch_per_device
            new_grad_accum = max(1, effective_batch_size // new_batch_size)
        else:
            new_batch_size = base_batch_size
            new_grad_accum = base_grad_accum
        
        return new_batch_size, new_grad_accum
    
    def _optimize_batch_size_rocm(self, base_batch_size: int, base_grad_accum: int, device_info: DeviceInfo) -> Tuple[int, int]:
        """Optimize batch size for ROCm devices."""
        # Similar to CUDA but more conservative
        memory_per_sample_gb = 0.6  # Slightly more conservative for ROCm
        max_batch_per_device = max(1, int(device_info.memory_per_device * 0.7 / memory_per_sample_gb))
        
        if base_batch_size > max_batch_per_device:
            effective_batch_size = base_batch_size * base_grad_accum
            new_batch_size = max_batch_per_device
            new_grad_accum = max(1, effective_batch_size // new_batch_size)
        else:
            new_batch_size = base_batch_size
            new_grad_accum = base_grad_accum
        
        return new_batch_size, new_grad_accum
    
    def _optimize_batch_size_mps(self, base_batch_size: int, base_grad_accum: int, device_info: DeviceInfo) -> Tuple[int, int]:
        """Optimize batch size for Apple Silicon MPS."""
        # MPS uses unified memory, be more conservative
        memory_per_sample_gb = 0.8
        max_batch_size = max(1, int(device_info.total_memory * 0.5 / memory_per_sample_gb))
        
        if base_batch_size > max_batch_size:
            effective_batch_size = base_batch_size * base_grad_accum
            new_batch_size = max_batch_size
            new_grad_accum = max(1, effective_batch_size // new_batch_size)
        else:
            new_batch_size = base_batch_size
            new_grad_accum = base_grad_accum
        
        return new_batch_size, new_grad_accum
    
    def _optimize_batch_size_cpu(self, base_batch_size: int, base_grad_accum: int, device_info: DeviceInfo) -> Tuple[int, int]:
        """Optimize batch size for CPU training."""
        # CPU training is memory-bound, use smaller batches
        memory_per_sample_gb = 1.0
        max_batch_size = max(1, int(device_info.total_memory * 0.3 / memory_per_sample_gb))
        
        # CPU training benefits from smaller batches
        max_batch_size = min(max_batch_size, 4)
        
        if base_batch_size > max_batch_size:
            effective_batch_size = base_batch_size * base_grad_accum
            new_batch_size = max_batch_size
            new_grad_accum = max(1, effective_batch_size // new_batch_size)
        else:
            new_batch_size = base_batch_size
            new_grad_accum = base_grad_accum
        
        return new_batch_size, new_grad_accum
    
    def _update_config_for_device(self, config: DictConfig, device_info: DeviceInfo, training_env: TrainingEnvironment) -> DictConfig:
        """
        Update configuration with device-specific settings.
        
        Args:
            config: Base configuration
            device_info: Device information
            training_env: Training environment configuration
            
        Returns:
            Updated configuration
        """
        # Ensure training section exists
        if 'training' not in config:
            config.training = {}
        
        # Update training settings
        config.training.per_device_train_batch_size = training_env.batch_size
        config.training.gradient_accumulation_steps = training_env.gradient_accumulation_steps
        config.training.mixed_precision = training_env.mixed_precision
        config.training.compile_model = training_env.compile_model
        
        # Update dataloader settings
        if 'dataloader' not in config.training:
            config.training.dataloader = {}
        config.training.dataloader.num_workers = training_env.dataloader_num_workers
        config.training.dataloader.pin_memory = training_env.pin_memory
        config.training.dataloader.persistent_workers = training_env.persistent_workers
        
        # Ensure system section exists
        if 'system' not in config:
            config.system = {}
        
        # Device-specific optimizations
        if device_info.device_type == 'cuda':
            config.system.device = 'cuda'
            if device_info.device_count > 1:
                config.system.distributed = True
                config.system.world_size = device_info.device_count
        elif device_info.device_type == 'rocm':
            config.system.device = 'cuda'  # ROCm uses CUDA API
            os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Set for ROCm
        elif device_info.device_type == 'mps':
            config.system.device = 'mps'
        else:
            config.system.device = 'cpu'
        
        # Docker-specific settings
        if device_info.is_docker:
            # Reduce resource usage in Docker
            config.training.dataloader.num_workers = min(config.training.dataloader.num_workers, 2)
            if hasattr(config, 'logging') and hasattr(config.logging, 'wandb') and config.logging.wandb.get('enabled', False):
                config.logging.wandb.enabled = False  # Disable wandb in Docker by default
        
        return config


def setup_logging(config: DictConfig) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration containing logging settings
        
    Returns:
        Configured logger
    """
    log_level = config.get('logging', {}).get('local', {}).get('log_level', 'INFO').upper()
    log_format = config.get('logging', {}).get('local', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger('strato_peft')
    
    # Add file handler if specified
    if config.get('logging', {}).get('local', {}).get('log_file'):
        log_file = Path(config.logging.local.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def get_device_string(config: DictConfig) -> str:
    """
    Get device string for PyTorch.
    
    Args:
        config: Configuration containing device information
        
    Returns:
        Device string (e.g., 'cuda:0', 'mps', 'cpu')
    """
    device_info = config.get('device_info')
    if not device_info:
        return 'cpu'
    
    device_type = device_info.device_type
    
    if device_type == 'cuda':
        return 'cuda:0'  # Default to first GPU
    elif device_type == 'rocm':
        return 'cuda:0'  # ROCm uses CUDA API
    elif device_type == 'mps':
        return 'mps'
    else:
        return 'cpu'


def estimate_memory_usage(config: DictConfig) -> Dict[str, float]:
    """
    Estimate memory usage for training configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with memory usage estimates in GB
    """
    # Rough estimates based on model size and batch size
    model_name = config.get('model', {}).get('name', '')
    batch_size = config.get('training', {}).get('per_device_train_batch_size', 1)
    
    # Model size estimates (in GB)
    model_sizes = {
        'llama2-7b': 14.0,
        'llama2-13b': 26.0,
        'llama-7b': 14.0,
        'llama-13b': 26.0,
    }
    
    # Find matching model size
    base_model_size = 14.0  # Default
    for model_key, size in model_sizes.items():
        if model_key in model_name.lower():
            base_model_size = size
            break
    
    # Estimate components
    model_memory = base_model_size
    optimizer_memory = base_model_size * 2  # AdamW uses 2x model size
    activation_memory = batch_size * 0.5  # Rough estimate
    gradient_memory = base_model_size * 0.1  # PEFT gradients are small
    
    total_memory = model_memory + optimizer_memory + activation_memory + gradient_memory
    
    return {
        'model': model_memory,
        'optimizer': optimizer_memory,
        'activations': activation_memory,
        'gradients': gradient_memory,
        'total': total_memory
    }