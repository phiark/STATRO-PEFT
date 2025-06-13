#!/usr/bin/env python3
"""
Reproducibility utilities for STRATO-PEFT experimental framework.

This module provides utilities for ensuring reproducible experiments
through proper seed setting, deterministic operations, and verification.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import os
import random
import warnings
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if deterministic:
        # Make CuDNN deterministic
        cudnn.deterministic = True
        cudnn.benchmark = False
        
        # Set environment variables for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Enable deterministic algorithms in PyTorch
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            warnings.warn(f"Could not enable deterministic algorithms: {e}")
        
        # Set number of threads for reproducibility
        torch.set_num_threads(1)
    else:
        # Allow non-deterministic but faster operations
        cudnn.deterministic = False
        cudnn.benchmark = True


def get_random_states() -> Dict[str, Any]:
    """
    Get current random states for all random number generators.
    
    Returns:
        Dict[str, Any]: Dictionary containing all random states
    """
    states = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    # Add CUDA random states if available
    if torch.cuda.is_available():
        states['torch_cuda_random'] = torch.cuda.get_rng_state()
        
        # Get states for all CUDA devices
        for i in range(torch.cuda.device_count()):
            states[f'torch_cuda_random_device_{i}'] = torch.cuda.get_rng_state(device=i)
    
    return states


def set_random_states(states: Dict[str, Any]) -> None:
    """
    Set random states for all random number generators.
    
    Args:
        states: Dictionary containing random states
    """
    if 'python_random' in states:
        random.setstate(states['python_random'])
    
    if 'numpy_random' in states:
        np.random.set_state(states['numpy_random'])
    
    if 'torch_random' in states:
        torch.set_rng_state(states['torch_random'])
    
    if torch.cuda.is_available():
        if 'torch_cuda_random' in states:
            torch.cuda.set_rng_state(states['torch_cuda_random'])
        
        # Set states for all CUDA devices
        for i in range(torch.cuda.device_count()):
            device_key = f'torch_cuda_random_device_{i}'
            if device_key in states:
                torch.cuda.set_rng_state(states[device_key], device=i)


def save_random_states(filepath: str) -> None:
    """
    Save current random states to file.
    
    Args:
        filepath: Path to save the random states
    """
    states = get_random_states()
    torch.save(states, filepath)


def load_random_states(filepath: str) -> None:
    """
    Load random states from file.
    
    Args:
        filepath: Path to load the random states from
    """
    states = torch.load(filepath)
    set_random_states(states)


def verify_reproducibility(seed: int, num_iterations: int = 3) -> bool:
    """
    Verify that the random number generators produce consistent results.
    
    Args:
        seed: Seed to test with
        num_iterations: Number of iterations to test
        
    Returns:
        bool: True if reproducible, False otherwise
    """
    results = []
    
    for i in range(num_iterations):
        # Reset seed
        set_seed(seed, deterministic=True)
        
        # Generate some random numbers
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        
        # Simple tensor operation
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        z = torch.mm(x, y).sum().item()
        
        result = {
            'python': python_rand,
            'numpy': numpy_rand,
            'torch': torch_rand,
            'operation': z
        }
        
        results.append(result)
    
    # Check if all results are identical
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        for key in first_result:
            if abs(first_result[key] - result[key]) > 1e-10:
                warnings.warn(
                    f"Reproducibility check failed at iteration {i} for {key}: "
                    f"{first_result[key]} != {result[key]}"
                )
                return False
    
    return True


class ReproducibilityContext:
    """
    Context manager for reproducible code blocks.
    
    Usage:
        with ReproducibilityContext(seed=42):
            # Your reproducible code here
            result = some_random_operation()
    """
    
    def __init__(self, seed: int, deterministic: bool = True, restore_state: bool = True):
        """
        Initialize reproducibility context.
        
        Args:
            seed: Random seed to use
            deterministic: Whether to enable deterministic operations
            restore_state: Whether to restore original random states on exit
        """
        self.seed = seed
        self.deterministic = deterministic
        self.restore_state = restore_state
        self.original_states = None
        self.original_deterministic = None
        self.original_benchmark = None
    
    def __enter__(self):
        # Save original states
        if self.restore_state:
            self.original_states = get_random_states()
            self.original_deterministic = cudnn.deterministic
            self.original_benchmark = cudnn.benchmark
        
        # Set new seed and deterministic behavior
        set_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        if self.restore_state and self.original_states is not None:
            set_random_states(self.original_states)
            cudnn.deterministic = self.original_deterministic
            cudnn.benchmark = self.original_benchmark


def create_reproducibility_report(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive reproducibility report.
    
    Args:
        seed: Random seed used
        config: Experiment configuration
        
    Returns:
        Dict[str, Any]: Reproducibility report
    """
    import torch
    import sys
    import platform
    
    report = {
        'seed': seed,
        'deterministic_enabled': cudnn.deterministic,
        'benchmark_enabled': cudnn.benchmark,
        'python_version': sys.version,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'random_states_snapshot': get_random_states(),
        'reproducibility_verified': verify_reproducibility(seed),
    }
    
    # Add relevant environment variables
    env_vars = [
        'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS',
        'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES'
    ]
    
    report['environment_variables'] = {
        var: os.environ.get(var) for var in env_vars
    }
    
    # Add configuration-specific reproducibility info
    if 'system' in config:
        system_config = config['system']
        report['config_seed'] = system_config.get('seed')
        report['config_deterministic'] = system_config.get('deterministic')
    
    return report


def check_deterministic_operations() -> Dict[str, bool]:
    """
    Check which operations are deterministic.
    
    Returns:
        Dict[str, bool]: Status of deterministic operations
    """
    checks = {}
    
    # Check CuDNN settings
    checks['cudnn_deterministic'] = cudnn.deterministic
    checks['cudnn_benchmark'] = not cudnn.benchmark  # benchmark=False is better for determinism
    
    # Check PyTorch deterministic algorithms
    try:
        checks['torch_deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        checks['torch_deterministic_algorithms'] = None
    
    # Check environment variables
    checks['pythonhashseed_set'] = 'PYTHONHASHSEED' in os.environ
    checks['cublas_workspace_config_set'] = 'CUBLAS_WORKSPACE_CONFIG' in os.environ
    
    # Check thread settings
    checks['torch_num_threads'] = torch.get_num_threads()
    
    return checks


def suggest_reproducibility_improvements() -> List[str]:
    """
    Suggest improvements for better reproducibility.
    
    Returns:
        List[str]: List of suggestions
    """
    suggestions = []
    checks = check_deterministic_operations()
    
    if not checks.get('cudnn_deterministic', False):
        suggestions.append("Enable CuDNN deterministic mode: torch.backends.cudnn.deterministic = True")
    
    if checks.get('cudnn_benchmark', False):
        suggestions.append("Disable CuDNN benchmark: torch.backends.cudnn.benchmark = False")
    
    if not checks.get('pythonhashseed_set', False):
        suggestions.append("Set PYTHONHASHSEED environment variable")
    
    if not checks.get('cublas_workspace_config_set', False):
        suggestions.append("Set CUBLAS_WORKSPACE_CONFIG environment variable")
    
    if checks.get('torch_deterministic_algorithms') is False:
        suggestions.append("Enable PyTorch deterministic algorithms: torch.use_deterministic_algorithms(True)")
    
    if checks.get('torch_num_threads', 0) > 1:
        suggestions.append("Consider setting torch.set_num_threads(1) for maximum reproducibility")
    
    return suggestions


class SeedManager:
    """
    Advanced seed management for complex experiments.
    """
    
    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self.seed_counter = 0
        self.seed_history = []
    
    def get_next_seed(self, component: str = "default") -> int:
        """
        Get the next seed for a component.
        
        Args:
            component: Name of the component requesting a seed
            
        Returns:
            int: Next seed value
        """
        seed = self.base_seed + self.seed_counter
        self.seed_counter += 1
        
        self.seed_history.append({
            'component': component,
            'seed': seed,
            'counter': self.seed_counter - 1
        })
        
        return seed
    
    def get_seed_for_component(self, component: str) -> int:
        """
        Get a deterministic seed for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            int: Deterministic seed for the component
        """
        # Use hash of component name for deterministic but different seeds
        component_hash = hash(component) % 10000
        return self.base_seed + component_hash
    
    def reset_counter(self) -> None:
        """Reset the seed counter."""
        self.seed_counter = 0
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the seed usage history."""
        return self.seed_history.copy()
    
    def save_state(self, filepath: str) -> None:
        """Save the seed manager state."""
        state = {
            'base_seed': self.base_seed,
            'seed_counter': self.seed_counter,
            'seed_history': self.seed_history
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath: str) -> None:
        """Load the seed manager state."""
        state = torch.load(filepath)
        self.base_seed = state['base_seed']
        self.seed_counter = state['seed_counter']
        self.seed_history = state['seed_history']


def ensure_reproducible_dataloader(dataloader, seed: int):
    """
    Ensure a DataLoader is reproducible by setting worker seeds.
    
    Args:
        dataloader: PyTorch DataLoader
        seed: Base seed for workers
        
    Returns:
        DataLoader with reproducible worker initialization
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Update the dataloader's worker_init_fn
    dataloader.worker_init_fn = worker_init_fn
    
    return dataloader


def create_reproducible_split(dataset_size: int, train_ratio: float, val_ratio: float, seed: int):
    """
    Create reproducible train/validation/test splits.
    
    Args:
        dataset_size: Total size of the dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed for splitting
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    with ReproducibilityContext(seed):
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return train_indices, val_indices, test_indices