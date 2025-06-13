#!/usr/bin/env python3
"""
STRATO-PEFT Experimental Framework - Main Entry Point

This script serves as the main entry point for running STRATO-PEFT experiments.
It handles configuration loading, experiment setup, and execution coordination.

Usage:
    python main.py --config configs/llama2_7b_mmlu_lora.yaml --seed 42
    python main.py --config configs/llama2_7b_mmlu_strato.yaml --seed 42

Author: STRATO-PEFT Research Team
Date: 2024
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import wandb
import yaml
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.utils.logging_utils import setup_logging, log_system_info
    from src.utils.config_utils import (
        validate_config, merge_configs, ConfigManager, 
        setup_logging as config_setup_logging, get_device_string
    )
    from src.utils.reproducibility_utils import set_seed, verify_reproducibility
    from src.trainer import ExperimentTrainer
    from src.models.model_factory import ModelFactory
    from src.tasks.task_factory import TaskFactory
    from src.peft_methods.peft_factory import PEFTFactory
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all source files are present and requirements are installed")
    sys.exit(1)

# 初始化Rich控制台
console = Console()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="STRATO-PEFT Experimental Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration file"
    )
    
    # 可选参数
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast development test"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2')"
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        choices=["auto", "cuda", "rocm", "mps", "cpu"],
        default="auto",
        help="Force specific platform (default: auto-detect)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and setup without running training"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="STRATO-PEFT v1.0.0"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> DictConfig:
    """
    Load and validate experiment configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # 使用OmegaConf加载配置，支持继承
    config = OmegaConf.load(config_path)
    
    # 验证配置
    validate_config(config)
    
    return config


def override_config_with_args(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """
    Override configuration with command line arguments.
    
    Args:
        config: Base configuration
        args: Command line arguments
        
    Returns:
        DictConfig: Updated configuration
    """
    # 覆盖种子
    if args.seed is not None:
        config.system.seed = args.seed
    
    # 覆盖输出目录
    if args.output_dir is not None:
        config.output.base_dir = args.output_dir
    
    # 覆盖检查点恢复
    if args.resume is not None:
        config.checkpoint.resume_from = args.resume
    
    # 调试模式
    if args.debug:
        config.logging.local.log_level = "DEBUG"
        if "debug" not in config:
            config.debug = {}
        config.debug.debug_mode = True
    
    # 快速开发运行
    if args.fast_dev_run:
        config.training.num_epochs = 1
        config.training.save_steps = 10
        config.training.eval_steps = 5
        if hasattr(config.task, 'max_samples'):
            config.task.max_samples = 100
        if "debug" not in config:
            config.debug = {}
        config.debug.fast_dev_run = True
    
    # 禁用WandB
    if args.no_wandb:
        config.logging.wandb.enabled = False
    
    # GPU配置
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        config.system.gpu.device_ids = gpu_ids
    
    # 平台配置
    if args.platform != "auto":
        if "device_info" not in config:
            config.device_info = {}
        config.device_info.device_type = args.platform
    
    # 性能分析
    if args.profile:
        if "debug" not in config:
            config.debug = {}
        config.debug.enable_profiling = True
    
    return config


def setup_experiment_environment(config: DictConfig) -> Dict[str, Any]:
    """
    Setup the experiment environment including logging, reproducibility, and directories.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dict[str, Any]: Environment setup information
    """
    # 使用 ConfigManager 进行平台适配
    config_manager = ConfigManager()
    config = config_manager.adapt_config_for_environment(config)
    
    # 设置随机种子
    set_seed(config.system.seed, config.system.deterministic)
    
    # 创建输出目录
    output_dir = Path(config.output.base_dir)
    if config.output.experiment_dir:
        output_dir = output_dir / config.output.experiment_dir
    else:
        # 自动生成实验目录名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config.experiment.name}_{timestamp}"
        output_dir = output_dir / exp_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(
        log_dir=str(log_dir),
        log_level=config.logging.local.log_level,
        experiment_name=config.experiment.name
    )
    
    # 保存配置文件
    config_save_path = output_dir / "config.yaml"
    OmegaConf.save(config, config_save_path)
    
    # 初始化WandB
    wandb_run = None
    if config.logging.wandb.enabled:
        wandb_config = OmegaConf.to_container(config, resolve=True)
        
        wandb_run = wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            name=config.logging.wandb.name or config.experiment.name,
            tags=config.logging.wandb.tags + config.experiment.tags,
            notes=config.logging.wandb.notes or config.experiment.description,
            config=wandb_config,
            dir=str(output_dir)
        )
    
    # 记录系统信息
    log_system_info(logger)
    
    # 验证可重现性设置
    if config.validation.verify_deterministic:
        verify_reproducibility(config.system.seed)
    
    return {
        "output_dir": output_dir,
        "logger": logger,
        "wandb_run": wandb_run
    }


def display_experiment_info(config: DictConfig, env_info: Dict[str, Any]) -> None:
    """
    Display experiment information in a formatted table.
    
    Args:
        config: Experiment configuration
        env_info: Environment setup information
    """
    # 创建实验信息表格
    table = Table(title="🚀 STRATO-PEFT Experiment Configuration")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    # 基本信息
    table.add_row("Experiment Name", config.experiment.name)
    table.add_row("Phase", config.experiment.phase)
    table.add_row("Model", config.model.name)
    table.add_row("Task", config.task.name)
    table.add_row("PEFT Method", config.peft.method)
    table.add_row("Seed", str(config.system.seed))
    table.add_row("Output Directory", str(env_info["output_dir"]))
    
    # PEFT特定信息
    if config.peft.method == "lora":
        table.add_row("LoRA Rank", str(config.peft.lora.r))
        table.add_row("LoRA Alpha", str(config.peft.lora.alpha))
    elif config.peft.method == "strato_peft":
        table.add_row("Lambda (Cost Weight)", str(config.peft.strato_peft.lambda_cost))
        table.add_row("Max Rank", str(config.peft.strato_peft.scheduler.rank_max))
        table.add_row("Inner Loop Steps", str(config.peft.strato_peft.inner_loop.num_steps))
    
    # 训练信息
    table.add_row("Learning Rate", str(config.training.optimizer.lr))
    table.add_row("Epochs", str(config.training.num_epochs))
    table.add_row("Batch Size", str(config.training.per_device_train_batch_size))
    table.add_row("Gradient Accumulation", str(config.training.gradient_accumulation_steps))
    
    console.print(table)
    console.print()


def main() -> None:
    """
    Main execution function.
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 显示启动信息
        console.print(Panel.fit(
            "[bold blue]STRATO-PEFT Experimental Framework[/bold blue]\n"
            "Strategic Resource-Aware Tunable Optimization for Parameter-Efficient Fine-Tuning",
            border_style="blue"
        ))
        
        # 加载配置
        console.print("[yellow]Loading configuration...[/yellow]")
        config = load_config(args.config)
        
        # 使用命令行参数覆盖配置
        config = override_config_with_args(config, args)
        
        # 设置实验环境
        console.print("[yellow]Setting up experiment environment...[/yellow]")
        env_info = setup_experiment_environment(config)
        logger = env_info["logger"]
        
        # 显示实验信息
        display_experiment_info(config, env_info)
        
        # 显示平台信息
        device_info = config.get('device_info', {})
        device_str = get_device_string(device_info.get('device_type', 'cpu'))
        console.print(f"[blue]Platform: {device_str}[/blue]")
        
        logger.info(f"Starting experiment: {config.experiment.name}")
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Output directory: {env_info['output_dir']}")
        logger.info(f"Platform: {device_str}")
        
        # Dry run - 仅验证配置和设置
        if args.dry_run:
            console.print("[green]✅ Dry run completed successfully![/green]")
            console.print("Configuration and environment validation passed.")
            logger.info("Dry run completed successfully")
            return
        
        # 初始化组件工厂
        console.print("[yellow]Initializing components...[/yellow]")
        
        # 创建模型
        model_factory = ModelFactory(config.model)
        model, tokenizer = model_factory.create_model_and_tokenizer()
        
        # 创建任务
        task_factory = TaskFactory(config.task)
        task_handler = task_factory.create_task_handler(tokenizer)
        
        # 创建PEFT方法
        peft_factory = PEFTFactory(config.peft)
        peft_handler = peft_factory.create_peft_handler(model)
        
        # 创建训练器
        trainer = ExperimentTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            task_handler=task_handler,
            peft_handler=peft_handler,
            output_dir=env_info["output_dir"],
            logger=logger,
            wandb_run=env_info["wandb_run"]
        )
        
        # 开始训练
        console.print("[green]Starting training...[/green]")
        results = trainer.train()
        
        # 运行评估
        console.print("[green]Running final evaluation...[/green]")
        eval_results = trainer.evaluate()
        
        # 保存结果
        results.update(eval_results)
        trainer.save_results(results)
        
        # 显示结果摘要
        console.print("[green]✅ Experiment completed successfully![/green]")
        
        # 创建结果表格
        results_table = Table(title="📊 Experiment Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        for key, value in results.items():
            if isinstance(value, float):
                results_table.add_row(key, f"{value:.4f}")
            else:
                results_table.add_row(key, str(value))
        
        console.print(results_table)
        
        logger.info("Experiment completed successfully")
        logger.info(f"Results saved to: {env_info['output_dir']}")
        
    except KeyboardInterrupt:
        console.print("\n[red]❌ Experiment interrupted by user[/red]")
        if 'logger' in locals():
            logger.warning("Experiment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        console.print(f"\n[red]❌ Experiment failed with error: {str(e)}[/red]")
        if 'logger' in locals():
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise
        
    finally:
        # 清理资源
        if 'env_info' in locals() and env_info.get('wandb_run'):
            wandb.finish()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()