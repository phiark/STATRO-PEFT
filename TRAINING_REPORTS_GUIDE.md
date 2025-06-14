# 训练报告和结果文件位置指南

## 📊 训练报告存储位置

### 默认输出结构
```
./results/
├── {experiment_name}_{timestamp}/          # 实验目录
│   ├── config.yaml                         # 实验配置备份
│   ├── logs/                              # 训练日志
│   │   ├── training.log                   # 详细训练日志
│   │   ├── evaluation.log                 # 评估日志
│   │   └── system.log                     # 系统资源日志
│   ├── checkpoints/                       # 模型检查点
│   │   ├── checkpoint-epoch-1/
│   │   ├── checkpoint-epoch-2/
│   │   └── best-model/
│   ├── metrics/                           # 训练指标
│   │   ├── training_metrics.json         # 训练过程指标
│   │   ├── evaluation_results.json       # 最终评估结果
│   │   └── perplexity_history.json       # 困惑度变化历史
│   ├── plots/                             # 可视化图表
│   │   ├── loss_curve.png
│   │   ├── perplexity_curve.png
│   │   └── learning_rate_schedule.png
│   └── final_report.html                  # 完整训练报告
```

### 关键指标文件

#### 1. 困惑度和损失 (`metrics/training_metrics.json`)
```json
{
  "epoch_1": {
    "train_loss": 2.456,
    "train_perplexity": 11.65,
    "eval_loss": 2.234,
    "eval_perplexity": 9.34,
    "learning_rate": 0.0002
  },
  "epoch_2": {
    "train_loss": 2.123,
    "train_perplexity": 8.35,
    "eval_loss": 2.045,
    "eval_perplexity": 7.73,
    "learning_rate": 0.00018
  }
}
```

#### 2. 最终评估结果 (`metrics/evaluation_results.json`)
```json
{
  "final_perplexity": 7.23,
  "final_loss": 1.978,
  "accuracy": 0.856,
  "parameter_efficiency": 0.9997,
  "training_time": 1845.6,
  "inference_speed": 125.3,
  "memory_usage_mb": 4832.1
}
```

#### 3. PEFT特定指标
```json
{
  "lora_metrics": {
    "trainable_params": 18432,
    "total_params": 124458240,
    "trainable_ratio": 0.000148,
    "rank_distribution": {
      "transformer.h.0.attn.c_attn": 8,
      "transformer.h.0.attn.c_proj": 8
    }
  },
  "strato_metrics": {
    "exploration_episodes": 15,
    "best_configuration": {
      "transformer.h.0.attn.c_attn": 12,
      "transformer.h.1.attn.c_attn": 8
    },
    "reward_history": [0.23, 0.45, 0.67, 0.82],
    "cost_efficiency": 0.956
  }
}
```

## 🔍 如何查看和分析报告

### 快速查看最新结果
```bash
# 查看最近的实验结果
ls -la ./results/ | head -10

# 查看特定实验的困惑度
cat ./results/gpt2_lora_experiment_20240614/metrics/evaluation_results.json | jq '.final_perplexity'

# 查看训练过程
tail -f ./results/latest_experiment/logs/training.log
```

### 使用内置报告工具
```bash
# 生成可视化报告
python scripts/generate_report.py --experiment_dir ./results/your_experiment

# 比较多个实验
python scripts/compare_experiments.py --experiments_dir ./results

# 导出为CSV
python scripts/export_metrics.py --experiment_dir ./results/your_experiment --format csv
```

## 📈 自动报告生成

框架会自动生成：
1. **实时监控**: WandB仪表板 (如果启用)
2. **终端输出**: 训练过程中的实时指标
3. **HTML报告**: 完整的实验报告 (`final_report.html`)
4. **JSON指标**: 机器可读的详细指标
5. **可视化图表**: 训练曲线和性能图表

## 🎯 快速定位关键指标

| 你想查看的指标 | 文件位置 | 关键字段 |
|---------------|----------|----------|
| 最终困惑度 | `metrics/evaluation_results.json` | `final_perplexity` |
| 训练损失历史 | `metrics/training_metrics.json` | `train_loss` |
| 参数效率 | `metrics/evaluation_results.json` | `parameter_efficiency` |
| 训练时间 | `metrics/evaluation_results.json` | `training_time` |
| 内存使用 | `metrics/evaluation_results.json` | `memory_usage_mb` |
| LoRA配置 | `config.yaml` | `peft.lora` |
| 错误日志 | `logs/training.log` | grep "ERROR" |