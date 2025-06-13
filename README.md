# STRATO-PEFT 实验框架

> **Strategic Resource-Aware Tunable Optimization for Parameter-Efficient Fine-Tuning**

## 项目概述

本项目实现了 STRATO-PEFT 方法的完整实验框架，用于与 LoRA、AdaLoRA、DoRA 等基线方法进行对比实验。支持多平台部署（CUDA、ROCm、Apple Silicon、CPU）和 Docker 容器化运行。

### 核心特性

- **成本感知优化**: 基于参数、FLOPs、VRAM 的多维成本建模
- **强化学习智能体**: PPO 策略用于适配器放置和秩分配
- **动态秩调度**: 支持训练过程中的秩增长和收缩
- **Go-Explore 缓存**: 避免重复探索，提高搜索效率
- **多任务支持**: MMLU、GSM8K、HumanEval 等标准评估任务
- **多平台支持**: CUDA、ROCm、Apple Silicon (MPS)、CPU
- **Docker 容器化**: 完整的 Docker 支持，包含多架构构建
- **配置管理**: 智能的平台适配和资源优化

## 项目结构

```
strato_peft_experimental_framework/
├── main.py                          # 主入口脚本
├── configs/                         # 实验配置文件
│   ├── base_config.yaml            # 基础配置模板
│   ├── llama2_7b_mmlu_lora.yaml    # LoRA 基线配置
│   └── llama2_7b_mmlu_strato.yaml  # STRATO-PEFT 配置
├── src/                             # 核心源代码
│   ├── models/                      # 模型定义与加载
│   ├── peft_methods/                # PEFT 方法实现
│   ├── tasks/                       # 任务特定逻辑
│   ├── trainer.py                   # 训练循环
│   └── utils/                       # 工具函数
├── scripts/                         # 辅助脚本
│   ├── eval.py                      # 评估脚本
│   └── compare.py                   # 结果比较与可视化
├── results/                         # 实验结果存储
└── requirements.txt                 # Python 依赖
```

## 实验阶段

### P-0: 冒烟测试
- **模型**: Llama-2-7B
- **任务**: Alpaca-eval 子集 (10k)
- **目标**: 验证 RL 循环稳定性，≤2小时/种子

### P-1: 核心验证
- **模型**: Llama-2-7B
- **任务**: MMLU-dev, GSM8K-easy, HumanEval-10%
- **目标**: ≥LoRA 分数，-20% 参数

### P-2: 扩展验证
- **模型**: Llama-2-13B
- **任务**: 同 P-1
- **目标**: 复制 P-1 收益，测量 FLOPs 减少

## 系统要求

### 基础要求
- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM

### 平台特定要求

#### CUDA
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA 11.8+ 或 12.0+
- cuDNN 8.0+

#### ROCm
- AMD GPU (支持 ROCm)
- ROCm 5.4+

#### Apple Silicon
- M1/M2/M3 Mac
- macOS 12.0+
- Metal Performance Shaders

#### CPU
- 多核 CPU (推荐 8+ 核心)
- 16GB+ RAM (推荐)

## 安装

### 方法 1: 本地安装

```bash
# 创建虚拟环境 (推荐使用 conda)
conda create -n strato-peft python=3.9
conda activate strato-peft

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 方法 2: Docker 安装

```bash
# 自动检测平台并构建
./scripts/docker_run.sh build

# 或指定平台构建
./scripts/docker_run.sh build --platform cuda
./scripts/docker_run.sh build --platform rocm
./scripts/docker_run.sh build --platform cpu
```

## 快速开始

### 1. 本地运行

```bash
# LoRA 基线
python main.py --config configs/llama2_7b_mmlu_lora.yaml --seed 42

# STRATO-PEFT
python main.py --config configs/llama2_7b_mmlu_strato.yaml --seed 42

# 指定平台运行
python main.py --config configs/llama2_7b_mmlu_strato.yaml --platform cuda

# 调试模式
python main.py --config configs/llama2_7b_mmlu_strato.yaml --debug

# 验证配置（不运行训练）
python main.py --config configs/llama2_7b_mmlu_strato.yaml --dry-run
```

### 2. Docker 运行

```bash
# 自动检测平台运行
./scripts/docker_run.sh train configs/llama2_7b_mmlu_strato.yaml

# 指定平台运行
./scripts/docker_run.sh train configs/llama2_7b_mmlu_strato.yaml --platform cuda

# 交互式运行
./scripts/docker_run.sh bash

# 启动 Jupyter
./scripts/docker_run.sh jupyter
```

### 3. 评估与比较

```bash
# 评估单个检查点
python scripts/eval.py --checkpoint results/llama2_7b/mmlu/lora_rank16_seed42/

# 比较多个方法
python scripts/compare.py --results_dir results/llama2_7b/mmlu/

# Docker 中运行评估
./scripts/docker_run.sh eval results/llama2_7b/mmlu/lora_rank16_seed42/
```

## Docker 使用详解

### 构建镜像

```bash
# 自动检测平台
./scripts/docker_run.sh build

# 指定平台构建
./scripts/docker_run.sh build --platform cuda
./scripts/docker_run.sh build --platform rocm
./scripts/docker_run.sh build --platform cpu
```

### 运行容器

```bash
# 训练
./scripts/docker_run.sh train configs/my_experiment.yaml

# 评估
./scripts/docker_run.sh eval results/my_experiment

# 交互式 bash
./scripts/docker_run.sh bash

# Jupyter Notebook
./scripts/docker_run.sh jupyter

# TensorBoard
./scripts/docker_run.sh tensorboard
```

### Docker Compose

```bash
# CUDA 环境
docker-compose --profile cuda up

# ROCm 环境
docker-compose --profile rocm up

# CPU 环境
docker-compose --profile cpu up

# 开发环境（包含 Jupyter）
docker-compose --profile dev up
```

## 项目结构

```
strato_peft_experimental_framework/
├── configs/                    # 实验配置文件
│   ├── base/                  # 基础配置
│   ├── models/                # 模型配置
│   ├── tasks/                 # 任务配置
│   └── experiments/           # 完整实验配置
├── src/                       # 源代码
│   ├── models/               # 模型实现
│   ├── peft/                 # PEFT 方法
│   ├── tasks/                # 任务和数据集
│   ├── training/             # 训练逻辑
│   ├── evaluation/           # 评估工具
│   └── utils/                # 工具函数
├── scripts/                   # 脚本工具
│   ├── docker_run.sh         # Docker 运行脚本
│   ├── eval.py               # 评估脚本
│   └── compare.py            # 比较脚本
├── docker/                    # Docker 配置
│   ├── Dockerfile.cuda       # CUDA 镜像
│   ├── Dockerfile.rocm       # ROCm 镜像
│   └── Dockerfile.cpu        # CPU 镜像
├── results/                   # 实验结果
├── logs/                      # 日志文件
├── requirements.txt           # Python 依赖
├── docker-compose.yml         # Docker Compose 配置
└── README.md                  # 项目文档
```

## 配置说明

### 基础配置结构

```yaml
# 模型配置
model:
  name: "llama2_7b"
  path: "/path/to/model"
  device_map: "auto"

# PEFT 配置
peft:
  method: "strato"  # lora, adalora, strato
  rank: 16
  alpha: 32
  dropout: 0.1

# 任务配置
task:
  name: "mmlu"
  data_path: "/path/to/data"
  max_length: 512

# 训练配置
training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 3
  gradient_accumulation_steps: 16

# 设备配置
device_info:
  device_type: "auto"  # auto, cuda, rocm, mps, cpu
  mixed_precision: true
  compile_model: false
```

### 平台特定配置

#### CUDA 配置
```yaml
device_info:
  device_type: "cuda"
  gpu_ids: [0, 1]  # 多 GPU 支持
  mixed_precision: true
  compile_model: true
```

#### ROCm 配置
```yaml
device_info:
  device_type: "rocm"
  mixed_precision: true
  compile_model: false
```

#### Apple Silicon 配置
```yaml
device_info:
  device_type: "mps"
  mixed_precision: false  # MPS 不支持 AMP
  compile_model: false
```

## 实验管理

### 运行实验

```bash
# 基础实验
python main.py --config configs/experiments/llama2_mmlu_baseline.yaml

# 多种子实验
for seed in 42 43 44; do
    python main.py --config configs/experiments/llama2_mmlu_strato.yaml --seed $seed
done

# 批量实验
python scripts/run_experiments.py --config_dir configs/experiments/
```

### 结果分析

```bash
# 生成报告
python scripts/generate_report.py --results_dir results/

# 可视化结果
python scripts/visualize_results.py --results_dir results/

# 统计分析
python scripts/statistical_analysis.py --results_dir results/
```

## 监控与调试

### WandB 集成

```bash
# 登录 WandB
wandb login

# 运行实验（自动记录）
python main.py --config configs/experiments/my_experiment.yaml

# 禁用 WandB
python main.py --config configs/experiments/my_experiment.yaml --no-wandb
```

### 性能分析

```bash
# 启用性能分析
python main.py --config configs/experiments/my_experiment.yaml --profile

# 查看性能报告
python scripts/analyze_profile.py --profile_dir logs/profiles/
```

### 调试模式

```bash
# 调试模式（详细日志）
python main.py --config configs/experiments/my_experiment.yaml --debug

# 快速开发模式（小数据集）
python main.py --config configs/experiments/my_experiment.yaml --fast-dev-run
```

## 成功标准

- **效率胜利**: 参数 ≤ 0.7×LoRA_参数 或 FLOPs ≤ 0.8×LoRA_FLOPs
- **准确率护栏**: 任务分数 ≥ 基线分数 - 0.5
- **稳定性**: 种子间标准差 ≤ 0.4

## 日志与监控

- **WandB 项目**: `strato-peft`
- **实验标签**: `model/task/phase/lambdaX_rankY_seedZ`
- **性能分析**: 每轮次 `torch.profiler` 10步窗口

## 贡献指南

### 开发环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd strato_peft_experimental_framework

# 安装开发依赖
pip install -r requirements-dev.txt
pip install -e .

# 安装 pre-commit hooks
pre-commit install
```

### 代码规范

- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 使用 flake8 进行代码检查
- 使用 mypy 进行类型检查

```bash
# 运行代码检查
black src/ scripts/
isort src/ scripts/
flake8 src/ scripts/
mypy src/
```

### 测试

```bash
# 运行单元测试
pytest tests/

# 运行集成测试
pytest tests/integration/

# 生成覆盖率报告
pytest --cov=src tests/
```

## 故障排除

### 常见问题

#### CUDA 内存不足
```bash
# 减少批次大小
python main.py --config configs/my_experiment.yaml --override training.batch_size=2

# 启用梯度检查点
python main.py --config configs/my_experiment.yaml --override training.gradient_checkpointing=true
```

#### 模型加载失败
```bash
# 检查模型路径
python -c "from transformers import AutoModel; AutoModel.from_pretrained('your_model_path')"

# 使用本地模型
python main.py --config configs/my_experiment.yaml --override model.path=/local/path/to/model
```

#### Docker 权限问题
```bash
# 添加用户到 docker 组
sudo usermod -aG docker $USER

# 重新登录或运行
newgrp docker
```

### 日志分析

```bash
# 查看训练日志
tail -f logs/training.log

# 查看错误日志
grep ERROR logs/training.log

# 分析性能日志
python scripts/analyze_logs.py --log_file logs/training.log
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 引用

如果您在研究中使用了本框架，请引用：

```bibtex
@article{strato-peft-2024,
  title={STRATO-PEFT: Strategic Parameter-Efficient Fine-Tuning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 支持

- 📧 邮件支持: [your-email@domain.com]
- 🐛 问题报告: [GitHub Issues]
- 💬 讨论交流: [GitHub Discussions]
- 📖 文档: [项目 Wiki]

## 注意事项

- 所有实验使用固定种子 {42, 43, 44} 确保可重现性
- 梯度累积步数固定为 16
- 启用混合精度训练以节省内存
- STRATO 控制器仅增加 ≤0.5M 参数开销

---

*本框架严格遵循论文中的实验协议和评估标准*