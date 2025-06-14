# STRATO-PEFT Docker部署指南

本指南详细说明了STRATO-PEFT框架的Docker部署配置，特别是项目输出文件夹的自动挂载机制。

## 🚀 快速开始

### 一键部署（推荐）
```bash
# 自动检测环境并部署
./scripts/one_click_deploy.sh

# 仅Docker部署
./scripts/one_click_deploy.sh --type docker
```

### 手动部署
```bash
# 1. 创建项目目录结构
./scripts/setup_project_structure.sh --docker

# 2. 部署Docker环境
./scripts/deploy_docker.sh --platform cuda

# 3. 启动服务
docker-compose --profile production up -d
```

## 📁 输出文件夹自动挂载

### 自动创建的目录结构

所有部署脚本都会自动创建以下输出目录：

```
strato_peft_experimental_framework/
├── results/                    # 训练结果输出 🔄
│   ├── checkpoints/           # 模型检查点
│   ├── metrics/               # 训练指标JSON
│   ├── plots/                 # 可视化图表
│   └── reports/               # 自动生成报告
├── logs/                      # 日志文件 🔄
│   ├── training/              # 训练过程日志
│   ├── evaluation/            # 评估过程日志
│   ├── system/                # 系统资源日志
│   └── error/                 # 错误日志
├── cache/                     # 缓存文件 🔄
│   ├── huggingface/           # HF模型缓存
│   ├── datasets/              # 数据集缓存
│   └── compiled/              # 编译缓存
├── data/                      # 数据存储 🔄
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   ├── external/              # 外部数据
│   └── interim/               # 中间数据
├── models/                    # 模型存储 🔄
│   ├── pretrained/            # 预训练模型
│   ├── fine_tuned/            # 微调模型
│   └── adapters/              # PEFT适配器
├── experiments/               # 实验记录 🔄
│   ├── configs/               # 实验配置
│   ├── results/               # 实验结果
│   └── analysis/              # 结果分析
└── notebooks/                 # Jupyter笔记本 🔄
    ├── exploratory/           # 探索分析
    ├── training/              # 训练笔记本
    └── evaluation/            # 评估笔记本
```

🔄 = 自动挂载到Docker容器

### Docker挂载配置

在`docker-compose.yml`中的挂载配置：

```yaml
x-common-volumes: &common-volumes
  - ./results:/app/results:rw              # 训练结果输出
  - ./logs:/app/logs:rw                    # 日志文件
  - ./cache:/app/cache:rw                  # 缓存文件 
  - ./data:/app/data:rw                    # 数据集存储
  - ./models:/app/models:rw                # 保存的模型
  - ./experiments:/app/experiments:rw      # 实验记录
  - ./notebooks:/app/notebooks:rw          # Jupyter笔记本
  - ./configs:/app/configs:ro              # 配置文件（只读）
  - ./cache/huggingface:/app/cache/huggingface:rw  # HuggingFace缓存
```

## 🔧 自动化功能

### 1. 目录自动创建

所有部署脚本都包含自动目录创建功能：

- `./scripts/setup_project_structure.sh` - 专用目录结构创建脚本
- `./scripts/deploy_docker.sh` - Docker部署时自动创建
- `./scripts/one_click_deploy.sh` - 一键部署时自动创建
- `docker/entrypoint.sh` - 容器启动时验证和创建

### 2. 权限自动设置

根据部署模式自动设置合适的权限：

```bash
# Docker模式 - 确保容器可写
chmod -R 777 results logs cache data models experiments notebooks

# 原生模式 - 用户权限
chmod -R 755 results logs cache data models experiments notebooks

# 只读目录
chmod -R 644 configs/* src/*
```

### 3. .gitkeep文件

自动创建`.gitkeep`文件保持空目录结构：

```bash
# 每个输出目录都有.gitkeep文件
find results logs cache data models experiments notebooks -name ".gitkeep"
```

## 🐳 Docker部署选项

### 生产环境部署
```bash
# CUDA GPU
./scripts/deploy_docker.sh --platform cuda --profile production

# ROCm GPU  
./scripts/deploy_docker.sh --platform rocm --profile production

# CPU only
./scripts/deploy_docker.sh --platform cpu --profile production
```

### 开发环境部署
```bash
# 带Jupyter Lab的开发环境
./scripts/deploy_docker.sh --platform cuda --profile dev

# 访问地址
# - Jupyter Lab: http://localhost:8888
# - TensorBoard: http://localhost:6006
# - MLflow: http://localhost:5000
```

### 监控服务部署
```bash
# 启动监控服务
./scripts/deploy_docker.sh --profile monitoring

# 访问地址
# - TensorBoard: http://localhost:6007
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin123)
```

## 📊 输出文件位置

### 训练结果

训练完成后，可以在以下位置找到输出文件：

```bash
# 训练结果
ls -la results/your_experiment_name/

# 困惑度等指标 (JSON格式)
cat results/your_experiment_name/metrics/evaluation_results.json

# 训练过程日志
tail -f logs/training/training.log

# 模型检查点
ls -la results/your_experiment_name/checkpoints/
```

### 实时监控

```bash
# 查看容器日志
docker-compose logs -f strato-peft-cuda

# 查看训练进度
docker exec strato-peft-cuda tail -f /app/logs/training/training.log

# 检查GPU使用
docker exec strato-peft-cuda nvidia-smi
```

## 🔍 故障排除

### 常见问题

#### 1. 目录权限问题
```bash
# 症状：容器无法写入文件
# 解决：重新设置权限
./scripts/setup_project_structure.sh --docker
```

#### 2. 挂载点不存在
```bash
# 症状：Docker启动失败，找不到挂载目录
# 解决：创建目录结构
./scripts/setup_project_structure.sh
```

#### 3. 缓存问题
```bash
# 症状：模型下载重复或失败
# 解决：清理并重建缓存目录
rm -rf cache/huggingface/*
./scripts/setup_project_structure.sh --docker
```

#### 4. 磁盘空间不足
```bash
# 检查磁盘使用
du -sh results/ logs/ cache/ models/

# 清理旧的实验结果
find results/ -name "*" -mtime +30 -delete

# 清理Docker缓存
docker system prune -a
```

### 健康检查

```bash
# 容器健康检查
./scripts/deploy_docker.sh health-check

# 目录结构验证
./scripts/setup_project_structure.sh

# GPU环境验证
./scripts/verify_gpu.sh
```

## 📈 性能优化

### 1. 存储优化

```bash
# 使用SSD存储挂载关键目录
# 在docker-compose.yml中配置
volumes:
  - /fast-ssd/strato-peft/results:/app/results:rw
  - /fast-ssd/strato-peft/cache:/app/cache:rw
```

### 2. 网络存储

```bash
# 使用网络存储保存大型模型
volumes:
  - nfs-server:/shared/models:/app/models:rw
  - local-ssd:/app/cache:rw  # 缓存仍使用本地SSD
```

### 3. 内存优化

```yaml
# 在docker-compose.yml中限制内存使用
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      memory: 8G
```

## 📝 最佳实践

### 1. 定期备份
```bash
# 备份重要结果
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/
aws s3 cp results_backup_*.tar.gz s3://your-backup-bucket/
```

### 2. 日志轮转
```bash
# 设置日志轮转避免磁盘满
echo "*/6 * * * * find /path/to/logs -name '*.log' -mtime +7 -delete" | crontab -
```

### 3. 监控磁盘使用
```bash
# 添加磁盘监控脚本
echo "df -h | grep -E '^/dev/' | awk '{if($5 > 80) print $0}'" > check_disk.sh
chmod +x check_disk.sh
```

## 🚀 高级配置

### 多节点部署
```yaml
# docker-compose-cluster.yml
version: '3.8'
services:
  strato-peft-master:
    extends: strato-peft-cuda
    deploy:
      placement:
        constraints: [node.role == manager]
  
  strato-peft-worker:
    extends: strato-peft-cuda
    deploy:
      replicas: 3
      placement:
        constraints: [node.role == worker]
```

### 自定义挂载
```yaml
# 添加自定义挂载点
volumes:
  - ./custom_data:/app/custom_data:rw
  - ~/.ssh:/home/strato/.ssh:ro
  - /etc/timezone:/etc/timezone:ro
```

---

## 📞 获取帮助

如果遇到问题，请按以下顺序排查：

1. 查看容器日志：`docker-compose logs strato-peft-cuda`
2. 运行健康检查：`./scripts/deploy_docker.sh health-check`
3. 检查目录权限：`ls -la results/ logs/ cache/`
4. 查看部署日志：`cat deployment.log`
5. 参考故障排除文档：`CLAUDE.md`

**记住**：所有输出文件夹都会自动创建和挂载，无需手动干预！🎉