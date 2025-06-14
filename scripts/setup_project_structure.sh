#!/bin/bash
# STRATO-PEFT 项目目录结构初始化脚本
# 创建所有必要的输出文件夹，确保Docker挂载正常工作

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 日志函数
log_info() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[SETUP]${NC} $1"
}

log_error() {
    echo -e "${RED}[SETUP]${NC} $1"
}

log_header() {
    echo -e "${BLUE}===========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===========================================${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
STRATO-PEFT 项目目录结构初始化脚本

用法: $0 [选项]

选项:
  --docker               为Docker环境优化权限
  --reset               重置所有目录（删除后重建）
  --gitkeep             创建.gitkeep文件保持目录结构
  --no-permissions      跳过权限设置
  --help                显示此帮助信息

功能:
  - 创建完整的项目目录结构
  - 设置适当的文件权限
  - 创建.gitkeep文件保持空目录
  - 生成目录结构说明文档

EOF
}

# 创建项目目录结构
create_project_structure() {
    log_header "创建STRATO-PEFT项目目录结构"
    
    cd "$PROJECT_DIR"
    
    # 定义完整的目录结构
    local directories=(
        # 核心输出目录
        "results"                    # 训练结果和实验输出
        "results/checkpoints"        # 模型检查点
        "results/metrics"            # 训练指标和评估结果
        "results/plots"              # 可视化图表
        "results/reports"            # 自动生成的报告
        
        # 日志目录
        "logs"                       # 应用日志
        "logs/training"              # 训练日志
        "logs/evaluation"            # 评估日志
        "logs/system"                # 系统日志
        "logs/error"                 # 错误日志
        
        # 缓存目录
        "cache"                      # 通用缓存
        "cache/huggingface"          # HuggingFace模型缓存
        "cache/datasets"             # 数据集缓存
        "cache/compiled"             # 编译缓存
        
        # 数据目录
        "data"                       # 数据根目录
        "data/raw"                   # 原始数据
        "data/processed"             # 处理后的数据
        "data/external"              # 外部数据源
        "data/interim"               # 中间处理数据
        
        # 模型目录
        "models"                     # 保存的模型
        "models/pretrained"          # 预训练模型
        "models/fine_tuned"          # 微调后的模型
        "models/adapters"            # PEFT适配器
        
        # 实验目录
        "experiments"                # 实验记录
        "experiments/configs"        # 实验配置
        "experiments/results"        # 实验结果
        "experiments/analysis"       # 结果分析
        
        # 开发目录
        "notebooks"                  # Jupyter笔记本
        "notebooks/exploratory"     # 探索性分析
        "notebooks/training"         # 训练相关笔记本
        "notebooks/evaluation"      # 评估相关笔记本
        
        # Docker相关目录
        "docker/prometheus"          # Prometheus监控配置
        "docker/grafana"             # Grafana仪表板配置
        "docker/logs"                # Docker容器日志
        
        # 文档目录
        "docs"                       # 项目文档
        "docs/images"                # 文档图片
        "docs/api"                   # API文档
        
        # 测试目录
        "tests/unit"                 # 单元测试
        "tests/integration"          # 集成测试
        "tests/data"                 # 测试数据
        
        # 配置目录（如果不存在）
        "configs"                    # 配置文件
        "configs/experiments"        # 实验配置
        "configs/models"             # 模型配置
        "configs/datasets"           # 数据集配置
        
        # 工具目录
        "tools"                      # 实用工具
        "tools/scripts"              # 辅助脚本
        "tools/analysis"             # 分析工具
    )
    
    # 创建目录
    local created_count=0
    local existing_count=0
    
    for dir in "${directories[@]}"; do
        if [ "$RESET_DIRS" = "true" ] && [ -d "$dir" ]; then
            log_warn "重置目录: $dir"
            rm -rf "$dir"
        fi
        
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
            ((created_count++))
        else
            ((existing_count++))
        fi
    done
    
    log_info "目录创建完成 - 新建: $created_count, 已存在: $existing_count"
}

# 设置目录权限
set_permissions() {
    if [ "$SKIP_PERMISSIONS" = "true" ]; then
        log_info "跳过权限设置"
        return
    fi
    
    log_header "设置目录权限"
    
    # 可写目录 - Docker容器需要写入权限
    local writable_dirs=(
        "results" "logs" "cache" "data" "models" 
        "experiments" "notebooks" "docker/logs"
    )
    
    for dir in "${writable_dirs[@]}"; do
        if [ -d "$dir" ]; then
            if [ "$DOCKER_MODE" = "true" ]; then
                # Docker模式：设置为777确保容器可写
                chmod -R 777 "$dir" 2>/dev/null || {
                    log_warn "无法设置 $dir 权限为777，尝试755"
                    chmod -R 755 "$dir" 2>/dev/null || true
                }
                log_info "设置Docker权限: $dir (777)"
            else
                # 原生模式：设置为755
                chmod -R 755 "$dir" 2>/dev/null || true
                log_info "设置权限: $dir (755)"
            fi
        fi
    done
    
    # 只读目录
    local readonly_dirs=("configs" "src")
    
    for dir in "${readonly_dirs[@]}"; do
        if [ -d "$dir" ]; then
            chmod -R 644 "$dir"/* 2>/dev/null || true
            chmod 755 "$dir" 2>/dev/null || true
            log_info "设置只读权限: $dir"
        fi
    done
    
    # 脚本目录 - 可执行权限
    if [ -d "scripts" ]; then
        chmod 755 scripts/*.sh 2>/dev/null || true
        log_info "设置脚本执行权限: scripts/"
    fi
}

# 创建.gitkeep文件
create_gitkeep_files() {
    if [ "$CREATE_GITKEEP" = "false" ]; then
        log_info "跳过.gitkeep文件创建"
        return
    fi
    
    log_header "创建.gitkeep文件"
    
    # 需要保持的空目录
    local keep_dirs=(
        "results" "results/checkpoints" "results/metrics" "results/plots" "results/reports"
        "logs" "logs/training" "logs/evaluation" "logs/system" "logs/error"
        "cache" "cache/huggingface" "cache/datasets" "cache/compiled"
        "data/raw" "data/processed" "data/external" "data/interim"
        "models" "models/pretrained" "models/fine_tuned" "models/adapters"
        "experiments" "experiments/configs" "experiments/results" "experiments/analysis"
        "notebooks" "notebooks/exploratory" "notebooks/training" "notebooks/evaluation"
        "docker/logs" "tests/data"
    )
    
    local gitkeep_count=0
    
    for dir in "${keep_dirs[@]}"; do
        if [ -d "$dir" ] && [ ! -f "$dir/.gitkeep" ]; then
            cat > "$dir/.gitkeep" << EOF
# .gitkeep文件
# 此文件用于保持Git仓库中的空目录结构
# 特别是为了确保Docker挂载点正确工作
# 创建时间: $(date)
# 目录用途: $(get_directory_purpose "$dir")
EOF
            ((gitkeep_count++))
            log_info "创建.gitkeep: $dir"
        fi
    done
    
    log_info "创建.gitkeep文件完成: $gitkeep_count 个"
}

# 获取目录用途说明
get_directory_purpose() {
    local dir="$1"
    case "$dir" in
        "results"*) echo "训练结果和实验输出" ;;
        "logs"*) echo "应用和训练日志" ;;
        "cache"*) echo "缓存文件和临时数据" ;;
        "data"*) echo "数据集和数据文件" ;;
        "models"*) echo "模型文件和检查点" ;;
        "experiments"*) echo "实验配置和结果" ;;
        "notebooks"*) echo "Jupyter笔记本" ;;
        "docker"*) echo "Docker相关配置和日志" ;;
        "tests"*) echo "测试相关文件" ;;
        *) echo "项目文件存储" ;;
    esac
}

# 生成目录结构文档
generate_structure_documentation() {
    log_header "生成目录结构文档"
    
    local doc_file="PROJECT_STRUCTURE.md"
    
    cat > "$doc_file" << 'EOF'
# STRATO-PEFT 项目目录结构

本文档描述了STRATO-PEFT项目的完整目录结构，包括各目录的用途和Docker挂载配置。

## 核心目录结构

```
strato_peft_experimental_framework/
├── src/                    # 源代码
├── configs/                # 配置文件
│   ├── experiments/        # 实验配置
│   ├── models/            # 模型配置
│   └── datasets/          # 数据集配置
├── scripts/               # 部署和管理脚本
├── docker/                # Docker相关文件
│   ├── prometheus/        # Prometheus配置
│   ├── grafana/          # Grafana配置
│   └── logs/             # Docker容器日志
└── requirements.txt       # Python依赖
```

## 输出目录（Docker挂载）

### 训练结果 (`results/`)
```
results/
├── checkpoints/           # 模型检查点
├── metrics/              # 训练指标JSON文件
├── plots/                # 可视化图表
└── reports/              # 自动生成的报告
```

### 日志文件 (`logs/`)
```
logs/
├── training/             # 训练过程日志
├── evaluation/           # 评估过程日志
├── system/               # 系统资源日志
└── error/                # 错误日志
```

### 缓存文件 (`cache/`)
```
cache/
├── huggingface/          # HuggingFace模型缓存
├── datasets/             # 数据集缓存
└── compiled/             # 编译缓存
```

### 数据存储 (`data/`)
```
data/
├── raw/                  # 原始数据
├── processed/            # 处理后数据
├── external/             # 外部数据源
└── interim/              # 中间处理数据
```

### 模型存储 (`models/`)
```
models/
├── pretrained/           # 预训练模型
├── fine_tuned/          # 微调后模型
└── adapters/            # PEFT适配器
```

### 实验记录 (`experiments/`)
```
experiments/
├── configs/             # 实验配置备份
├── results/             # 实验结果
└── analysis/            # 结果分析
```

### 开发文件 (`notebooks/`)
```
notebooks/
├── exploratory/         # 探索性分析
├── training/            # 训练相关笔记本
└── evaluation/          # 评估相关笔记本
```

## Docker挂载配置

所有输出目录都配置为Docker挂载点，确保容器内外数据同步：

```yaml
volumes:
  - ./results:/app/results:rw
  - ./logs:/app/logs:rw
  - ./cache:/app/cache:rw
  - ./data:/app/data:rw
  - ./models:/app/models:rw
  - ./experiments:/app/experiments:rw
  - ./notebooks:/app/notebooks:rw
  - ./configs:/app/configs:ro
```

## 权限设置

- **可写目录**: `results`, `logs`, `cache`, `data`, `models`, `experiments`, `notebooks`
  - Docker模式: 777 (确保容器可写)
  - 原生模式: 755 (用户可写)

- **只读目录**: `configs`, `src`
  - 权限: 644 (文件) / 755 (目录)

- **可执行目录**: `scripts`
  - 权限: 755 (脚本文件)

## 使用说明

### 初始化项目结构
```bash
# 创建完整目录结构
./scripts/setup_project_structure.sh

# Docker环境优化
./scripts/setup_project_structure.sh --docker

# 重置所有目录
./scripts/setup_project_structure.sh --reset
```

### 查看训练结果
```bash
# 查看最新实验结果
ls -la results/
cat results/*/metrics/evaluation_results.json

# 查看训练日志
tail -f logs/training/training.log
```

### 清理缓存
```bash
# 清理HuggingFace缓存
rm -rf cache/huggingface/*

# 清理所有缓存
rm -rf cache/*
```

## 注意事项

1. **Docker挂载**: 确保在运行Docker容器前创建所有目录
2. **权限问题**: 如果遇到权限错误，重新运行权限设置脚本
3. **存储空间**: 定期清理`cache/`和旧的`results/`以节省空间
4. **备份**: 重要的训练结果建议定期备份到云存储

---
**文档生成时间**: $(date)
**脚本版本**: setup_project_structure.sh v1.0
EOF

    log_info "目录结构文档已生成: $doc_file"
}

# 验证目录结构
verify_structure() {
    log_header "验证目录结构"
    
    local issues=0
    local total_dirs=0
    
    # 检查关键目录
    local critical_dirs=("results" "logs" "cache" "data" "models")
    
    for dir in "${critical_dirs[@]}"; do
        ((total_dirs++))
        if [ ! -d "$dir" ]; then
            log_error "关键目录缺失: $dir"
            ((issues++))
        elif [ ! -w "$dir" ]; then
            log_warn "目录不可写: $dir"
            ((issues++))
        else
            log_info "✓ $dir"
        fi
    done
    
    # 检查Docker compose挂载点
    if [ -f "docker-compose.yml" ]; then
        log_info "检查Docker挂载点配置..."
        local mount_dirs=($(grep -o '\./[^:]*' docker-compose.yml | sed 's/^\.\///' | sort | uniq))
        
        for mount_dir in "${mount_dirs[@]}"; do
            if [ ! -d "$mount_dir" ]; then
                log_warn "Docker挂载点目录不存在: $mount_dir"
                ((issues++))
            fi
        done
    fi
    
    echo ""
    if [ $issues -eq 0 ]; then
        log_info "✅ 目录结构验证通过 ($total_dirs 个目录)"
        return 0
    else
        log_warn "⚠️ 发现 $issues 个问题，请检查"
        return 1
    fi
}

# 主函数
main() {
    # 默认配置
    DOCKER_MODE="false"
    RESET_DIRS="false"
    CREATE_GITKEEP="true"
    SKIP_PERMISSIONS="false"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                DOCKER_MODE="true"
                shift
                ;;
            --reset)
                RESET_DIRS="true"
                shift
                ;;
            --gitkeep)
                CREATE_GITKEEP="true"
                shift
                ;;
            --no-permissions)
                SKIP_PERMISSIONS="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_header "STRATO-PEFT 项目目录结构初始化"
    log_info "项目目录: $PROJECT_DIR"
    log_info "Docker模式: $DOCKER_MODE"
    log_info "重置目录: $RESET_DIRS"
    
    # 执行主要功能
    create_project_structure
    set_permissions
    create_gitkeep_files
    generate_structure_documentation
    verify_structure
    
    log_header "初始化完成"
    log_info "🎉 STRATO-PEFT项目目录结构已成功初始化！"
    
    echo ""
    echo "后续步骤:"
    echo "1. 查看 PROJECT_STRUCTURE.md 了解目录用途"
    echo "2. 运行 Docker: ./scripts/deploy_docker.sh"
    echo "3. 运行训练: python main.py --config configs/your_config.yaml"
    echo ""
}

# 运行主函数
main "$@"