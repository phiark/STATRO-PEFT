#!/bin/bash
# STRATO-PEFT Docker容器入口点脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${GREEN}[CONTAINER]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[CONTAINER]${NC} $1"
}

log_error() {
    echo -e "${RED}[CONTAINER]${NC} $1"
}

log_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}"
}

# 显示容器启动信息
show_startup_info() {
    log_header "STRATO-PEFT Container Starting"
    log_info "Platform: ${STRATO_PLATFORM:-unknown}"
    log_info "Python: $(python --version)"
    log_info "Working Directory: $(pwd)"
    log_info "User: $(whoami)"
    log_info "Home: ${HOME}"
    
    # 显示GPU信息
    if command -v nvidia-smi > /dev/null 2>&1; then
        log_info "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | \
            while read name memory_total memory_used; do
                log_info "GPU: $name (${memory_used}MB/${memory_total}MB used)"
            done
    elif command -v rocm-smi > /dev/null 2>&1; then
        log_info "AMD ROCm GPU detected"
        rocm-smi --showproductname --showmeminfo --csv | head -5
    else
        log_info "No GPU detected or using CPU mode"
    fi
    
    # 显示PyTorch信息
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS Available: True')
"
}

# 设置环境
setup_environment() {
    log_info "Setting up environment..."
    
    # 确保所有必要的目录存在
    local required_dirs=(
        "/app/results" "/app/results/checkpoints" "/app/results/metrics" "/app/results/plots"
        "/app/logs" "/app/logs/training" "/app/logs/evaluation" "/app/logs/system"
        "/app/cache" "/app/cache/huggingface" "/app/cache/datasets"
        "/app/data" "/app/data/raw" "/app/data/processed"
        "/app/models" "/app/models/pretrained" "/app/models/fine_tuned"
        "/app/experiments" "/app/experiments/configs" "/app/experiments/results"
        "/app/notebooks" "/app/tmp"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # 设置权限
    if [ -w /app ]; then
        chmod -R 755 /app/scripts/ 2>/dev/null || true
        chmod -R 755 /app/docker/ 2>/dev/null || true
        # 确保输出目录可写
        chmod -R 777 /app/results /app/logs /app/cache /app/data /app/models /app/experiments /app/notebooks /app/tmp 2>/dev/null || true
    fi
    
    # 设置Python路径
    export PYTHONPATH="/app/src:${PYTHONPATH}"
    
    # 设置CUDA可见设备（如果未设置）
    if [ -z "${CUDA_VISIBLE_DEVICES}" ] && [ "${STRATO_PLATFORM}" = "cuda" ]; then
        export CUDA_VISIBLE_DEVICES=0
    fi
    
    # 优化PyTorch设置
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
    
    log_info "Environment setup complete"
}

# 健康检查
health_check() {
    log_info "Running health check..."
    
    # 检查Python环境
    if ! python -c "import torch, transformers, peft" 2>/dev/null; then
        log_error "Failed to import required packages"
        return 1
    fi
    
    # 检查配置文件
    if [ ! -f "/app/main.py" ]; then
        log_error "Main application file not found"
        return 1
    fi
    
    # 检查GPU（如果应该可用）
    if [ "${STRATO_PLATFORM}" = "cuda" ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log_warn "CUDA not available despite CUDA platform"
        fi
    fi
    
    log_info "Health check passed"
    return 0
}

# 处理特殊命令
handle_special_commands() {
    case "$1" in
        "bash"|"sh")
            log_info "Starting interactive shell"
            exec /bin/bash
            ;;
        "jupyter")
            log_info "Starting Jupyter Lab"
            exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
            ;;
        "tensorboard")
            log_info "Starting TensorBoard"
            exec tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006
            ;;
        "smoke-test")
            log_info "Running smoke test"
            exec python main.py --config configs/gpt2_smoke_test.yaml --fast-dev-run
            ;;
        "verify-gpu")
            log_info "Running GPU verification"
            exec ./scripts/verify_gpu.sh
            ;;
        "setup-env")
            log_info "Setting up Python environment"
            exec ./scripts/setup_python_env.sh
            ;;
        *)
            return 1
            ;;
    esac
}

# 主函数
main() {
    # 显示启动信息
    show_startup_info
    
    # 设置环境
    setup_environment
    
    # 运行健康检查
    if ! health_check; then
        log_error "Health check failed, but continuing..."
    fi
    
    # 处理特殊命令
    if [ $# -gt 0 ]; then
        if handle_special_commands "$1"; then
            exit 0
        fi
    fi
    
    # 执行传入的命令
    log_info "Executing command: $*"
    exec "$@"
}

# 信号处理
trap 'log_info "Received shutdown signal, cleaning up..."; exit 0' SIGTERM SIGINT

# 运行主函数
main "$@"