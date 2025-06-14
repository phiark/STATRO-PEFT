#!/bin/bash
# STRATO-PEFT Docker部署脚本
# 一键部署优化的Docker环境，支持多种GPU平台

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认配置
DEFAULT_PLATFORM="cuda"
DEFAULT_BUILD_ENV="production"
DEFAULT_PROFILE="production"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 日志函数
log_info() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[DEPLOY]${NC} $1"
}

log_error() {
    echo -e "${RED}[DEPLOY]${NC} $1"
}

log_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
STRATO-PEFT Docker部署脚本

用法: $0 [选项] [命令]

选项:
  --platform PLATFORM      GPU平台 (cuda|rocm|cpu) [默认: cuda]
  --build-env ENV          构建环境 (production|development) [默认: production] 
  --profile PROFILE        Docker Compose配置文件 [默认: production]
  --no-cache              禁用Docker构建缓存
  --parallel              并行构建多个平台
  --gpu-ids IDS           指定GPU IDs，逗号分隔 [默认: 0]
  --memory MEMORY         限制内存使用，如16g [默认: 无限制]
  --cpus CPUS             限制CPU使用，如4.0 [默认: 无限制]
  --help                  显示此帮助信息

命令:
  build                   构建Docker镜像
  deploy                  部署服务 (默认)
  start                   启动服务
  stop                    停止服务
  restart                 重启服务
  logs                    查看日志
  status                  查看状态
  clean                   清理未使用的镜像和容器
  shell                   进入容器shell
  smoke-test              运行烟雾测试
  health-check           检查容器健康状态
  
配置文件选择:
  production              生产环境 (默认)
  dev                     开发环境 (带Jupyter)
  training                训练任务
  monitoring              监控服务
  full                    完整环境 (所有服务)

示例:
  $0                                    # 部署CUDA生产环境
  $0 --platform cpu --profile dev      # 部署CPU开发环境
  $0 --parallel build                   # 并行构建所有平台
  $0 smoke-test                         # 运行烟雾测试
  $0 --profile monitoring start        # 启动监控服务

EOF
}

# 检测系统环境
detect_system() {
    log_header "检测系统环境"
    
    # 检查Docker
    if ! command -v docker > /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose > /dev/null && ! docker compose version > /dev/null 2>&1; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 设置compose命令
    if docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    log_info "Docker: $(docker --version)"
    log_info "Docker Compose: $($COMPOSE_CMD version --short 2>/dev/null || echo 'Unknown')"
    
    # 检查GPU支持
    GPU_AVAILABLE="false"
    if command -v nvidia-smi > /dev/null 2>&1; then
        if nvidia-smi > /dev/null 2>&1; then
            GPU_TYPE="nvidia"
            GPU_AVAILABLE="true"
            log_info "检测到NVIDIA GPU"
        fi
    elif command -v rocm-smi > /dev/null 2>&1; then
        GPU_TYPE="amd"
        GPU_AVAILABLE="true"
        log_info "检测到AMD ROCm GPU"
    else
        GPU_TYPE="none"
        log_info "未检测到GPU，将使用CPU"
    fi
    
    # 检查Docker GPU支持
    if [ "$GPU_AVAILABLE" = "true" ] && [ "$GPU_TYPE" = "nvidia" ]; then
        if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
            log_info "Docker GPU支持正常"
        else
            log_warn "Docker GPU支持可能有问题，请检查nvidia-container-toolkit"
        fi
    fi
}

# 准备部署环境
prepare_environment() {
    log_header "准备部署环境"
    
    cd "$PROJECT_DIR"
    
    # 创建必要的目录和子目录
    local dirs=(
        "results"           # 训练结果输出
        "logs"              # 日志文件
        "cache"             # 缓存文件
        "cache/huggingface" # HuggingFace模型缓存
        "data"              # 数据集存储
        "data/raw"          # 原始数据
        "data/processed"    # 处理后数据
        "models"            # 保存的模型
        "experiments"       # 实验记录
        "notebooks"         # Jupyter笔记本
        "docker/prometheus" # Prometheus配置
        "docker/grafana"    # Grafana配置
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
    
    # 设置目录权限，确保Docker容器可以写入
    chmod -R 755 results logs cache data models experiments notebooks 2>/dev/null || true
    
    # 创建.gitkeep文件保持目录结构
    for dir in "${dirs[@]}"; do
        if [ ! -f "$dir/.gitkeep" ] && [ -d "$dir" ]; then
            touch "$dir/.gitkeep"
        fi
    done
    
    # 检查requirements.txt
    if [ ! -f "requirements.txt" ]; then
        log_warn "requirements.txt不存在，创建基础版本"
        ./scripts/setup_python_env.sh --help > /dev/null || {
            log_info "运行Python环境脚本生成requirements.txt"
            bash -c "cd '$PROJECT_DIR' && python -c \"
import sys
print('# STRATO-PEFT基础依赖')
packages = ['torch>=2.0.0', 'transformers>=4.30.0', 'peft>=0.4.0', 'omegaconf>=2.3.0', 'rich>=13.0.0']
for pkg in packages:
    print(pkg)
\"" > requirements.txt
        }
    fi
    
    # 设置环境变量
    export STRATO_PLATFORM="${PLATFORM}"
    export STRATO_BUILD_ENV="${BUILD_ENV}"
    
    log_info "环境准备完成"
}

# 构建Docker镜像
build_images() {
    log_header "构建Docker镜像"
    
    local build_args=(
        "--build-arg" "PLATFORM=${PLATFORM}"
        "--build-arg" "PYTHON_VERSION=3.9"
        "--build-arg" "PYTORCH_VERSION=2.1.0"
        "--build-arg" "BUILD_ENV=${BUILD_ENV}"
    )
    
    if [ "$NO_CACHE" = "true" ]; then
        build_args+=("--no-cache")
    fi
    
    if [ "$PARALLEL_BUILD" = "true" ]; then
        log_info "并行构建多个平台..."
        
        # 后台构建不同平台
        platforms=("cuda" "rocm" "cpu")
        pids=()
        
        for platform in "${platforms[@]}"; do
            log_info "开始构建 $platform 平台..."
            (
                docker build \
                    "${build_args[@]}" \
                    --build-arg "PLATFORM=${platform}" \
                    --target production \
                    -t "strato-peft:${platform}-latest" \
                    .
            ) &
            pids+=($!)
        done
        
        # 等待所有构建完成
        for pid in "${pids[@]}"; do
            wait $pid
        done
        
        log_info "所有平台构建完成"
    else
        log_info "构建 ${PLATFORM} 平台镜像..."
        
        # 构建生产环境镜像
        docker build \
            "${build_args[@]}" \
            --target production \
            -t "strato-peft:${PLATFORM}-latest" \
            .
        
        # 如果是开发环境，同时构建开发镜像
        if [ "$BUILD_ENV" = "development" ]; then
            log_info "构建开发环境镜像..."
            docker build \
                "${build_args[@]}" \
                --target development \
                -t "strato-peft:dev-latest" \
                .
        fi
        
        log_info "镜像构建完成"
    fi
}

# 部署服务
deploy_services() {
    log_header "部署服务"
    
    # 设置Docker Compose配置
    local compose_profiles=()
    case "$PROFILE" in
        "production")
            compose_profiles=(${PLATFORM} production)
            ;;
        "dev"|"development")
            compose_profiles=(dev jupyter)
            ;;
        "training")
            compose_profiles=(${PLATFORM} training)
            ;;
        "monitoring")
            compose_profiles=(monitoring)
            ;;
        "full")
            compose_profiles=(${PLATFORM} production dev monitoring database cache)
            ;;
        *)
            compose_profiles=(${PROFILE})
            ;;
    esac
    
    # 设置环境变量
    export COMPOSE_PROFILES=$(IFS=, ; echo "${compose_profiles[*]}")
    export STRATO_PLATFORM="${PLATFORM}"
    export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
    
    # 资源限制
    if [ -n "$MEMORY_LIMIT" ]; then
        export STRATO_MEMORY_LIMIT="$MEMORY_LIMIT"
    fi
    if [ -n "$CPU_LIMIT" ]; then
        export STRATO_CPU_LIMIT="$CPU_LIMIT"
    fi
    
    log_info "使用配置: ${compose_profiles[*]}"
    log_info "平台: $PLATFORM"
    
    # 启动服务
    $COMPOSE_CMD --profile "${COMPOSE_PROFILES}" up -d
    
    log_info "服务部署完成"
    
    # 显示服务状态
    show_status
}

# 显示服务状态
show_status() {
    log_header "服务状态"
    
    $COMPOSE_CMD ps
    
    echo ""
    log_info "可用服务端口:"
    echo "  - Jupyter Lab: http://localhost:8888"
    echo "  - TensorBoard: http://localhost:6006"
    echo "  - Web UI: http://localhost:8080"
    echo "  - MLflow: http://localhost:5000"
    echo "  - WandB Local: http://localhost:8081"
    echo "  - Grafana: http://localhost:3000 (admin/admin123)"
    echo "  - Prometheus: http://localhost:9090"
}

# 运行烟雾测试
run_smoke_test() {
    log_header "运行烟雾测试"
    
    local container_name="strato-peft-${PLATFORM}"
    
    # 检查容器是否运行
    if ! docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        log_info "启动测试容器..."
        $COMPOSE_CMD --profile "$PLATFORM" up -d "$container_name"
        sleep 10
    fi
    
    log_info "执行烟雾测试..."
    docker exec "$container_name" python main.py \
        --config configs/gpt2_smoke_test.yaml \
        --fast-dev-run \
        --no-wandb
    
    log_info "烟雾测试完成"
}

# 健康检查
health_check() {
    log_header "健康检查"
    
    local containers=($($COMPOSE_CMD ps --services))
    
    for container in "${containers[@]}"; do
        log_info "检查容器: $container"
        if docker exec "${container}" python /app/docker/healthcheck.py; then
            log_info "✅ $container 健康状态正常"
        else
            log_warn "⚠️ $container 健康状态异常"
        fi
    done
}

# 清理环境
clean_environment() {
    log_header "清理环境"
    
    log_info "停止所有服务..."
    $COMPOSE_CMD down
    
    log_info "清理未使用的镜像..."
    docker image prune -f
    
    log_info "清理未使用的容器..."
    docker container prune -f
    
    log_info "清理未使用的网络..."
    docker network prune -f
    
    if [ "$1" = "--volumes" ]; then
        log_warn "清理数据卷..."
        docker volume prune -f
    fi
    
    log_info "清理完成"
}

# 进入容器shell
enter_shell() {
    local container_name="strato-peft-${PLATFORM}"
    
    if ! docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        log_info "启动容器..."
        $COMPOSE_CMD --profile "$PLATFORM" up -d "$container_name"
        sleep 5
    fi
    
    log_info "进入容器 $container_name..."
    docker exec -it "$container_name" /bin/bash
}

# 主函数
main() {
    # 设置默认值
    PLATFORM="$DEFAULT_PLATFORM"
    BUILD_ENV="$DEFAULT_BUILD_ENV"
    PROFILE="$DEFAULT_PROFILE"
    NO_CACHE="false"
    PARALLEL_BUILD="false"
    COMMAND="deploy"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --build-env)
                BUILD_ENV="$2"
                shift 2
                ;;
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --gpu-ids)
                GPU_IDS="$2"
                shift 2
                ;;
            --memory)
                MEMORY_LIMIT="$2"
                shift 2
                ;;
            --cpus)
                CPU_LIMIT="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --parallel)
                PARALLEL_BUILD="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            build|deploy|start|stop|restart|logs|status|clean|shell|smoke-test|health-check)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证平台参数
    if [[ ! "$PLATFORM" =~ ^(cuda|rocm|cpu)$ ]]; then
        log_error "无效的平台: $PLATFORM"
        exit 1
    fi
    
    # 执行检测
    detect_system
    prepare_environment
    
    # 执行命令
    case "$COMMAND" in
        "build")
            build_images
            ;;
        "deploy")
            build_images
            deploy_services
            ;;
        "start")
            $COMPOSE_CMD --profile "$PROFILE" up -d
            show_status
            ;;
        "stop")
            $COMPOSE_CMD down
            ;;
        "restart")
            $COMPOSE_CMD restart
            show_status
            ;;
        "logs")
            $COMPOSE_CMD logs -f
            ;;
        "status")
            show_status
            ;;
        "clean")
            clean_environment "$@"
            ;;
        "shell")
            enter_shell
            ;;
        "smoke-test")
            run_smoke_test
            ;;
        "health-check")
            health_check
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
    
    log_info "部署脚本执行完成"
}

# 信号处理
trap 'log_info "收到中断信号，正在清理..."; exit 0' SIGTERM SIGINT

# 运行主函数
main "$@"