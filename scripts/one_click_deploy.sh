#!/bin/bash
# STRATO-PEFT 一键部署脚本
# 自动检测环境、安装依赖、配置GPU驱动、设置Python环境并部署项目

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置变量
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOYMENT_TYPE="auto"  # auto, docker, native, hybrid
SKIP_GPU_DRIVERS="false"
SKIP_PYTHON_ENV="false"
SKIP_DOCKER="false"
FORCE_REINSTALL="false"
INTERACTIVE_MODE="true"
DEPLOYMENT_LOG=""

# 日志函数
log_info() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
    [ -n "$DEPLOYMENT_LOG" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$DEPLOYMENT_LOG"
}

log_warn() {
    echo -e "${YELLOW}[DEPLOY]${NC} $1"
    [ -n "$DEPLOYMENT_LOG" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[DEPLOY]${NC} $1"
    [ -n "$DEPLOYMENT_LOG" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$DEPLOYMENT_LOG"
}

log_header() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
    [ -n "$DEPLOYMENT_LOG" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [HEADER] $1" >> "$DEPLOYMENT_LOG"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    [ -n "$DEPLOYMENT_LOG" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1" >> "$DEPLOYMENT_LOG"
}

log_step() {
    echo -e "${CYAN}🔄 $1${NC}"
    [ -n "$DEPLOYMENT_LOG" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP] $1" >> "$DEPLOYMENT_LOG"
}

# 显示欢迎信息
show_welcome() {
    clear
    cat << 'EOF'
 ███████╗████████╗██████╗  █████╗ ████████╗ ██████╗       ██████╗ ███████╗███████╗████████╗
 ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗      ██╔══██╗██╔════╝██╔════╝╚══██╔══╝
 ███████╗   ██║   ██████╔╝███████║   ██║   ██║   ██║█████╗██████╔╝█████╗  █████╗     ██║   
 ╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ██║   ██║╚════╝██╔═══╝ ██╔══╝  ██╔══╝     ██║   
 ███████║   ██║   ██║  ██║██║  ██║   ██║   ╚██████╔╝      ██║     ███████╗██║        ██║   
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝       ╚═╝     ╚══════╝╚═╝        ╚═╝   
                                                                                             
         Strategic Resource-Aware Tunable Optimization for Parameter-Efficient Fine-Tuning
                                    🚀 一键部署脚本 🚀
EOF
    echo ""
    echo -e "${PURPLE}欢迎使用STRATO-PEFT一键部署脚本！${NC}"
    echo -e "${CYAN}本脚本将自动检测您的环境并配置STRATO-PEFT框架。${NC}"
    echo ""
}

# 显示帮助信息
show_help() {
    cat << EOF
STRATO-PEFT 一键部署脚本

用法: $0 [选项]

选项:
  --type TYPE             部署类型 (auto|docker|native|hybrid) [默认: auto]
  --skip-gpu-drivers      跳过GPU驱动安装
  --skip-python-env       跳过Python环境设置
  --skip-docker          跳过Docker相关设置
  --force-reinstall      强制重新安装所有组件
  --non-interactive      非交互模式
  --log-file FILE        指定日志文件路径
  --help                 显示此帮助信息

部署类型说明:
  auto                   自动检测最适合的部署方式
  docker                 优先使用Docker部署
  native                 原生Python环境部署
  hybrid                 混合部署 (Docker + 原生环境)

示例:
  $0                                    # 自动部署
  $0 --type docker                      # Docker部署
  $0 --type native --skip-gpu-drivers   # 原生部署，跳过GPU驱动
  $0 --non-interactive --log-file deployment.log  # 静默部署

EOF
}

# 用户确认
confirm_action() {
    local message="$1"
    local default="${2:-n}"
    
    if [ "$INTERACTIVE_MODE" = "false" ]; then
        return 0
    fi
    
    while true; do
        if [ "$default" = "y" ]; then
            read -p "$message [Y/n]: " response
            response=${response:-y}
        else
            read -p "$message [y/N]: " response
            response=${response:-n}
        fi
        
        case $response in
            [Yy]|[Yy][Ee][Ss])
                return 0
                ;;
            [Nn]|[Nn][Oo])
                return 1
                ;;
            *)
                echo "请输入 y 或 n"
                ;;
        esac
    done
}

# 检测系统环境
detect_system_environment() {
    log_header "检测系统环境"
    
    # 操作系统检测
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO=$NAME
            VERSION=$VERSION_ID
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macOS"
        VERSION=$(sw_vers -productVersion)
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    log_info "操作系统: $DISTRO $VERSION"
    
    # 检测包管理器
    if command -v apt-get > /dev/null; then
        PACKAGE_MANAGER="apt"
    elif command -v yum > /dev/null; then
        PACKAGE_MANAGER="yum"
    elif command -v dnf > /dev/null; then
        PACKAGE_MANAGER="dnf"
    elif command -v brew > /dev/null; then
        PACKAGE_MANAGER="brew"
    else
        log_warn "未检测到支持的包管理器"
    fi
    
    # 检测GPU
    GPU_TYPE="none"
    if command -v nvidia-smi > /dev/null; then
        if nvidia-smi > /dev/null 2>&1; then
            GPU_TYPE="nvidia"
            GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
            log_info "检测到 $GPU_COUNT 块NVIDIA GPU"
        fi
    elif command -v rocm-smi > /dev/null; then
        GPU_TYPE="amd"
        log_info "检测到AMD ROCm GPU"
    elif [[ "$OS" == "macos" ]] && sysctl -n machdep.cpu.brand_string | grep -q "Apple"; then
        GPU_TYPE="apple"
        log_info "检测到Apple Silicon"
    fi
    
    # 检测已安装的软件
    DOCKER_INSTALLED="false"
    PYTHON_INSTALLED="false"
    GIT_INSTALLED="false"
    
    if command -v docker > /dev/null; then
        DOCKER_INSTALLED="true"
        DOCKER_VERSION=$(docker --version)
        log_info "Docker: $DOCKER_VERSION"
    fi
    
    if command -v python3 > /dev/null; then
        PYTHON_INSTALLED="true"
        PYTHON_VERSION=$(python3 --version)
        log_info "Python: $PYTHON_VERSION"
    fi
    
    if command -v git > /dev/null; then
        GIT_INSTALLED="true"
        GIT_VERSION=$(git --version)
        log_info "Git: $GIT_VERSION"
    fi
    
    # 检测内存和CPU
    if [[ "$OS" == "linux" ]]; then
        TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
        CPU_CORES=$(nproc)
    elif [[ "$OS" == "macos" ]]; then
        TOTAL_RAM=$(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)GB
        CPU_CORES=$(sysctl -n hw.ncpu)
    fi
    
    log_info "内存: $TOTAL_RAM"
    log_info "CPU核心: $CPU_CORES"
}

# 推荐部署方案
recommend_deployment_type() {
    log_header "分析部署方案"
    
    if [ "$DEPLOYMENT_TYPE" != "auto" ]; then
        log_info "使用指定的部署类型: $DEPLOYMENT_TYPE"
        return
    fi
    
    # 评分系统
    local docker_score=0
    local native_score=0
    
    # Docker可用性评分
    if [ "$DOCKER_INSTALLED" = "true" ]; then
        docker_score=$((docker_score + 30))
        
        # 检查Docker GPU支持
        if [ "$GPU_TYPE" = "nvidia" ] && docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
            docker_score=$((docker_score + 20))
        fi
    fi
    
    # Python环境评分
    if [ "$PYTHON_INSTALLED" = "true" ]; then
        native_score=$((native_score + 25))
        
        # 检查Python版本
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$python_version >= 3.8" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
            native_score=$((native_score + 15))
        fi
    fi
    
    # 系统稳定性评分
    if [[ "$DISTRO" =~ Ubuntu|Debian ]]; then
        docker_score=$((docker_score + 10))
        native_score=$((native_score + 15))
    elif [[ "$DISTRO" =~ CentOS|Red.*Hat|Rocky ]]; then
        docker_score=$((docker_score + 8))
        native_score=$((native_score + 12))
    elif [[ "$OS" == "macos" ]]; then
        native_score=$((native_score + 20))
        docker_score=$((docker_score + 5))
    fi
    
    # 资源评分
    ram_gb=$(echo "$TOTAL_RAM" | sed 's/[^0-9.]//g')
    if [[ $(echo "$ram_gb >= 16" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        docker_score=$((docker_score + 10))
        native_score=$((native_score + 5))
    fi
    
    # 决定部署类型
    if [ $docker_score -gt $native_score ] && [ $docker_score -gt 40 ]; then
        DEPLOYMENT_TYPE="docker"
        log_info "推荐部署方案: Docker (评分: $docker_score vs $native_score)"
        log_info "优势: 环境隔离、依赖管理简单、可移植性强"
    elif [ $native_score -gt 35 ]; then
        DEPLOYMENT_TYPE="native"
        log_info "推荐部署方案: 原生环境 (评分: $native_score vs $docker_score)"
        log_info "优势: 性能更好、资源使用更高效"
    else
        DEPLOYMENT_TYPE="hybrid"
        log_info "推荐部署方案: 混合部署 (评分接近: Docker=$docker_score, Native=$native_score)"
        log_info "优势: 灵活性高、可根据需要选择运行方式"
    fi
    
    if [ "$INTERACTIVE_MODE" = "true" ]; then
        echo ""
        echo "可选部署方案:"
        echo "1. Docker部署    - 环境隔离，易于管理"
        echo "2. 原生环境部署  - 性能更好，直接使用系统资源"
        echo "3. 混合部署     - 同时支持Docker和原生环境"
        echo ""
        
        if confirm_action "是否使用推荐的 $DEPLOYMENT_TYPE 部署方案？" "y"; then
            log_info "将使用 $DEPLOYMENT_TYPE 部署方案"
        else
            echo "请选择部署方案:"
            echo "1) Docker"
            echo "2) Native"
            echo "3) Hybrid"
            read -p "请输入选择 (1-3): " choice
            
            case $choice in
                1) DEPLOYMENT_TYPE="docker" ;;
                2) DEPLOYMENT_TYPE="native" ;;
                3) DEPLOYMENT_TYPE="hybrid" ;;
                *) log_warn "无效选择，使用推荐方案: $DEPLOYMENT_TYPE" ;;
            esac
        fi
    fi
}

# 安装系统依赖
install_system_dependencies() {
    if [ "$SKIP_GPU_DRIVERS" = "true" ]; then
        log_info "跳过系统依赖安装"
        return
    fi
    
    log_header "安装系统依赖"
    
    if [ ! -f "$PROJECT_DIR/scripts/setup_gpu_drivers.sh" ]; then
        log_error "GPU驱动安装脚本不存在"
        return 1
    fi
    
    log_step "检查GPU驱动状态..."
    
    # 检查是否需要安装GPU驱动
    local need_gpu_install="false"
    
    if [ "$GPU_TYPE" = "nvidia" ]; then
        if ! command -v nvidia-smi > /dev/null || ! nvidia-smi > /dev/null 2>&1; then
            need_gpu_install="true"
        fi
    elif [ "$GPU_TYPE" = "amd" ]; then
        if ! command -v rocm-smi > /dev/null; then
            need_gpu_install="true"
        fi
    fi
    
    if [ "$need_gpu_install" = "true" ] || [ "$FORCE_REINSTALL" = "true" ]; then
        if confirm_action "是否安装GPU驱动和系统依赖？" "y"; then
            log_step "运行GPU驱动安装脚本..."
            
            local gpu_args=""
            if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
                gpu_args="--install-docker"
            fi
            
            if ! sudo "$PROJECT_DIR/scripts/setup_gpu_drivers.sh" $gpu_args; then
                log_warn "GPU驱动安装可能失败，但继续部署过程"
            else
                log_success "GPU驱动安装完成"
            fi
        fi
    else
        log_success "GPU驱动已安装，跳过安装步骤"
    fi
}

# 设置Python环境
setup_python_environment() {
    if [ "$SKIP_PYTHON_ENV" = "true" ]; then
        log_info "跳过Python环境设置"
        return
    fi
    
    log_header "设置Python环境"
    
    if [ ! -f "$PROJECT_DIR/scripts/setup_python_env.sh" ]; then
        log_error "Python环境安装脚本不存在"
        return 1
    fi
    
    log_step "配置Python虚拟环境..."
    
    # 检查是否需要设置Python环境
    local need_python_setup="false"
    
    if [ ! -d "$PROJECT_DIR/strato_peft_env" ] || [ "$FORCE_REINSTALL" = "true" ]; then
        need_python_setup="true"
    fi
    
    if [ "$need_python_setup" = "true" ]; then
        if confirm_action "是否设置Python虚拟环境？" "y"; then
            log_step "运行Python环境安装脚本..."
            
            local python_args=""
            if [ "$GPU_TYPE" != "none" ]; then
                python_args="--gpu-type $GPU_TYPE"
            fi
            
            if "$PROJECT_DIR/scripts/setup_python_env.sh" $python_args; then
                log_success "Python环境设置完成"
            else
                log_error "Python环境设置失败"
                return 1
            fi
        fi
    else
        log_success "Python环境已存在，跳过设置步骤"
    fi
}

# 设置Docker环境
setup_docker_environment() {
    if [ "$SKIP_DOCKER" = "true" ] || [ "$DEPLOYMENT_TYPE" = "native" ]; then
        log_info "跳过Docker环境设置"
        return
    fi
    
    log_header "设置Docker环境"
    
    if [ ! -f "$PROJECT_DIR/scripts/deploy_docker.sh" ]; then
        log_error "Docker部署脚本不存在"
        return 1
    fi
    
    log_step "配置Docker部署环境..."
    
    # 创建项目输出目录结构
    log_step "创建项目目录结构..."
    local project_dirs=(
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
    
    cd "$PROJECT_DIR"
    for dir in "${project_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
    
    # 设置目录权限，确保Docker容器可以写入
    chmod -R 755 results logs cache data models experiments notebooks 2>/dev/null || true
    
    # 创建.gitkeep文件保持目录结构
    for dir in "${project_dirs[@]}"; do
        if [ ! -f "$dir/.gitkeep" ] && [ -d "$dir" ]; then
            echo "# 保持目录结构，用于Docker挂载" > "$dir/.gitkeep"
        fi
    done
    
    log_success "项目目录结构创建完成"
    
    if [ "$DOCKER_INSTALLED" = "false" ]; then
        log_warn "Docker未安装，某些功能可能不可用"
        if confirm_action "是否现在安装Docker？" "y"; then
            install_docker
        fi
    fi
    
    # 构建Docker镜像
    if [ "$DOCKER_INSTALLED" = "true" ]; then
        log_step "构建Docker镜像..."
        
        local docker_platform="cpu"
        if [ "$GPU_TYPE" = "nvidia" ]; then
            docker_platform="cuda"
        elif [ "$GPU_TYPE" = "amd" ]; then
            docker_platform="rocm"
        fi
        
        if "$PROJECT_DIR/scripts/deploy_docker.sh" \
           --platform "$docker_platform" \
           --build-env production \
           build; then
            log_success "Docker镜像构建完成"
        else
            log_warn "Docker镜像构建失败，但继续部署过程"
        fi
    fi
}

# 安装Docker
install_docker() {
    log_step "安装Docker..."
    
    case $PACKAGE_MANAGER in
        "apt")
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            ;;
        "yum"|"dnf")
            sudo $PACKAGE_MANAGER install -y docker
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        "brew")
            log_info "请手动安装Docker Desktop for Mac"
            open "https://docs.docker.com/desktop/mac/install/"
            ;;
    esac
    
    DOCKER_INSTALLED="true"
    log_success "Docker安装完成"
}

# 运行验证测试
run_verification_tests() {
    log_header "运行验证测试"
    
    local test_passed=0
    local test_total=0
    
    # Python环境测试
    if [ "$DEPLOYMENT_TYPE" = "native" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]; then
        log_step "测试Python环境..."
        ((test_total++))
        
        if [ -f "$PROJECT_DIR/activate_env.sh" ]; then
            cd "$PROJECT_DIR"
            if source activate_env.sh && python -c "import torch, transformers, peft; print('Python环境测试通过')"; then
                log_success "Python环境测试通过"
                ((test_passed++))
            else
                log_warn "Python环境测试失败"
            fi
        fi
    fi
    
    # Docker环境测试
    if [ "$DEPLOYMENT_TYPE" = "docker" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]; then
        log_step "测试Docker环境..."
        ((test_total++))
        
        if [ "$DOCKER_INSTALLED" = "true" ] && [ -f "$PROJECT_DIR/scripts/deploy_docker.sh" ]; then
            if "$PROJECT_DIR/scripts/deploy_docker.sh" health-check; then
                log_success "Docker环境测试通过"
                ((test_passed++))
            else
                log_warn "Docker环境测试失败"
            fi
        fi
    fi
    
    # GPU验证测试
    if [ "$GPU_TYPE" != "none" ]; then
        log_step "测试GPU环境..."
        ((test_total++))
        
        if [ -f "$PROJECT_DIR/scripts/verify_gpu.sh" ]; then
            if "$PROJECT_DIR/scripts/verify_gpu.sh" > /dev/null 2>&1; then
                log_success "GPU环境测试通过"
                ((test_passed++))
            else
                log_warn "GPU环境测试失败"
            fi
        fi
    fi
    
    # 烟雾测试
    log_step "运行烟雾测试..."
    ((test_total++))
    
    local smoke_test_passed="false"
    
    if [ "$DEPLOYMENT_TYPE" = "native" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]; then
        cd "$PROJECT_DIR"
        if source activate_env.sh && python main.py --config configs/gpt2_smoke_test.yaml --fast-dev-run --dry-run; then
            smoke_test_passed="true"
        fi
    fi
    
    if [ "$smoke_test_passed" = "false" ] && ([ "$DEPLOYMENT_TYPE" = "docker" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]); then
        if [ "$DOCKER_INSTALLED" = "true" ]; then
            if "$PROJECT_DIR/scripts/deploy_docker.sh" smoke-test; then
                smoke_test_passed="true"
            fi
        fi
    fi
    
    if [ "$smoke_test_passed" = "true" ]; then
        log_success "烟雾测试通过"
        ((test_passed++))
    else
        log_warn "烟雾测试失败"
    fi
    
    # 测试结果总结
    echo ""
    log_header "验证测试结果"
    log_info "通过测试: $test_passed/$test_total"
    
    if [ $test_passed -eq $test_total ]; then
        log_success "所有测试通过！部署成功！"
        return 0
    elif [ $test_passed -gt 0 ]; then
        log_warn "部分测试通过，部署基本成功"
        return 0
    else
        log_error "所有测试失败，部署可能存在问题"
        return 1
    fi
}

# 生成部署报告
generate_deployment_report() {
    log_header "生成部署报告"
    
    local report_file="$PROJECT_DIR/DEPLOYMENT_REPORT.md"
    
    cat > "$report_file" << EOF
# STRATO-PEFT 部署报告

## 部署信息
- **部署时间**: $(date)
- **部署类型**: $DEPLOYMENT_TYPE
- **操作系统**: $DISTRO $VERSION
- **GPU类型**: $GPU_TYPE
- **内存**: $TOTAL_RAM
- **CPU核心**: $CPU_CORES

## 环境状态
- **Docker**: ${DOCKER_INSTALLED}
- **Python**: ${PYTHON_INSTALLED}
- **Git**: ${GIT_INSTALLED}

## 部署组件
EOF

    if [ "$DEPLOYMENT_TYPE" = "native" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]; then
        cat >> "$report_file" << EOF

### Python环境
- **虚拟环境**: $PROJECT_DIR/strato_peft_env
- **激活脚本**: $PROJECT_DIR/activate_env.sh
- **启动命令**: 
  \`\`\`bash
  cd $PROJECT_DIR
  source activate_env.sh
  python main.py --config configs/gpt2_smoke_test.yaml
  \`\`\`
EOF
    fi

    if [ "$DEPLOYMENT_TYPE" = "docker" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]; then
        cat >> "$report_file" << EOF

### Docker环境
- **部署脚本**: $PROJECT_DIR/scripts/deploy_docker.sh
- **启动命令**:
  \`\`\`bash
  cd $PROJECT_DIR
  ./scripts/deploy_docker.sh --platform ${GPU_TYPE:-cpu}
  \`\`\`
- **访问地址**:
  - Jupyter Lab: http://localhost:8888
  - TensorBoard: http://localhost:6006
  - MLflow: http://localhost:5000
EOF
    fi

    cat >> "$report_file" << EOF

## 常用命令

### 运行训练
\`\`\`bash
# 原生环境
cd $PROJECT_DIR && source activate_env.sh
python main.py --config configs/your_config.yaml

# Docker环境
./scripts/deploy_docker.sh --profile training
\`\`\`

### 验证环境
\`\`\`bash
# GPU验证
./scripts/verify_gpu.sh

# 健康检查
./scripts/deploy_docker.sh health-check
\`\`\`

### 查看日志
\`\`\`bash
# 训练日志
tail -f results/*/logs/training.log

# Docker日志
docker-compose logs -f
\`\`\`

## 故障排除

### 常见问题
1. **GPU不可用**: 检查驱动安装，运行 \`./scripts/verify_gpu.sh\`
2. **依赖包错误**: 重新运行 \`./scripts/setup_python_env.sh\`
3. **Docker权限问题**: 将用户添加到docker组: \`sudo usermod -aG docker \$USER\`

### 获取帮助
- 查看CLAUDE.md文档
- 运行 \`python main.py --help\`
- 检查 \`deployment.log\` 文件

---
**部署完成时间**: $(date)
EOF

    log_info "部署报告已保存到: $report_file"
}

# 显示完成信息
show_completion_info() {
    log_header "部署完成"
    
    echo ""
    echo -e "${GREEN}🎉 STRATO-PEFT 部署成功！🎉${NC}"
    echo ""
    
    case "$DEPLOYMENT_TYPE" in
        "native")
            echo -e "${CYAN}原生环境已配置完成${NC}"
            echo "启动命令:"
            echo "  cd $PROJECT_DIR"
            echo "  source activate_env.sh"
            echo "  python main.py --config configs/gpt2_smoke_test.yaml"
            ;;
        "docker")
            echo -e "${CYAN}Docker环境已配置完成${NC}"
            echo "启动命令:"
            echo "  cd $PROJECT_DIR"
            echo "  ./scripts/deploy_docker.sh --platform ${GPU_TYPE:-cpu}"
            echo ""
            echo "访问地址:"
            echo "  - Jupyter Lab: http://localhost:8888"
            echo "  - TensorBoard: http://localhost:6006"
            ;;
        "hybrid")
            echo -e "${CYAN}混合环境已配置完成${NC}"
            echo "原生环境启动:"
            echo "  cd $PROJECT_DIR && source activate_env.sh"
            echo "Docker环境启动:"
            echo "  ./scripts/deploy_docker.sh --platform ${GPU_TYPE:-cpu}"
            ;;
    esac
    
    echo ""
    echo -e "${YELLOW}重要文件:${NC}"
    echo "  📋 部署报告: $PROJECT_DIR/DEPLOYMENT_REPORT.md"
    echo "  📖 使用文档: $PROJECT_DIR/CLAUDE.md"
    echo "  📝 训练报告: $PROJECT_DIR/TRAINING_REPORTS_GUIDE.md"
    
    if [ -n "$DEPLOYMENT_LOG" ]; then
        echo "  📄 部署日志: $DEPLOYMENT_LOG"
    fi
    
    echo ""
    echo -e "${PURPLE}下一步操作:${NC}"
    echo "1. 查看部署报告了解详细信息"
    echo "2. 运行烟雾测试验证环境: python main.py --config configs/gpt2_smoke_test.yaml --fast-dev-run"
    echo "3. 开始您的PEFT实验！"
    echo ""
}

# 主函数
main() {
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            --skip-gpu-drivers)
                SKIP_GPU_DRIVERS="true"
                shift
                ;;
            --skip-python-env)
                SKIP_PYTHON_ENV="true"
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER="true"
                shift
                ;;
            --force-reinstall)
                FORCE_REINSTALL="true"
                shift
                ;;
            --non-interactive)
                INTERACTIVE_MODE="false"
                shift
                ;;
            --log-file)
                DEPLOYMENT_LOG="$2"
                shift 2
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
    
    # 初始化日志
    if [ -n "$DEPLOYMENT_LOG" ]; then
        touch "$DEPLOYMENT_LOG"
        log_info "部署日志: $DEPLOYMENT_LOG"
    fi
    
    # 显示欢迎信息
    show_welcome
    
    # 执行部署流程
    detect_system_environment
    recommend_deployment_type
    
    # 确认开始部署
    if [ "$INTERACTIVE_MODE" = "true" ]; then
        if ! confirm_action "开始部署STRATO-PEFT？" "y"; then
            log_info "部署已取消"
            exit 0
        fi
    fi
    
    # 执行部署步骤
    install_system_dependencies
    
    if [ "$DEPLOYMENT_TYPE" = "native" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]; then
        setup_python_environment
    fi
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ] || [ "$DEPLOYMENT_TYPE" = "hybrid" ]; then
        setup_docker_environment
    fi
    
    # 验证部署
    if run_verification_tests; then
        generate_deployment_report
        show_completion_info
        exit 0
    else
        log_error "部署验证失败"
        generate_deployment_report
        exit 1
    fi
}

# 错误处理
trap 'log_error "部署过程中发生错误，请检查日志"; exit 1' ERR

# 信号处理
trap 'log_info "收到中断信号，正在清理..."; exit 0' SIGTERM SIGINT

# 运行主函数
main "$@"