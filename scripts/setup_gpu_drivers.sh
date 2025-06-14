#!/bin/bash
# STRATO-PEFT GPU驱动自动安装脚本
# 支持NVIDIA CUDA、AMD ROCm和Intel Arc GPU驱动安装

set -e  # 出错时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

# 检查是否以root权限运行
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warn "检测到root权限，继续安装..."
    else
        log_error "此脚本需要sudo权限，请使用 sudo $0"
        exit 1
    fi
}

# 检测操作系统
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        OS_VERSION=$VERSION_ID
        OS_CODENAME=$VERSION_CODENAME
    else
        log_error "无法检测操作系统"
        exit 1
    fi
    log_info "检测到操作系统: $OS $OS_VERSION"
}

# 检测GPU硬件
detect_gpu() {
    log_header "检测GPU硬件"
    
    # 检测NVIDIA GPU
    if lspci | grep -i nvidia > /dev/null; then
        NVIDIA_GPU=$(lspci | grep -i nvidia | head -1)
        log_info "检测到NVIDIA GPU: $NVIDIA_GPU"
        GPU_VENDOR="nvidia"
        return 0
    fi
    
    # 检测AMD GPU
    if lspci | grep -i amd.*vga > /dev/null || lspci | grep -i amd.*display > /dev/null; then
        AMD_GPU=$(lspci | grep -i amd | grep -E "(vga|display)" | head -1)
        log_info "检测到AMD GPU: $AMD_GPU"
        GPU_VENDOR="amd"
        return 0
    fi
    
    # 检测Intel GPU
    if lspci | grep -i intel.*graphics > /dev/null; then
        INTEL_GPU=$(lspci | grep -i intel.*graphics | head -1)
        log_info "检测到Intel GPU: $INTEL_GPU"
        GPU_VENDOR="intel"
        return 0
    fi
    
    log_warn "未检测到支持的GPU硬件，将安装CPU版本依赖"
    GPU_VENDOR="cpu"
}

# 更新系统包管理器
update_system() {
    log_header "更新系统包管理器"
    
    if command -v apt-get > /dev/null; then
        apt-get update -y
        apt-get upgrade -y
        PACKAGE_MANAGER="apt"
    elif command -v yum > /dev/null; then
        yum update -y
        PACKAGE_MANAGER="yum"
    elif command -v dnf > /dev/null; then
        dnf update -y
        PACKAGE_MANAGER="dnf"
    else
        log_error "不支持的包管理器"
        exit 1
    fi
    
    log_info "系统更新完成"
}

# 安装基础依赖
install_base_dependencies() {
    log_header "安装基础依赖"
    
    case $PACKAGE_MANAGER in
        "apt")
            apt-get install -y \
                build-essential \
                cmake \
                git \
                curl \
                wget \
                unzip \
                software-properties-common \
                apt-transport-https \
                ca-certificates \
                gnupg \
                lsb-release \
                dkms \
                linux-headers-$(uname -r)
            ;;
        "yum"|"dnf")
            $PACKAGE_MANAGER install -y \
                gcc \
                gcc-c++ \
                cmake \
                git \
                curl \
                wget \
                unzip \
                kernel-devel \
                kernel-headers \
                dkms
            ;;
    esac
    
    log_info "基础依赖安装完成"
}

# 安装NVIDIA CUDA驱动
install_nvidia_cuda() {
    log_header "安装NVIDIA CUDA驱动和工具"
    
    # 检测CUDA版本
    CUDA_VERSION="12.3"
    
    case $OS in
        "Ubuntu")
            # 添加NVIDIA官方仓库
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo $OS_VERSION | tr -d '.')/x86_64/cuda-keyring_1.0-1_all.deb
            dpkg -i cuda-keyring_1.0-1_all.deb
            apt-get update
            
            # 安装CUDA toolkit
            apt-get install -y cuda-toolkit-12-3
            
            # 安装NVIDIA驱动
            apt-get install -y nvidia-driver-535
            ;;
        "CentOS Linux"|"Red Hat Enterprise Linux"|"Rocky Linux")
            # 添加NVIDIA仓库
            $PACKAGE_MANAGER config-manager --add-repo \
                https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
            
            # 安装CUDA
            $PACKAGE_MANAGER install -y cuda-toolkit-12-3
            ;;
    esac
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/environment
    
    log_info "NVIDIA CUDA安装完成"
    log_warn "请重启系统以加载新的GPU驱动"
}

# 安装AMD ROCm
install_amd_rocm() {
    log_header "安装AMD ROCm"
    
    case $OS in
        "Ubuntu")
            # 添加ROCm仓库
            wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
            echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' > /etc/apt/sources.list.d/rocm.list
            apt-get update
            
            # 安装ROCm
            apt-get install -y rocm-dkms rocm-dev rocm-libs
            ;;
        "CentOS Linux"|"Red Hat Enterprise Linux")
            # 添加ROCm仓库
            cat > /etc/yum.repos.d/rocm.repo << 'EOF'
[rocm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/yum/5.7/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
            
            # 安装ROCm
            $PACKAGE_MANAGER install -y rocm-dkms rocm-dev rocm-libs
            ;;
    esac
    
    # 添加用户到video组
    usermod -a -G video $(logname) 2>/dev/null || true
    
    # 设置环境变量
    echo 'export PATH=/opt/rocm/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> /etc/environment
    
    log_info "AMD ROCm安装完成"
    log_warn "请重启系统以加载新的GPU驱动"
}

# 安装Intel GPU驱动
install_intel_gpu() {
    log_header "安装Intel GPU驱动"
    
    case $OS in
        "Ubuntu")
            # 安装Intel GPU驱动
            apt-get install -y \
                intel-opencl-icd \
                intel-level-zero-gpu \
                level-zero \
                intel-media-va-driver-non-free \
                libmfx1
            ;;
        "CentOS Linux"|"Red Hat Enterprise Linux")
            # 启用PowerTools仓库
            $PACKAGE_MANAGER config-manager --set-enabled powertools || true
            
            # 安装Intel GPU支持
            $PACKAGE_MANAGER install -y \
                intel-opencl \
                level-zero
            ;;
    esac
    
    log_info "Intel GPU驱动安装完成"
}

# 验证GPU安装
verify_gpu_installation() {
    log_header "验证GPU安装"
    
    case $GPU_VENDOR in
        "nvidia")
            if command -v nvidia-smi > /dev/null; then
                log_info "NVIDIA GPU检测成功:"
                nvidia-smi
            else
                log_error "NVIDIA GPU检测失败"
                return 1
            fi
            
            if command -v nvcc > /dev/null; then
                log_info "CUDA编译器检测成功:"
                nvcc --version
            else
                log_warn "CUDA编译器未检测到"
            fi
            ;;
        "amd")
            if command -v rocm-smi > /dev/null; then
                log_info "AMD GPU检测成功:"
                rocm-smi
            else
                log_error "AMD GPU检测失败"
                return 1
            fi
            ;;
        "intel")
            if command -v clinfo > /dev/null; then
                log_info "Intel GPU OpenCL检测:"
                clinfo | head -20
            else
                log_warn "Intel GPU检测工具未安装"
            fi
            ;;
    esac
    
    return 0
}

# 安装Docker (可选)
install_docker() {
    if [[ "$INSTALL_DOCKER" == "yes" ]]; then
        log_header "安装Docker"
        
        case $PACKAGE_MANAGER in
            "apt")
                # 添加Docker官方仓库
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
                echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
                apt-get update
                apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
                ;;
            "yum"|"dnf")
                $PACKAGE_MANAGER install -y docker docker-compose
                ;;
        esac
        
        # 启动Docker服务
        systemctl enable docker
        systemctl start docker
        
        # 安装NVIDIA Container Toolkit (如果是NVIDIA GPU)
        if [[ "$GPU_VENDOR" == "nvidia" ]]; then
            install_nvidia_container_toolkit
        fi
        
        log_info "Docker安装完成"
    fi
}

# 安装NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    log_info "安装NVIDIA Container Toolkit..."
    
    case $PACKAGE_MANAGER in
        "apt")
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list > /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt-get update
            apt-get install -y nvidia-container-toolkit
            ;;
        "yum"|"dnf")
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo > /etc/yum.repos.d/nvidia-container-toolkit.repo
            $PACKAGE_MANAGER install -y nvidia-container-toolkit
            ;;
    esac
    
    # 配置Docker使用NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    
    log_info "NVIDIA Container Toolkit安装完成"
}

# 创建安装报告
create_installation_report() {
    log_header "生成安装报告"
    
    REPORT_FILE="/tmp/gpu_installation_report.txt"
    
    cat > $REPORT_FILE << EOF
STRATO-PEFT GPU驱动安装报告
================================

安装时间: $(date)
操作系统: $OS $OS_VERSION
GPU类型: $GPU_VENDOR

已安装组件:
EOF
    
    case $GPU_VENDOR in
        "nvidia")
            echo "- NVIDIA GPU驱动" >> $REPORT_FILE
            echo "- CUDA Toolkit" >> $REPORT_FILE
            if command -v docker > /dev/null; then
                echo "- NVIDIA Container Toolkit" >> $REPORT_FILE
            fi
            ;;
        "amd")
            echo "- AMD ROCm" >> $REPORT_FILE
            ;;
        "intel")
            echo "- Intel GPU驱动" >> $REPORT_FILE
            echo "- OpenCL支持" >> $REPORT_FILE
            ;;
    esac
    
    if command -v docker > /dev/null; then
        echo "- Docker Engine" >> $REPORT_FILE
    fi
    
    echo "" >> $REPORT_FILE
    echo "下一步:" >> $REPORT_FILE
    echo "1. 重启系统以加载新驱动" >> $REPORT_FILE
    echo "2. 运行验证脚本: ./scripts/verify_gpu.sh" >> $REPORT_FILE
    echo "3. 安装Python依赖: ./scripts/setup_python_env.sh" >> $REPORT_FILE
    
    log_info "安装报告已保存到: $REPORT_FILE"
    cat $REPORT_FILE
}

# 主函数
main() {
    log_header "STRATO-PEFT GPU驱动安装脚本"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-docker)
                INSTALL_DOCKER="yes"
                shift
                ;;
            --force-gpu)
                FORCE_GPU="$2"
                shift 2
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --install-docker    同时安装Docker"
                echo "  --force-gpu TYPE    强制指定GPU类型 (nvidia/amd/intel/cpu)"
                echo "  --help             显示此帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 开始安装流程
    check_root
    detect_os
    
    if [[ -n "$FORCE_GPU" ]]; then
        GPU_VENDOR="$FORCE_GPU"
        log_info "强制使用GPU类型: $GPU_VENDOR"
    else
        detect_gpu
    fi
    
    update_system
    install_base_dependencies
    
    # 根据GPU类型安装对应驱动
    case $GPU_VENDOR in
        "nvidia")
            install_nvidia_cuda
            ;;
        "amd")
            install_amd_rocm
            ;;
        "intel")
            install_intel_gpu
            ;;
        "cpu")
            log_info "跳过GPU驱动安装，将使用CPU"
            ;;
    esac
    
    # 安装Docker (如果需要)
    install_docker
    
    # 验证安装
    if [[ "$GPU_VENDOR" != "cpu" ]]; then
        if verify_gpu_installation; then
            log_info "GPU安装验证通过"
        else
            log_warn "GPU安装验证失败，但安装可能仍然成功"
        fi
    fi
    
    # 生成报告
    create_installation_report
    
    log_header "安装完成"
    log_info "请重启系统，然后运行 ./scripts/verify_gpu.sh 验证安装"
}

# 运行主函数
main "$@"