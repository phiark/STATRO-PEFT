#!/bin/bash
# GPU安装验证脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_header() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

# 检测GPU类型
detect_gpu() {
    if command -v nvidia-smi > /dev/null; then
        GPU_TYPE="nvidia"
    elif command -v rocm-smi > /dev/null; then
        GPU_TYPE="amd"
    elif lspci | grep -i intel.*graphics > /dev/null; then
        GPU_TYPE="intel"
    else
        GPU_TYPE="cpu"
    fi
}

# 验证NVIDIA GPU
verify_nvidia() {
    log_header "验证NVIDIA GPU安装"
    
    # 检查nvidia-smi
    if command -v nvidia-smi > /dev/null; then
        log_info "nvidia-smi 可用"
        nvidia-smi
        echo ""
    else
        log_error "nvidia-smi 不可用"
        return 1
    fi
    
    # 检查CUDA
    if command -v nvcc > /dev/null; then
        log_info "CUDA编译器可用"
        nvcc --version
        echo ""
    else
        log_warn "CUDA编译器不可用"
    fi
    
    # 检查PyTorch CUDA支持
    if command -v python3 > /dev/null; then
        log_info "测试PyTorch CUDA支持..."
        python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || log_warn "PyTorch未安装或CUDA支持有问题"
    fi
    
    # 检查Docker NVIDIA支持
    if command -v docker > /dev/null; then
        log_info "测试Docker NVIDIA支持..."
        if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
            log_info "Docker NVIDIA支持正常"
        else
            log_warn "Docker NVIDIA支持可能有问题"
        fi
    fi
}

# 验证AMD GPU
verify_amd() {
    log_header "验证AMD GPU安装"
    
    # 检查rocm-smi
    if command -v rocm-smi > /dev/null; then
        log_info "rocm-smi 可用"
        rocm-smi
        echo ""
    else
        log_error "rocm-smi 不可用"
        return 1
    fi
    
    # 检查ROCm信息
    if command -v rocminfo > /dev/null; then
        log_info "ROCm信息:"
        rocminfo | head -20
        echo ""
    fi
    
    # 检查PyTorch ROCm支持
    if command -v python3 > /dev/null; then
        log_info "测试PyTorch ROCm支持..."
        python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'ROCm可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || log_warn "PyTorch未安装或ROCm支持有问题"
    fi
}

# 验证Intel GPU
verify_intel() {
    log_header "验证Intel GPU安装"
    
    # 检查Intel GPU设备
    if lspci | grep -i intel.*graphics > /dev/null; then
        log_info "Intel GPU设备:"
        lspci | grep -i intel.*graphics
        echo ""
    else
        log_error "未检测到Intel GPU"
        return 1
    fi
    
    # 检查OpenCL支持
    if command -v clinfo > /dev/null; then
        log_info "OpenCL设备信息:"
        clinfo | head -30
        echo ""
    else
        log_warn "clinfo 不可用，安装: sudo apt install clinfo"
    fi
}

# 运行性能基准测试
run_benchmark() {
    if [[ "$1" == "--benchmark" ]]; then
        log_header "运行GPU性能基准测试"
        
        if command -v python3 > /dev/null; then
            cat > /tmp/gpu_benchmark.py << 'EOF'
import torch
import time
import sys

def benchmark_gpu():
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过GPU基准测试")
        return
    
    device = torch.cuda.current_device()
    print(f"使用GPU: {torch.cuda.get_device_name(device)}")
    
    # 矩阵乘法基准测试
    size = 4096
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    
    # 基准测试
    start_time = time.time()
    for _ in range(50):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    ops_per_sec = 50 / total_time
    gflops = (2 * size**3 * ops_per_sec) / 1e9
    
    print(f"矩阵乘法性能: {gflops:.2f} GFLOPS")
    print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

if __name__ == "__main__":
    benchmark_gpu()
EOF
            
            python3 /tmp/gpu_benchmark.py
            rm /tmp/gpu_benchmark.py
        fi
    fi
}

# 创建验证报告
create_verification_report() {
    log_header "生成验证报告"
    
    REPORT_FILE="/tmp/gpu_verification_report.txt"
    
    cat > $REPORT_FILE << EOF
STRATO-PEFT GPU验证报告
========================

验证时间: $(date)
GPU类型: $GPU_TYPE

系统信息:
- 操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
- 内核版本: $(uname -r)
- 内存: $(free -h | grep Mem | awk '{print $2}')

GPU状态:
EOF

    case $GPU_TYPE in
        "nvidia")
            echo "✓ NVIDIA GPU支持" >> $REPORT_FILE
            if command -v nvcc > /dev/null; then
                echo "✓ CUDA工具链" >> $REPORT_FILE
            else
                echo "✗ CUDA工具链" >> $REPORT_FILE
            fi
            ;;
        "amd")
            echo "✓ AMD GPU支持" >> $REPORT_FILE
            if command -v rocminfo > /dev/null; then
                echo "✓ ROCm工具链" >> $REPORT_FILE
            else
                echo "✗ ROCm工具链" >> $REPORT_FILE
            fi
            ;;
        "intel")
            echo "✓ Intel GPU支持" >> $REPORT_FILE
            ;;
        "cpu")
            echo "! 仅CPU模式" >> $REPORT_FILE
            ;;
    esac
    
    if command -v docker > /dev/null; then
        echo "✓ Docker可用" >> $REPORT_FILE
    else
        echo "✗ Docker不可用" >> $REPORT_FILE
    fi
    
    echo "" >> $REPORT_FILE
    echo "建议:" >> $REPORT_FILE
    
    if [[ "$GPU_TYPE" == "cpu" ]]; then
        echo "- 考虑安装GPU驱动以获得更好性能" >> $REPORT_FILE
    else
        echo "- GPU配置正常，可以开始训练" >> $REPORT_FILE
    fi
    
    log_info "验证报告已保存到: $REPORT_FILE"
    cat $REPORT_FILE
}

# 主函数
main() {
    log_header "STRATO-PEFT GPU验证脚本"
    
    detect_gpu
    log_info "检测到GPU类型: $GPU_TYPE"
    
    case $GPU_TYPE in
        "nvidia")
            verify_nvidia
            ;;
        "amd")
            verify_amd
            ;;
        "intel")
            verify_intel
            ;;
        "cpu")
            log_warn "未检测到GPU，将使用CPU模式"
            ;;
    esac
    
    # 运行基准测试 (如果指定)
    run_benchmark "$1"
    
    # 生成验证报告
    create_verification_report
    
    log_header "验证完成"
    
    if [[ "$GPU_TYPE" != "cpu" ]]; then
        log_info "GPU验证通过，可以继续安装Python环境"
        log_info "运行: ./scripts/setup_python_env.sh"
    else
        log_warn "未检测到GPU，建议先安装GPU驱动"
        log_info "运行: sudo ./scripts/setup_gpu_drivers.sh"
    fi
}

# 显示帮助
if [[ "$1" == "--help" ]]; then
    echo "GPU验证脚本"
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --benchmark    运行性能基准测试"
    echo "  --help        显示此帮助信息"
    exit 0
fi

main "$@"