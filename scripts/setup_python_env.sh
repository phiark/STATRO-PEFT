#!/bin/bash
# STRATO-PEFT Python环境自动安装脚本
# 支持多种Python版本、虚拟环境管理和依赖优化

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
PYTHON_VERSION="3.9"
VENV_NAME="strato_peft_env"
REQUIREMENTS_FILE="requirements.txt"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get > /dev/null; then
            PACKAGE_MANAGER="apt"
        elif command -v yum > /dev/null; then
            PACKAGE_MANAGER="yum"
        elif command -v dnf > /dev/null; then
            PACKAGE_MANAGER="dnf"
        else
            log_error "不支持的Linux发行版"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PACKAGE_MANAGER="brew"
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    log_info "检测到操作系统: $OS"
}

# 检测GPU类型
detect_gpu() {
    GPU_TYPE="cpu"
    
    if command -v nvidia-smi > /dev/null; then
        GPU_TYPE="cuda"
        log_info "检测到NVIDIA GPU，将安装CUDA版本的PyTorch"
    elif command -v rocm-smi > /dev/null; then
        GPU_TYPE="rocm"
        log_info "检测到AMD GPU，将安装ROCm版本的PyTorch"
    elif [[ "$OS" == "macos" ]] && sysctl -n machdep.cpu.brand_string | grep -q "Apple"; then
        GPU_TYPE="mps"
        log_info "检测到Apple Silicon，将安装MPS版本的PyTorch"
    else
        log_info "未检测到GPU或使用CPU模式"
    fi
}

# 安装系统依赖
install_system_dependencies() {
    log_header "安装系统依赖"
    
    case $PACKAGE_MANAGER in
        "apt")
            sudo apt-get update -y
            sudo apt-get install -y \
                python3-dev \
                python3-pip \
                python3-venv \
                build-essential \
                cmake \
                git \
                curl \
                wget \
                unzip \
                libssl-dev \
                libffi-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                llvm \
                libncurses5-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev \
                libxml2-dev \
                libxmlsec1-dev \
                libhdf5-dev \
                pkg-config
            ;;
        "yum"|"dnf")
            sudo $PACKAGE_MANAGER install -y \
                python3-devel \
                python3-pip \
                gcc \
                gcc-c++ \
                cmake \
                git \
                curl \
                wget \
                unzip \
                openssl-devel \
                libffi-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                llvm-devel \
                ncurses-devel \
                xz-devel \
                tk-devel \
                libxml2-devel \
                xmlsec1-devel \
                hdf5-devel \
                pkgconfig
            ;;
        "brew")
            # 检查是否安装了Homebrew
            if ! command -v brew > /dev/null; then
                log_info "安装Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install \
                python@3.9 \
                cmake \
                git \
                curl \
                wget \
                openssl \
                libffi \
                readline \
                sqlite \
                xz \
                tk \
                libxml2 \
                xmlsec1 \
                hdf5 \
                pkg-config
                
            # 确保Python 3.9可用
            if ! command -v python3.9 > /dev/null; then
                ln -sf /opt/homebrew/bin/python3.9 /usr/local/bin/python3.9 2>/dev/null || true
            fi
            ;;
    esac
    
    log_info "系统依赖安装完成"
}

# 检查并安装Python
check_python() {
    log_header "检查Python版本"
    
    # 检查Python版本
    if command -v python3 > /dev/null; then
        CURRENT_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "当前Python版本: $CURRENT_VERSION"
        
        # 检查版本是否满足要求
        if [[ $(echo "$CURRENT_VERSION >= $PYTHON_VERSION" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
            PYTHON_CMD="python3"
        else
            log_warn "Python版本过低，尝试安装Python $PYTHON_VERSION"
            install_python
        fi
    else
        log_warn "未找到Python3，开始安装"
        install_python
    fi
    
    # 验证Python安装
    if ! command -v $PYTHON_CMD > /dev/null; then
        log_error "Python安装失败"
        exit 1
    fi
    
    log_info "使用Python: $($PYTHON_CMD --version)"
}

# 安装Python
install_python() {
    log_info "安装Python $PYTHON_VERSION..."
    
    case $OS in
        "linux")
            # 使用pyenv安装特定版本的Python
            if ! command -v pyenv > /dev/null; then
                log_info "安装pyenv..."
                curl https://pyenv.run | bash
                
                # 添加到PATH
                export PATH="$HOME/.pyenv/bin:$PATH"
                eval "$(pyenv init --path)"
                eval "$(pyenv init -)"
            fi
            
            # 安装Python
            pyenv install $PYTHON_VERSION
            pyenv global $PYTHON_VERSION
            PYTHON_CMD="python"
            ;;
        "macos")
            # 使用Homebrew安装
            if [[ "$PYTHON_VERSION" == "3.9" ]]; then
                PYTHON_CMD="python3.9"
            else
                log_warn "使用系统默认Python3"
                PYTHON_CMD="python3"
            fi
            ;;
    esac
}

# 创建虚拟环境
create_virtual_environment() {
    log_header "创建虚拟环境"
    
    cd "$PROJECT_DIR"
    
    # 检查是否已存在虚拟环境
    if [[ -d "$VENV_NAME" ]]; then
        log_warn "虚拟环境已存在，将重新创建"
        rm -rf "$VENV_NAME"
    fi
    
    # 创建新的虚拟环境
    $PYTHON_CMD -m venv "$VENV_NAME"
    
    # 激活虚拟环境
    source "$VENV_NAME/bin/activate"
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    
    log_info "虚拟环境创建完成: $PROJECT_DIR/$VENV_NAME"
}

# 生成requirements.txt
generate_requirements() {
    log_header "生成requirements.txt"
    
    cat > "$PROJECT_DIR/requirements.txt" << 'EOF'
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.12.0
accelerate>=0.20.0
peft>=0.4.0

# 配置管理
omegaconf>=2.3.0
hydra-core>=1.3.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
h5py>=3.8.0

# 可视化和监控
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
wandb>=0.15.0
tensorboard>=2.13.0

# 用户界面
rich>=13.0.0
click>=8.1.0
tqdm>=4.65.0

# 开发工具
pytest>=7.3.0
pytest-cov>=4.1.0
black>=23.3.0
flake8>=6.0.0
mypy>=1.3.0
pre-commit>=3.3.0

# 性能优化
faiss-cpu>=1.7.4
ninja>=1.11.0
flash-attn>=2.0.0; sys_platform != "darwin"

# 实用工具
requests>=2.31.0
aiohttp>=3.8.0
packaging>=23.1.0
psutil>=5.9.0
GPUtil>=1.4.0

# 特定平台依赖
# Intel Extension for PyTorch (仅Linux)
intel-extension-for-pytorch>=2.0.0; sys_platform == "linux"

# 额外的NLP工具
nltk>=3.8.0
spacy>=3.6.0
sentencepiece>=0.1.99
sacrebleu>=2.3.0
rouge-score>=0.1.2
evaluate>=0.4.0

# 数据加载和处理
jsonlines>=3.1.0
pyarrow>=12.0.0
openpyxl>=3.1.0

# 调试和分析
line_profiler>=4.0.0
memory_profiler>=0.60.0
py-spy>=0.3.14

# 安全和验证
cryptography>=41.0.0
certifi>=2023.5.7
EOF

    log_info "requirements.txt已生成"
}

# 安装Python依赖
install_python_dependencies() {
    log_header "安装Python依赖"
    
    # 确保在虚拟环境中
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source "$PROJECT_DIR/$VENV_NAME/bin/activate"
    fi
    
    # 生成requirements.txt如果不存在
    if [[ ! -f "$PROJECT_DIR/$REQUIREMENTS_FILE" ]]; then
        generate_requirements
    fi
    
    # 根据GPU类型安装PyTorch
    install_pytorch
    
    # 安装其他依赖
    log_info "安装项目依赖..."
    pip install -r "$PROJECT_DIR/$REQUIREMENTS_FILE"
    
    # 安装项目本身 (如果有setup.py)
    if [[ -f "$PROJECT_DIR/setup.py" ]]; then
        pip install -e .
    fi
    
    log_info "Python依赖安装完成"
}

# 安装PyTorch
install_pytorch() {
    log_info "安装PyTorch (GPU类型: $GPU_TYPE)..."
    
    case $GPU_TYPE in
        "cuda")
            # 检测CUDA版本
            if command -v nvcc > /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
                log_info "检测到CUDA版本: $CUDA_VERSION"
                
                if [[ "$CUDA_VERSION" == "12."* ]]; then
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                elif [[ "$CUDA_VERSION" == "11."* ]]; then
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                else
                    log_warn "未知CUDA版本，安装默认CUDA版本"
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                fi
            else
                log_warn "未检测到CUDA编译器，安装默认CUDA版本"
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            fi
            ;;
        "rocm")
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
            ;;
        "mps")
            # Apple Silicon使用默认版本
            pip install torch torchvision torchaudio
            ;;
        "cpu")
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
}

# 安装开发工具
install_dev_tools() {
    log_header "安装开发工具"
    
    # 确保在虚拟环境中
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source "$PROJECT_DIR/$VENV_NAME/bin/activate"
    fi
    
    # 安装Jupyter
    pip install jupyter jupyterlab notebook ipywidgets
    
    # 安装代码质量工具
    pip install pre-commit
    
    # 如果存在.pre-commit-config.yaml，安装pre-commit钩子
    if [[ -f "$PROJECT_DIR/.pre-commit-config.yaml" ]]; then
        cd "$PROJECT_DIR"
        pre-commit install
    fi
    
    log_info "开发工具安装完成"
}

# 验证安装
verify_installation() {
    log_header "验证安装"
    
    # 确保在虚拟环境中
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source "$PROJECT_DIR/$VENV_NAME/bin/activate"
    fi
    
    # 创建验证脚本
    cat > /tmp/verify_installation.py << 'EOF'
import sys
import torch
import transformers
import peft
import omegaconf
import rich
import numpy as np
import pandas as pd

def verify_installation():
    print("=== STRATO-PEFT 环境验证 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Transformers版本: {transformers.__version__}")
    print(f"PEFT版本: {peft.__version__}")
    
    # 检查设备
    print(f"\n=== 设备信息 ===")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS可用: True")
    
    # 测试基本功能
    print(f"\n=== 功能测试 ===")
    try:
        # 创建一个简单的tensor
        x = torch.randn(2, 3)
        print(f"Tensor创建: ✓")
        
        # 测试设备移动
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        print(f"设备移动到 {device}: ✓")
        
        # 测试transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        print(f"Transformers加载: ✓")
        
        print(f"\n=== 验证结果 ===")
        print("✅ 所有组件验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
EOF
    
    # 运行验证
    if python /tmp/verify_installation.py; then
        log_info "✅ 环境验证通过"
        rm /tmp/verify_installation.py
        return 0
    else
        log_error "❌ 环境验证失败"
        rm /tmp/verify_installation.py
        return 1
    fi
}

# 创建激活脚本
create_activation_script() {
    log_header "创建激活脚本"
    
    cat > "$PROJECT_DIR/activate_env.sh" << EOF
#!/bin/bash
# STRATO-PEFT 环境激活脚本

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\${BLUE}激活STRATO-PEFT环境...\${NC}"

# 激活虚拟环境
source "$PROJECT_DIR/$VENV_NAME/bin/activate"

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR/src:\$PYTHONPATH"
export STRATO_PEFT_HOME="$PROJECT_DIR"

# 显示环境信息
echo -e "\${GREEN}✅ STRATO-PEFT环境已激活\${NC}"
echo "Python: \$(python --version)"
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: \$VIRTUAL_ENV"

# 提供有用的命令
echo ""
echo "常用命令:"
echo "  python main.py --config configs/gpt2_smoke_test.yaml  # 运行烟雾测试"
echo "  jupyter lab                                           # 启动Jupyter Lab"
echo "  python -m pytest tests/                             # 运行测试"
echo "  deactivate                                           # 退出环境"
EOF
    
    chmod +x "$PROJECT_DIR/activate_env.sh"
    log_info "激活脚本已创建: $PROJECT_DIR/activate_env.sh"
}

# 生成部署报告
generate_deployment_report() {
    log_header "生成部署报告"
    
    REPORT_FILE="$PROJECT_DIR/deployment_report.txt"
    
    cat > "$REPORT_FILE" << EOF
STRATO-PEFT Python环境部署报告
===============================

部署时间: $(date)
操作系统: $OS
Python版本: $($PYTHON_CMD --version)
GPU类型: $GPU_TYPE
项目目录: $PROJECT_DIR
虚拟环境: $PROJECT_DIR/$VENV_NAME

安装的主要组件:
- PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "安装中...")
- Transformers: $(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "安装中...")
- PEFT: $(python -c "import peft; print(peft.__version__)" 2>/dev/null || echo "安装中...")

使用说明:
1. 激活环境: source activate_env.sh
2. 运行测试: python main.py --config configs/gpt2_smoke_test.yaml
3. 查看GPU状态: ./scripts/verify_gpu.sh
4. 退出环境: deactivate

目录结构:
$PROJECT_DIR/
├── $VENV_NAME/                 # Python虚拟环境
├── activate_env.sh             # 环境激活脚本
├── requirements.txt            # Python依赖列表
├── main.py                     # 主程序入口
├── configs/                    # 配置文件
├── scripts/                    # 部署脚本
└── src/                        # 源代码

注意事项:
- 每次使用前需要激活虚拟环境
- GPU驱动需要单独安装 (运行 sudo ./scripts/setup_gpu_drivers.sh)
- 建议定期更新依赖包
EOF
    
    log_info "部署报告已保存到: $REPORT_FILE"
    cat "$REPORT_FILE"
}

# 主函数
main() {
    log_header "STRATO-PEFT Python环境安装脚本"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --venv-name)
                VENV_NAME="$2"
                shift 2
                ;;
            --gpu-type)
                GPU_TYPE="$2"
                shift 2
                ;;
            --no-dev-tools)
                SKIP_DEV_TOOLS="yes"
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --python-version VERSION  指定Python版本 (默认: 3.9)"
                echo "  --venv-name NAME          指定虚拟环境名称 (默认: strato_peft_env)"
                echo "  --gpu-type TYPE           强制指定GPU类型 (cuda/rocm/mps/cpu)"
                echo "  --no-dev-tools           跳过开发工具安装"
                echo "  --help                   显示此帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 开始安装流程
    detect_os
    
    # 如果未强制指定GPU类型，则自动检测
    if [[ -z "$GPU_TYPE" ]]; then
        detect_gpu
    fi
    
    install_system_dependencies
    check_python
    create_virtual_environment
    install_python_dependencies
    
    # 安装开发工具（如果需要）
    if [[ "$SKIP_DEV_TOOLS" != "yes" ]]; then
        install_dev_tools
    fi
    
    # 验证安装
    if verify_installation; then
        create_activation_script
        generate_deployment_report
        
        log_header "安装完成"
        log_info "✅ STRATO-PEFT Python环境安装成功！"
        log_info "运行以下命令激活环境："
        log_info "  source activate_env.sh"
        log_info "然后运行烟雾测试："
        log_info "  python main.py --config configs/gpt2_smoke_test.yaml"
    else
        log_error "❌ 安装验证失败，请检查错误信息"
        exit 1
    fi
}

# 运行主函数
main "$@"