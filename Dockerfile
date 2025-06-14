# STRATO-PEFT Docker优化配置
# 多阶段构建，支持多种GPU平台和优化部署

# ===============================
# 构建参数
# ===============================
ARG PLATFORM=cuda
ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=12.1
ARG UBUNTU_VERSION=22.04
ARG BUILD_ENV=production

# ===============================
# 基础镜像选择
# ===============================
FROM python:${PYTHON_VERSION}-slim-bullseye as base-cpu
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base-cuda
FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:5.7 as base-rocm

# 选择平台特定的基础镜像
FROM base-${PLATFORM} as base

# ===============================
# 系统依赖安装阶段
# ===============================
FROM base as system-deps

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（优化版本）
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python和基础开发工具
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    python3-venv \
    # 编译工具
    build-essential \
    cmake \
    ninja-build \
    gcc \
    g++ \
    gfortran \
    # 系统工具
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    pkg-config \
    # 科学计算依赖
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    # 性能工具
    htop \
    nvtop \
    iotop \
    # 清理缓存
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# 设置Python链接
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# 升级pip和安装核心包
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ===============================
# Python依赖预安装阶段
# ===============================
FROM system-deps as python-deps

# 工作目录
WORKDIR /tmp/install

# 创建优化的pip配置
RUN mkdir -p /root/.pip && \
    cat > /root/.pip/pip.conf << 'EOF'
[global]
no-cache-dir = true
disable-pip-version-check = true
timeout = 120
retries = 3
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
               download.pytorch.org
[install]
compile = false
EOF

# 复制requirements.txt（分层缓存优化）
COPY requirements.txt .

# 平台特定的PyTorch安装
ARG PLATFORM
RUN echo "Installing PyTorch for platform: $PLATFORM" && \
    if [ "$PLATFORM" = "cuda" ]; then \
        pip install --no-cache-dir torch==${PYTORCH_VERSION} torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$PLATFORM" = "rocm" ]; then \
        pip install --no-cache-dir torch==${PYTORCH_VERSION} torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/rocm5.6; \
    else \
        pip install --no-cache-dir torch==${PYTORCH_VERSION} torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# 安装其他Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 平台特定的监控工具
RUN if [ "$PLATFORM" = "cuda" ]; then \
        pip install --no-cache-dir nvidia-ml-py3 pynvml GPUtil; \
    elif [ "$PLATFORM" = "rocm" ]; then \
        pip install --no-cache-dir rocm-smi; \
    fi

# ===============================
# 应用程序构建阶段
# ===============================
FROM python-deps as app-build

# 设置工作目录
WORKDIR /app

# 复制项目文件（分层优化）
COPY setup.py setup.cfg pyproject.toml* ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY main.py ./
COPY README.md LICENSE* ./

# 设置脚本权限
RUN find scripts/ -name "*.sh" -exec chmod +x {} \;

# 安装项目本身
RUN if [ -f "setup.py" ]; then pip install --no-cache-dir -e .; fi

# ===============================
# 生产环境构建
# ===============================
FROM app-build as production

# 创建非root用户
RUN groupadd -r strato && \
    useradd -r -g strato -d /app -s /bin/bash strato && \
    chown -R strato:strato /app

# 设置环境变量
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV STRATO_PEFT_HOME=/app
ENV STRATO_PLATFORM=${PLATFORM}
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_VISIBLE_DEVICES=0

# 创建必要的目录
RUN mkdir -p /app/results /app/logs /app/cache /app/tmp && \
    chown -R strato:strato /app

# 复制Docker辅助脚本
COPY docker/ ./docker/
RUN chmod +x docker/*.sh docker/*.py

# 健康检查脚本
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python docker/healthcheck.py

# 暴露端口
EXPOSE 8888 6006 8080

# 设置用户
USER strato

# 工作目录
WORKDIR /app

# 入口点
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["python", "main.py", "--help"]

# ===============================
# 开发环境构建
# ===============================
FROM production as development

# 切换回root安装开发工具
USER root

# 安装开发和调试工具
RUN pip install --no-cache-dir \
    # Jupyter生态
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    jupyterlab-git \
    # 代码质量工具
    pre-commit \
    black \
    flake8 \
    mypy \
    pylint \
    # 测试工具
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    # 性能分析
    line_profiler \
    memory_profiler \
    py-spy \
    # 可视化工具
    tensorboard \
    wandb \
    # 调试工具
    ipdb \
    pdbpp

# 配置Jupyter Lab
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

# 安装JupyterLab扩展
RUN jupyter lab build

# 切换回普通用户
USER strato

# 开发环境默认启动Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ===============================
# 标签和元数据
# ===============================
LABEL maintainer="STRATO-PEFT Research Team" \
      version="1.0.0" \
      description="STRATO-PEFT Experimental Framework - Optimized Docker Image" \
      platform="${PLATFORM}" \
      python_version="${PYTHON_VERSION}" \
      pytorch_version="${PYTORCH_VERSION}" \
      build_env="${BUILD_ENV}" \
      build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      vcs_url="https://github.com/STRATO-PEFT/strato-peft" \
      documentation="https://github.com/STRATO-PEFT/strato-peft/blob/main/README.md"