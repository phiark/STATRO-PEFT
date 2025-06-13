# Multi-stage Dockerfile for STRATO-PEFT experimental framework
# Supports CUDA, ROCm, CPU, and Apple Silicon (via Docker Desktop)

# Build arguments for platform selection
ARG PLATFORM=cuda
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.1.0

# Base image selection based on platform
FROM python:${PYTHON_VERSION}-slim as base-cpu
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base-cuda
FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1 as base-rocm

# Select the appropriate base
FROM base-${PLATFORM} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python if not already present (for CUDA/ROCm images)
RUN if [ "$PLATFORM" != "cpu" ]; then \
        apt-get update && \
        apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-pip python${PYTHON_VERSION}-dev && \
        ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
        ln -sf /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies based on platform
RUN if [ "$PLATFORM" = "cuda" ]; then \
        pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$PLATFORM" = "rocm" ]; then \
        pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6; \
    else \
        pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install other requirements
RUN pip install -r requirements.txt

# Install additional platform-specific packages
RUN if [ "$PLATFORM" = "cuda" ]; then \
        pip install nvidia-ml-py3 pynvml; \
    elif [ "$PLATFORM" = "rocm" ]; then \
        pip install rocm-smi; \
    fi

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user for security
RUN groupadd -r strato && useradd -r -g strato -m -d /home/strato strato
RUN chown -R strato:strato /workspace
USER strato

# Set environment variables for different platforms
ENV STRATO_PLATFORM=${PLATFORM}
ENV CUDA_VISIBLE_DEVICES=0

# Platform-specific environment setup
RUN if [ "$PLATFORM" = "rocm" ]; then \
        echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc; \
    fi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"

# Default command
CMD ["python", "main.py", "--help"]

# Labels for metadata
LABEL maintainer="STRATO-PEFT Team"
LABEL description="STRATO-PEFT Experimental Framework"
LABEL version="1.0.0"
LABEL platform="${PLATFORM}"