#!/bin/bash

# STRATO-PEFT Docker Runner Script
# Automatically detects platform and runs appropriate container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
PLATFORM="auto"
COMMAND="bash"
DETACH=false
BUILD=false
CLEAN=false
DEV_MODE=false
GPU_COUNT="all"
MEMORY_LIMIT="16g"
CPU_LIMIT="8"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  -p, --platform PLATFORM    Platform to use (auto|cuda|rocm|cpu|dev)"
    echo "  -d, --detach               Run container in detached mode"
    echo "  -b, --build                Build image before running"
    echo "  -c, --clean                Clean up containers and images"
    echo "  --dev                      Run in development mode with Jupyter"
    echo "  --gpu-count COUNT          Number of GPUs to use (default: all)"
    echo "  --memory LIMIT             Memory limit (default: 16g)"
    echo "  --cpu-limit LIMIT          CPU limit (default: 8)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Commands:"
    echo "  bash                       Start interactive bash shell (default)"
    echo "  train CONFIG               Run training with config file"
    echo "  eval CONFIG                Run evaluation with config file"
    echo "  jupyter                    Start Jupyter Lab"
    echo "  tensorboard                Start TensorBoard"
    echo ""
    echo "Examples:"
    echo "  $0                         # Auto-detect platform and start bash"
    echo "  $0 --platform cuda train configs/llama2_7b_mmlu_strato.yaml"
    echo "  $0 --dev                   # Start development environment with Jupyter"
    echo "  $0 --clean                 # Clean up Docker resources"
}

# Function to detect platform
detect_platform() {
    echo -e "${BLUE}Detecting platform...${NC}"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo -e "${GREEN}NVIDIA GPU detected${NC}"
            echo "cuda"
            return
        fi
    fi
    
    # Check for ROCm
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            echo -e "${GREEN}AMD ROCm GPU detected${NC}"
            echo "rocm"
            return
        fi
    fi
    
    # Check for Apple Silicon
    if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        echo -e "${YELLOW}Apple Silicon detected (using CPU mode in Docker)${NC}"
        echo "cpu"
        return
    fi
    
    # Default to CPU
    echo -e "${YELLOW}No GPU detected, using CPU${NC}"
    echo "cpu"
}

# Function to build image
build_image() {
    local platform=$1
    echo -e "${BLUE}Building image for platform: ${platform}${NC}"
    
    cd "$PROJECT_DIR"
    
    case $platform in
        "cuda")
            docker-compose build strato-cuda
            ;;
        "rocm")
            docker-compose build strato-rocm
            ;;
        "cpu")
            docker-compose build strato-cpu
            ;;
        "dev")
            docker-compose build strato-dev
            ;;
        *)
            echo -e "${RED}Unknown platform: $platform${NC}"
            exit 1
            ;;
    esac
}

# Function to clean up
cleanup() {
    echo -e "${BLUE}Cleaning up Docker resources...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    # Remove images
    docker images | grep "strato-peft" | awk '{print $3}' | xargs -r docker rmi -f
    
    # Clean up volumes (optional)
    read -p "Remove data volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    echo -e "${GREEN}Cleanup completed${NC}"
}

# Function to run container
run_container() {
    local platform=$1
    local command=$2
    
    cd "$PROJECT_DIR"
    
    # Prepare docker-compose command
    local compose_cmd="docker-compose"
    local service_name
    local extra_args=""
    
    case $platform in
        "cuda")
            service_name="strato-cuda"
            compose_cmd="$compose_cmd --profile cuda"
            ;;
        "rocm")
            service_name="strato-rocm"
            compose_cmd="$compose_cmd --profile rocm"
            ;;
        "cpu")
            service_name="strato-cpu"
            compose_cmd="$compose_cmd --profile cpu"
            ;;
        "dev")
            service_name="strato-dev"
            compose_cmd="$compose_cmd --profile dev"
            DEV_MODE=true
            ;;
        *)
            echo -e "${RED}Unknown platform: $platform${NC}"
            exit 1
            ;;
    esac
    
    # Handle different commands
    case $command in
        "bash")
            command="bash"
            ;;
        "train"*)
            # Extract config file from command
            config_file=$(echo "$command" | cut -d' ' -f2-)
            command="python main.py --config $config_file"
            ;;
        "eval"*)
            # Extract config file from command
            config_file=$(echo "$command" | cut -d' ' -f2-)
            command="python scripts/eval.py --config $config_file"
            ;;
        "jupyter")
            if [[ "$DEV_MODE" != "true" ]]; then
                echo -e "${YELLOW}Switching to dev mode for Jupyter${NC}"
                service_name="strato-dev"
                compose_cmd="docker-compose --profile dev"
            fi
            command="jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
            ;;
        "tensorboard")
            command="tensorboard --logdir=./results --host=0.0.0.0 --port=6006"
            ;;
    esac
    
    echo -e "${BLUE}Starting container: $service_name${NC}"
    echo -e "${BLUE}Command: $command${NC}"
    
    if [[ "$DETACH" == "true" ]]; then
        $compose_cmd up -d $service_name
        if [[ "$command" != "bash" ]]; then
            docker-compose exec $service_name $command
        fi
    else
        if [[ "$DEV_MODE" == "true" ]] && [[ "$command" == *"jupyter"* ]]; then
            echo -e "${GREEN}Starting Jupyter Lab...${NC}"
            echo -e "${GREEN}Access at: http://localhost:8888${NC}"
            $compose_cmd up $service_name
        else
            $compose_cmd run --rm $service_name $command
        fi
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            PLATFORM="dev"
            shift
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --memory)
            MEMORY_LIMIT="$2"
            shift 2
            ;;
        --cpu-limit)
            CPU_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
        *)
            # Remaining arguments are the command
            COMMAND="$*"
            break
            ;;
    esac
done

# Main execution
echo -e "${GREEN}STRATO-PEFT Docker Runner${NC}"
echo -e "${GREEN}=========================${NC}"

# Handle cleanup
if [[ "$CLEAN" == "true" ]]; then
    cleanup
    exit 0
fi

# Auto-detect platform if needed
if [[ "$PLATFORM" == "auto" ]]; then
    PLATFORM=$(detect_platform)
fi

echo -e "${BLUE}Platform: $PLATFORM${NC}"
echo -e "${BLUE}Command: $COMMAND${NC}"

# Build image if requested
if [[ "$BUILD" == "true" ]]; then
    build_image "$PLATFORM"
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Run container
run_container "$PLATFORM" "$COMMAND"

echo -e "${GREEN}Done!${NC}"