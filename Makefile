# STRATO-PEFT Makefile
# 提供常用的开发、测试和部署命令

.PHONY: help install install-dev clean test test-unit test-integration test-gpu lint format type-check security docs docker build-docker run-docker compose-up compose-down pre-commit setup-dev

# 默认目标
.DEFAULT_GOAL := help

# 项目配置
PROJECT_NAME := strato-peft
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# 帮助信息
help: ## 显示帮助信息
	@echo "$(BLUE)STRATO-PEFT 开发工具$(NC)"
	@echo ""
	@echo "$(YELLOW)可用命令:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)示例:$(NC)"
	@echo "  make setup-dev     # 设置开发环境"
	@echo "  make test          # 运行所有测试"
	@echo "  make lint          # 代码检查"
	@echo "  make docker        # 构建 Docker 镜像"

# 安装和环境设置
install: ## 安装项目依赖
	@echo "$(BLUE)安装项目依赖...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✓ 依赖安装完成$(NC)"

install-dev: ## 安装开发依赖
	@echo "$(BLUE)安装开发依赖...$(NC)"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "$(GREEN)✓ 开发依赖安装完成$(NC)"

setup-dev: install-dev pre-commit ## 设置完整开发环境
	@echo "$(GREEN)✓ 开发环境设置完成$(NC)"

pre-commit: ## 安装 pre-commit hooks
	@echo "$(BLUE)安装 pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks 安装完成$(NC)"

# 清理
clean: ## 清理构建文件和缓存
	@echo "$(BLUE)清理构建文件...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "$(GREEN)✓ 清理完成$(NC)"

clean-all: clean ## 深度清理（包括虚拟环境）
	@echo "$(BLUE)深度清理...$(NC)"
	rm -rf venv/ .venv/ env/ .env/
	rm -rf .tox/ .nox/
	@echo "$(GREEN)✓ 深度清理完成$(NC)"

# 代码质量
lint: ## 运行代码检查
	@echo "$(BLUE)运行代码检查...$(NC)"
	flake8 src/ tests/ scripts/
	pylint src/ tests/ scripts/
	bandit -r src/
	@echo "$(GREEN)✓ 代码检查完成$(NC)"

format: ## 格式化代码
	@echo "$(BLUE)格式化代码...$(NC)"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place src/ tests/ scripts/
	@echo "$(GREEN)✓ 代码格式化完成$(NC)"

format-check: ## 检查代码格式
	@echo "$(BLUE)检查代码格式...$(NC)"
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/
	@echo "$(GREEN)✓ 代码格式检查完成$(NC)"

type-check: ## 类型检查
	@echo "$(BLUE)运行类型检查...$(NC)"
	mypy src/
	@echo "$(GREEN)✓ 类型检查完成$(NC)"

security: ## 安全检查
	@echo "$(BLUE)运行安全检查...$(NC)"
	bandit -r src/
	safety check
	@echo "$(GREEN)✓ 安全检查完成$(NC)"

check-all: format-check lint type-check security ## 运行所有代码质量检查
	@echo "$(GREEN)✓ 所有检查完成$(NC)"

# 测试
test: ## 运行所有测试
	@echo "$(BLUE)运行所有测试...$(NC)"
	pytest -v
	@echo "$(GREEN)✓ 测试完成$(NC)"

test-unit: ## 运行单元测试
	@echo "$(BLUE)运行单元测试...$(NC)"
	pytest -v -m unit
	@echo "$(GREEN)✓ 单元测试完成$(NC)"

test-integration: ## 运行集成测试
	@echo "$(BLUE)运行集成测试...$(NC)"
	pytest -v -m integration
	@echo "$(GREEN)✓ 集成测试完成$(NC)"

test-gpu: ## 运行 GPU 测试
	@echo "$(BLUE)运行 GPU 测试...$(NC)"
	pytest -v -m gpu
	@echo "$(GREEN)✓ GPU 测试完成$(NC)"

test-fast: ## 运行快速测试（跳过慢速测试）
	@echo "$(BLUE)运行快速测试...$(NC)"
	pytest -v -m "not slow"
	@echo "$(GREEN)✓ 快速测试完成$(NC)"

test-coverage: ## 运行测试并生成覆盖率报告
	@echo "$(BLUE)运行测试覆盖率分析...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ 覆盖率分析完成，查看 htmlcov/index.html$(NC)"

# 文档
docs: ## 构建文档
	@echo "$(BLUE)构建文档...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ 文档构建完成，查看 docs/_build/html/index.html$(NC)"

docs-serve: ## 启动文档服务器
	@echo "$(BLUE)启动文档服务器...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## 清理文档构建文件
	@echo "$(BLUE)清理文档构建文件...$(NC)"
	cd docs && make clean
	@echo "$(GREEN)✓ 文档清理完成$(NC)"

# Docker
build-docker: ## 构建 Docker 镜像
	@echo "$(BLUE)构建 Docker 镜像...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✓ Docker 镜像构建完成$(NC)"

build-docker-cuda: ## 构建 CUDA Docker 镜像
	@echo "$(BLUE)构建 CUDA Docker 镜像...$(NC)"
	$(DOCKER) build --build-arg PLATFORM=cuda -t $(PROJECT_NAME):cuda .
	@echo "$(GREEN)✓ CUDA Docker 镜像构建完成$(NC)"

build-docker-rocm: ## 构建 ROCm Docker 镜像
	@echo "$(BLUE)构建 ROCm Docker 镜像...$(NC)"
	$(DOCKER) build --build-arg PLATFORM=rocm -t $(PROJECT_NAME):rocm .
	@echo "$(GREEN)✓ ROCm Docker 镜像构建完成$(NC)"

run-docker: ## 运行 Docker 容器
	@echo "$(BLUE)运行 Docker 容器...$(NC)"
	$(DOCKER) run -it --rm -v $(PWD):/workspace $(PROJECT_NAME):latest

run-docker-cuda: ## 运行 CUDA Docker 容器
	@echo "$(BLUE)运行 CUDA Docker 容器...$(NC)"
	$(DOCKER) run -it --rm --gpus all -v $(PWD):/workspace $(PROJECT_NAME):cuda

run-docker-gpu: run-docker-cuda ## 运行 GPU Docker 容器（别名）

docker-shell: ## 进入 Docker 容器 shell
	@echo "$(BLUE)进入 Docker 容器...$(NC)"
	$(DOCKER) run -it --rm -v $(PWD):/workspace $(PROJECT_NAME):latest /bin/bash

# Docker Compose
compose-up: ## 启动 Docker Compose 服务
	@echo "$(BLUE)启动 Docker Compose 服务...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ 服务启动完成$(NC)"

compose-up-dev: ## 启动开发环境服务
	@echo "$(BLUE)启动开发环境服务...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up -d
	@echo "$(GREEN)✓ 开发环境启动完成$(NC)"

compose-down: ## 停止 Docker Compose 服务
	@echo "$(BLUE)停止 Docker Compose 服务...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ 服务停止完成$(NC)"

compose-logs: ## 查看 Docker Compose 日志
	$(DOCKER_COMPOSE) logs -f

compose-ps: ## 查看 Docker Compose 服务状态
	$(DOCKER_COMPOSE) ps

# 实验和训练
train: ## 运行训练（默认配置）
	@echo "$(BLUE)开始训练...$(NC)"
	$(PYTHON) main.py
	@echo "$(GREEN)✓ 训练完成$(NC)"

train-cuda: ## 在 CUDA 上运行训练
	@echo "$(BLUE)在 CUDA 上开始训练...$(NC)"
	$(PYTHON) main.py --platform cuda
	@echo "$(GREEN)✓ CUDA 训练完成$(NC)"

train-rocm: ## 在 ROCm 上运行训练
	@echo "$(BLUE)在 ROCm 上开始训练...$(NC)"
	$(PYTHON) main.py --platform rocm
	@echo "$(GREEN)✓ ROCm 训练完成$(NC)"

train-cpu: ## 在 CPU 上运行训练
	@echo "$(BLUE)在 CPU 上开始训练...$(NC)"
	$(PYTHON) main.py --platform cpu
	@echo "$(GREEN)✓ CPU 训练完成$(NC)"

train-debug: ## 运行调试模式训练
	@echo "$(BLUE)开始调试模式训练...$(NC)"
	$(PYTHON) main.py --debug
	@echo "$(GREEN)✓ 调试训练完成$(NC)"

train-dry-run: ## 运行 dry-run 模式
	@echo "$(BLUE)运行 dry-run 模式...$(NC)"
	$(PYTHON) main.py --dry-run
	@echo "$(GREEN)✓ Dry-run 完成$(NC)"

# 评估
eval: ## 运行评估
	@echo "$(BLUE)开始评估...$(NC)"
	$(PYTHON) main.py --mode eval
	@echo "$(GREEN)✓ 评估完成$(NC)"

compare: ## 运行模型比较
	@echo "$(BLUE)开始模型比较...$(NC)"
	$(PYTHON) main.py --mode compare
	@echo "$(GREEN)✓ 模型比较完成$(NC)"

# 工具
notebook: ## 启动 Jupyter Notebook
	@echo "$(BLUE)启动 Jupyter Notebook...$(NC)"
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

lab: ## 启动 JupyterLab
	@echo "$(BLUE)启动 JupyterLab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard: ## 启动 TensorBoard
	@echo "$(BLUE)启动 TensorBoard...$(NC)"
	tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

mlflow: ## 启动 MLflow UI
	@echo "$(BLUE)启动 MLflow UI...$(NC)"
	mlflow ui --host=0.0.0.0 --port=5000

# 发布
build: clean ## 构建分发包
	@echo "$(BLUE)构建分发包...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ 分发包构建完成$(NC)"

upload-test: build ## 上传到 TestPyPI
	@echo "$(BLUE)上传到 TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ 上传到 TestPyPI 完成$(NC)"

upload: build ## 上传到 PyPI
	@echo "$(YELLOW)警告: 即将上传到生产 PyPI$(NC)"
	@read -p "确认上传? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "$(BLUE)上传到 PyPI...$(NC)"
	twine upload dist/*
	@echo "$(GREEN)✓ 上传到 PyPI 完成$(NC)"

# 版本管理
version: ## 显示当前版本
	@echo "$(BLUE)当前版本:$(NC)"
	@$(PYTHON) -c "import src; print(src.__version__)"

bump-patch: ## 增加补丁版本号
	@echo "$(BLUE)增加补丁版本号...$(NC)"
	bump2version patch
	@echo "$(GREEN)✓ 补丁版本号已更新$(NC)"

bump-minor: ## 增加次版本号
	@echo "$(BLUE)增加次版本号...$(NC)"
	bump2version minor
	@echo "$(GREEN)✓ 次版本号已更新$(NC)"

bump-major: ## 增加主版本号
	@echo "$(BLUE)增加主版本号...$(NC)"
	bump2version major
	@echo "$(GREEN)✓ 主版本号已更新$(NC)"

# 监控和分析
profile: ## 运行性能分析
	@echo "$(BLUE)运行性能分析...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats main.py
	@echo "$(GREEN)✓ 性能分析完成，查看 profile.stats$(NC)"

memory-profile: ## 运行内存分析
	@echo "$(BLUE)运行内存分析...$(NC)"
	mprof run main.py
	mprof plot
	@echo "$(GREEN)✓ 内存分析完成$(NC)"

# 数据库管理（如果使用）
db-init: ## 初始化数据库
	@echo "$(BLUE)初始化数据库...$(NC)"
	# 添加数据库初始化命令
	@echo "$(GREEN)✓ 数据库初始化完成$(NC)"

db-migrate: ## 运行数据库迁移
	@echo "$(BLUE)运行数据库迁移...$(NC)"
	# 添加数据库迁移命令
	@echo "$(GREEN)✓ 数据库迁移完成$(NC)"

# 实用工具
check-deps: ## 检查依赖更新
	@echo "$(BLUE)检查依赖更新...$(NC)"
	pip list --outdated

update-deps: ## 更新依赖
	@echo "$(BLUE)更新依赖...$(NC)"
	pip-review --auto
	@echo "$(GREEN)✓ 依赖更新完成$(NC)"

check-security: ## 检查安全漏洞
	@echo "$(BLUE)检查安全漏洞...$(NC)"
	safety check
	pip-audit
	@echo "$(GREEN)✓ 安全检查完成$(NC)"

# 开发工作流
dev-setup: setup-dev ## 开发环境设置（别名）

dev-check: check-all test-fast ## 开发检查（快速）

dev-test: test-unit test-integration ## 开发测试

ci-check: check-all test-coverage ## CI 检查

release-check: ci-check docs build ## 发布前检查

# 清理和重置
reset-env: clean-all install-dev ## 重置开发环境
	@echo "$(GREEN)✓ 开发环境重置完成$(NC)"

# 信息显示
info: ## 显示项目信息
	@echo "$(BLUE)项目信息:$(NC)"
	@echo "  项目名称: $(PROJECT_NAME)"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Pip: $$($(PIP) --version)"
	@echo "  Docker: $$($(DOCKER) --version 2>/dev/null || echo '未安装')"
	@echo "  工作目录: $(PWD)"
	@echo "  Git 分支: $$(git branch --show-current 2>/dev/null || echo '未知')"
	@echo "  Git 提交: $$(git rev-parse --short HEAD 2>/dev/null || echo '未知')"

status: ## 显示项目状态
	@echo "$(BLUE)项目状态:$(NC)"
	@echo "  Git 状态:"
	@git status --porcelain 2>/dev/null || echo "    未在 Git 仓库中"
	@echo "  Docker 镜像:"
	@$(DOCKER) images $(PROJECT_NAME) 2>/dev/null || echo "    未找到 Docker 镜像"
	@echo "  运行中的容器:"
	@$(DOCKER_COMPOSE) ps 2>/dev/null || echo "    无运行中的容器"