# Contributing to STRATO-PEFT

我们欢迎并感谢所有形式的贡献！本文档提供了参与 STRATO-PEFT 项目开发的指南。

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [提交指南](#提交指南)
- [测试](#测试)
- [文档](#文档)
- [发布流程](#发布流程)

## 行为准则

参与本项目即表示您同意遵守我们的行为准则。我们致力于为所有人提供友好、安全和欢迎的环境。

### 我们的承诺

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 专注于对社区最有利的事情
- 对其他社区成员表现出同理心

## 如何贡献

### 报告 Bug

在报告 bug 之前，请检查是否已有相关的 issue。如果没有，请创建新的 issue 并包含：

- 清晰的标题和描述
- 重现步骤
- 预期行为和实际行为
- 环境信息（操作系统、Python 版本、依赖版本等）
- 相关的日志或错误信息

### 功能请求

我们欢迎新功能的建议！请创建 issue 并包含：

- 功能的详细描述
- 使用场景和动机
- 可能的实现方案
- 是否愿意实现该功能

### 代码贡献

1. **Fork 仓库**
2. **创建功能分支**：`git checkout -b feature/amazing-feature`
3. **进行更改**
4. **提交更改**：`git commit -m 'Add amazing feature'`
5. **推送到分支**：`git push origin feature/amazing-feature`
6. **创建 Pull Request**

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/strato-peft.git
cd strato-peft
```

### 2. 创建虚拟环境

```bash
# 使用 conda
conda create -n strato-peft-dev python=3.9
conda activate strato-peft-dev

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 以开发模式安装项目
pip install -e .
```

### 4. 设置 pre-commit hooks

```bash
pre-commit install
```

### 5. 验证安装

```bash
# 运行测试
pytest

# 检查代码格式
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/
```

## 代码规范

### Python 代码风格

我们使用以下工具确保代码质量：

- **Black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

### 代码格式化

```bash
# 格式化代码
black src/ tests/ scripts/
isort src/ tests/ scripts/

# 检查格式
black --check src/ tests/ scripts/
isort --check-only src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/
```

### 命名约定

- **文件名**: 使用小写字母和下划线 (`my_module.py`)
- **类名**: 使用 PascalCase (`MyClass`)
- **函数名**: 使用小写字母和下划线 (`my_function`)
- **常量**: 使用大写字母和下划线 (`MY_CONSTANT`)
- **私有成员**: 以单下划线开头 (`_private_method`)

### 文档字符串

使用 Google 风格的文档字符串：

```python
def my_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this exception is raised.
    """
    pass
```

### 类型注解

所有公共函数和方法都应该有类型注解：

```python
from typing import List, Optional, Dict, Any

def process_data(data: List[Dict[str, Any]], 
                 config: Optional[str] = None) -> Dict[str, float]:
    """Process input data according to configuration."""
    pass
```

## 提交指南

### 提交信息格式

使用以下格式编写提交信息：

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式化（不影响功能）
- `refactor`: 代码重构
- `test`: 添加或修改测试
- `chore`: 构建过程或辅助工具的变动

#### 示例

```
feat(peft): add STRATO-PEFT implementation

Implement the core STRATO-PEFT algorithm with:
- Dynamic rank scheduling
- Cost-aware optimization
- Go-Explore caching mechanism

Closes #123
```

### Pull Request 指南

1. **确保所有测试通过**
2. **更新相关文档**
3. **添加或更新测试用例**
4. **遵循代码规范**
5. **提供清晰的 PR 描述**

#### PR 模板

```markdown
## 描述
简要描述此 PR 的更改内容。

## 更改类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 重大更改
- [ ] 文档更新

## 测试
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试完成

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 自我审查了代码
- [ ] 添加了必要的注释
- [ ] 更新了相关文档
- [ ] 没有引入新的警告
```

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_models.py

# 运行特定测试
pytest tests/test_models.py::test_model_loading

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 测试分类

使用标记来分类测试：

```python
import pytest

@pytest.mark.unit
def test_unit_function():
    """单元测试"""
    pass

@pytest.mark.integration
def test_integration_workflow():
    """集成测试"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """耗时测试"""
    pass

@pytest.mark.gpu
def test_gpu_functionality():
    """需要 GPU 的测试"""
    pass
```

### 运行特定类型的测试

```bash
# 只运行单元测试
pytest -m unit

# 跳过慢速测试
pytest -m "not slow"

# 只运行 GPU 测试
pytest -m gpu
```

## 文档

### 文档结构

- **README.md**: 项目概述和快速开始
- **docs/**: 详细文档
- **CONTRIBUTING.md**: 贡献指南
- **CHANGELOG.md**: 版本更新日志

### 构建文档

```bash
# 安装文档依赖
pip install -r docs/requirements.txt

# 构建文档
cd docs/
make html

# 查看文档
open _build/html/index.html
```

### API 文档

使用 Sphinx 自动生成 API 文档：

```bash
# 生成 API 文档
sphinx-apidoc -o docs/api src/

# 构建文档
cd docs/
make html
```

## 发布流程

### 版本号规范

使用语义化版本号 (SemVer)：

- **MAJOR**: 不兼容的 API 更改
- **MINOR**: 向后兼容的功能添加
- **PATCH**: 向后兼容的 bug 修复

### 发布步骤

1. **更新版本号**
   ```bash
   # 更新 setup.py 和 pyproject.toml 中的版本号
   ```

2. **更新 CHANGELOG.md**
   ```bash
   # 添加新版本的更改记录
   ```

3. **创建发布标签**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

4. **构建和发布**
   ```bash
   # 构建包
   python -m build
   
   # 发布到 PyPI
   twine upload dist/*
   ```

## 获取帮助

如果您有任何问题或需要帮助，请：

1. 查看现有的 [Issues](https://github.com/your-org/strato-peft/issues)
2. 创建新的 Issue
3. 参与 [Discussions](https://github.com/your-org/strato-peft/discussions)
4. 联系维护者

## 致谢

感谢所有为 STRATO-PEFT 项目做出贡献的开发者！

---

通过参与此项目，您同意遵守我们的行为准则和贡献指南。