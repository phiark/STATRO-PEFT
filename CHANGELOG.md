# Changelog

本文档记录了 STRATO-PEFT 项目的所有重要更改。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中的功能
- 多 GPU 分布式训练支持
- 更多 PEFT 方法集成
- 自动超参数调优
- 模型压缩和量化支持

## [0.1.0] - 2024-01-XX

### 新增
- 🎉 初始版本发布
- ✨ STRATO-PEFT 核心算法实现
  - 动态秩调度机制
  - 成本感知优化
  - Go-Explore 缓存机制
- 🚀 多平台支持
  - CUDA GPU 加速
  - ROCm GPU 支持
  - Apple Silicon (MPS) 优化
  - CPU 后备支持
- 🛠️ 完整的实验框架
  - 配置管理系统
  - 实验跟踪和监控
  - 结果分析工具
- 📦 Docker 容器化支持
  - 多平台 Docker 镜像
  - Docker Compose 配置
  - 开发环境容器
- 📊 集成监控工具
  - Weights & Biases 集成
  - TensorBoard 支持
  - MLflow 实验跟踪
- 🧪 测试和质量保证
  - 单元测试覆盖
  - 集成测试
  - 代码质量检查
- 📚 完整文档
  - 安装和使用指南
  - API 文档
  - 贡献指南

### 技术特性
- **核心算法**
  - STRATO-PEFT 实现
  - 动态适应性调整
  - 内存效率优化
- **模型支持**
  - Transformer 架构
  - 大语言模型 (LLM)
  - 视觉 Transformer (ViT)
- **训练优化**
  - 混合精度训练
  - 梯度累积
  - 学习率调度
- **数据处理**
  - 多格式数据加载
  - 数据预处理管道
  - 批处理优化

### 依赖项
- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- PEFT >= 0.4.0
- 其他核心依赖见 `requirements.txt`

### 已知问题
- 在某些 ROCm 版本上可能存在内存泄漏
- Windows 平台的 Docker 支持仍在测试中
- 大模型训练时的内存使用优化仍在改进

---

## 版本说明

### 版本号格式

本项目使用语义化版本号 `MAJOR.MINOR.PATCH`：

- **MAJOR**: 不兼容的 API 更改
- **MINOR**: 向后兼容的功能添加
- **PATCH**: 向后兼容的问题修复

### 更改类型

- `新增` - 新功能
- `更改` - 现有功能的更改
- `弃用` - 即将移除的功能
- `移除` - 已移除的功能
- `修复` - 问题修复
- `安全` - 安全相关的修复

### 发布周期

- **主要版本**: 每 6-12 个月
- **次要版本**: 每 1-3 个月
- **补丁版本**: 根据需要随时发布

### 支持政策

- 最新的主要版本：完全支持
- 前一个主要版本：安全更新和关键修复
- 更早版本：不再支持

---

## 贡献

如果您想为 STRATO-PEFT 做出贡献，请查看我们的 [贡献指南](CONTRIBUTING.md)。

## 反馈

如果您发现任何问题或有功能建议，请在 [GitHub Issues](https://github.com/your-org/strato-peft/issues) 中报告。