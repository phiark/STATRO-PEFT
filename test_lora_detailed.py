#!/usr/bin/env python3
"""
详细的LoRA测试脚本 - 验证LoRA在GPT2上的具体应用

该脚本专门测试LoRA功能，包括：
1. 检查GPT2模块结构
2. 正确应用LoRA
3. 验证参数数量变化
4. 测试训练效果

Author: STRATO-PEFT Research Team
Date: 2024
"""

import sys
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_gpt2_structure():
    """分析GPT2模型结构"""
    logger.info("🔍 分析GPT2模型结构...")
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    logger.info("GPT2模型的所有模块:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            logger.info(f"  Linear层: {name} - {module}")
            if len(name.split('.')) <= 3:  # 只显示主要层级
                logger.info(f"    输入维度: {module.in_features}, 输出维度: {module.out_features}")
    
    return model

def test_lora_with_correct_modules():
    """使用正确的模块名称测试LoRA"""
    logger.info("🧪 使用正确的模块名称测试LoRA...")
    
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 导入LoRA相关类
    from src.peft.lora import LoRAPEFT
    
    # 创建正确的LoRA配置 - 使用GPT2实际的模块名
    config = OmegaConf.create({
        'rank': 8,
        'alpha': 16,
        'dropout': 0.1,
        'target_modules': ['transformer.h.0.attn.c_attn', 'transformer.h.0.attn.c_proj']  # GPT2的实际Conv1D模块名
    })
    
    logger.info(f"LoRA配置: {config}")
    
    # 应用LoRA
    original_params = sum(p.numel() for p in model.parameters())
    logger.info(f"原始模型参数数量: {original_params:,}")
    
    lora_adapter = LoRAPEFT(config=config, model=model)
    adapted_model = lora_adapter.apply_peft()
    
    # 检查结果
    trainable_params = lora_adapter.get_trainable_parameters()
    trainable_count = sum(p.numel() for p in trainable_params)
    
    logger.info(f"LoRA适配后:")
    logger.info(f"  可训练参数: {trainable_count:,}")
    logger.info(f"  可训练比例: {trainable_count/original_params*100:.4f}%")
    
    # 测试训练步骤
    text = "The future of artificial intelligence is"
    inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
    inputs["labels"] = inputs["input_ids"].clone()
    
    # 前向传播
    adapted_model.train()
    outputs = adapted_model(**inputs)
    loss = outputs.loss
    
    logger.info(f"前向传播成功，loss: {loss.item():.4f}")
    
    # 反向传播
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    grad_norms = []
    for param in trainable_params:
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    logger.info(f"LoRA参数梯度范数: {grad_norms}")
    
    optimizer.step()
    logger.info("✅ LoRA训练步骤成功")
    
    return adapted_model, lora_adapter

def test_lora_on_multiple_layers():
    """在多个层上测试LoRA"""
    logger.info("🧪 在多个层上测试LoRA...")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # 创建配置，针对多个transformer层
    config = OmegaConf.create({
        'rank': 4,
        'alpha': 8,
        'dropout': 0.0,
        'target_modules': [
            'transformer.h.0.attn.c_attn',
            'transformer.h.0.attn.c_proj', 
            'transformer.h.1.attn.c_attn',
            'transformer.h.1.attn.c_proj'
        ]
    })
    
    from src.peft.lora import LoRAPEFT
    
    lora_adapter = LoRAPEFT(config=config, model=model)
    adapted_model = lora_adapter.apply_peft()
    
    # 统计
    metrics = lora_adapter.get_peft_metrics()
    logger.info(f"多层LoRA应用结果:")
    logger.info(f"  可训练参数: {metrics.trainable_params:,}")
    logger.info(f"  可训练比例: {metrics.trainable_ratio*100:.4f}%")
    logger.info(f"  适配效率: {metrics.adaptation_efficiency:.4f}")
    
    return adapted_model, lora_adapter

def benchmark_lora_vs_full_training():
    """对比LoRA和全量训练的性能"""
    logger.info("🏁 对比LoRA和全量训练性能...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试数据
    text = "Machine learning is revolutionizing the way we understand"
    inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
    inputs["labels"] = inputs["input_ids"].clone()
    
    # 1. 全量训练
    logger.info("测试全量训练...")
    full_model = AutoModelForCausalLM.from_pretrained("gpt2")
    full_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    
    full_model.train()
    outputs = full_model(**inputs)
    full_loss = outputs.loss.item()
    logger.info(f"全量训练 - 参数: {full_params:,}, Loss: {full_loss:.4f}")
    
    # 2. LoRA训练
    logger.info("测试LoRA训练...")
    lora_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    config = OmegaConf.create({
        'rank': 8,
        'alpha': 16,
        'dropout': 0.1,
        'target_modules': ['transformer.h.0.attn.c_attn', 'transformer.h.0.attn.c_proj']
    })
    
    from src.peft.lora import LoRAPEFT
    lora_adapter = LoRAPEFT(config=config, model=lora_model)
    lora_model = lora_adapter.apply_peft()
    
    lora_params = sum(p.numel() for p in lora_adapter.get_trainable_parameters())
    
    lora_model.train()
    outputs = lora_model(**inputs)
    lora_loss = outputs.loss.item()
    logger.info(f"LoRA训练 - 参数: {lora_params:,}, Loss: {lora_loss:.4f}")
    
    # 效率对比
    param_reduction = (full_params - lora_params) / full_params * 100
    logger.info(f"参数减少: {param_reduction:.2f}%")
    logger.info(f"参数效率: {lora_params / full_params:.6f}")

def main():
    """主函数"""
    logger.info("🚀 开始详细LoRA测试...")
    
    try:
        # 1. 分析GPT2结构
        model = analyze_gpt2_structure()
        
        # 2. 测试正确的LoRA应用
        adapted_model, adapter = test_lora_with_correct_modules()
        
        # 3. 测试多层LoRA
        multi_model, multi_adapter = test_lora_on_multiple_layers()
        
        # 4. 性能对比
        benchmark_lora_vs_full_training()
        
        logger.info("🎉 所有LoRA测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()