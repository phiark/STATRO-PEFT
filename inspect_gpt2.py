#!/usr/bin/env python3
"""
检查GPT2模型结构的脚本
"""

import torch
from transformers import AutoModelForCausalLM

def inspect_gpt2_modules():
    """详细检查GPT2的模块结构"""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    print("🔍 GPT2模型的详细结构:")
    print("=" * 80)
    
    # 先显示所有模块
    print("所有模块:")
    all_modules = []
    for name, module in model.named_modules():
        all_modules.append((name, type(module).__name__))
    
    # 显示前20个模块来了解结构
    for i, (name, type_name) in enumerate(all_modules[:20]):
        print(f"  {name} -> {type_name}")
    if len(all_modules) > 20:
        print(f"  ... 还有 {len(all_modules) - 20} 个模块")
    
    print("\n" + "-" * 80)
    
    linear_modules = []
    conv1d_modules = []
    
    # 检查Linear和Conv1D层
    from transformers.pytorch_utils import Conv1D
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.append({
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None,
                'type': 'Linear'
            })
        elif isinstance(module, Conv1D):
            conv1d_modules.append({
                'name': name,
                'in_features': module.weight.shape[0],
                'out_features': module.weight.shape[1], 
                'bias': module.bias is not None,
                'type': 'Conv1D'
            })
    
    all_linear_like = linear_modules + conv1d_modules
    
    print(f"找到 {len(linear_modules)} 个Linear层 和 {len(conv1d_modules)} 个Conv1D层:")
    print("-" * 80)
    
    # 显示所有线性层类型的模块
    for i, mod in enumerate(all_linear_like):
        print(f"{i+1:2d}. {mod['name']} ({mod['type']})")
        print(f"    输入: {mod['in_features']}, 输出: {mod['out_features']}, bias: {mod['bias']}")
    
    print("\n" + "=" * 80)
    print("🎯 推荐的LoRA target_modules:")
    
    # 注意力模块
    attention_modules = [mod['name'] for mod in all_linear_like if 'attn' in mod['name']]
    print(f"注意力模块 ({len(attention_modules)}个):")
    for mod in attention_modules[:5]:  # 显示前5个
        print(f"  - {mod}")
    if len(attention_modules) > 5:
        print(f"  ... 还有 {len(attention_modules) - 5} 个")
    
    # MLP模块  
    mlp_modules = [mod['name'] for mod in all_linear_like if 'mlp' in mod['name']]
    print(f"\nMLP模块 ({len(mlp_modules)}个):")
    for mod in mlp_modules[:5]:  # 显示前5个
        print(f"  - {mod}")
    if len(mlp_modules) > 5:
        print(f"  ... 还有 {len(mlp_modules) - 5} 个")
    
    # 其他重要模块
    other_modules = [mod['name'] for mod in all_linear_like if 'attn' not in mod['name'] and 'mlp' not in mod['name']]
    print(f"\n其他线性模块 ({len(other_modules)}个):")
    for mod in other_modules:
        print(f"  - {mod}")
    
    print("\n" + "=" * 80)
    print("📝 建议的配置:")
    print("最小配置（仅第一层注意力）:")
    if attention_modules:
        print(f"  target_modules: ['{attention_modules[0]}']")
    
    print("\n标准配置（多层注意力）:")
    if len(attention_modules) >= 4:
        print("  target_modules: [")
        for mod in attention_modules[:4]:
            print(f"    '{mod}',")
        print("  ]")
    
    print("\n完整配置（注意力+MLP）:")
    if attention_modules and mlp_modules:
        all_targets = attention_modules[:2] + mlp_modules[:2]
        print("  target_modules: [")
        for mod in all_targets:
            print(f"    '{mod}',")
        print("  ]")

if __name__ == "__main__":
    inspect_gpt2_modules()