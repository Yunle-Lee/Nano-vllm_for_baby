import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器：直接拷贝权重（无 TP 切分）"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从 safetensors 文件中加载模型权重。
    
    支持两种加载模式：
    1. 普通模块：直接根据参数名加载（如 "model.layers.0.input_layernorm.weight"）
    2. Packed 模块：通过 packed_modules_mapping 将 HF 的分散权重映射到 TP 合并权重
       例如 q_proj/k_proj/v_proj → qkv_proj（三个分开的参数合并为一个 QKVLinear）
       
    packed_modules_mapping 格式：{hf_name: (model_name, shard_id)}
    例如：{"q_proj": ("qkv_proj", "q"), "gate_proj": ("gate_up_proj", 0)}
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否是 packed module（合并权重）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)      # 替换参数名
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")  # 使用自定义加载器
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通参数：直接用参数名加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
