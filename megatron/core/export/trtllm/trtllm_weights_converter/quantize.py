# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn.functional as F

FP8_AMAX = 448.0

def fp8_quatization_by_tensor(weight: torch.Tensor, block_size: int=128):
    if len(weight.shape) == 3:
        value_list = []
        scale_list = []
        for i in range(weight.shape[0]):
            value = weight[i]
            value, scale = weight_only_blocked_fp8_quantization(value, block_size)
            value_list.append(value)
            scale_list.append(scale)
        value = torch.stack(value_list)
        scale = torch.stack(scale_list)
        return value, scale
    return weight_only_blocked_fp8_quantization(weight, block_size)

def weight_only_blocked_fp8_quantization(weight: torch.Tensor, block_size: int=128):
    out_features, in_features = weight.shape

    pad_h = (block_size - out_features % block_size) % block_size
    pad_w = (block_size - in_features % block_size) % block_size
    weight = F.pad(weight, (0, pad_w, 0, pad_h))

    new_h, new_w = weight.shape
    weight = weight.view(
        new_h // block_size, block_size,
        new_w // block_size, block_size,
    )

    weight_scale_inv = torch.max(weight.clone().detach().abs(), dim=-1, keepdim=True).values
    weight_scale_inv = torch.max(weight_scale_inv, dim=1, keepdim=True).values / FP8_AMAX
    real_quant_weight = weight.clone().detach() / weight_scale_inv

    weight_scale_inv = weight_scale_inv.squeeze(dim=-1).squeeze(dim=1).to(dtype=torch.float32)
    real_quant_weight = real_quant_weight.view(new_h, new_w)[:out_features, :in_features].to(dtype=torch.float8_e4m3fn)

    return real_quant_weight, weight_scale_inv