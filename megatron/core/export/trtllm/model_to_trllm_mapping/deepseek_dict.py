import torch
from megatron.core.export.trtllm.trtllm_layers import TRTLLMLayers

def deepseek_preprocessing_weight(model_state_dict: dict):
    for k in list(model_state_dict.keys()):
        if k.endswith('linear_q_down_proj.weight'):
            prefix = k[:-len('linear_q_down_proj.weight')]
            q_down = model_state_dict.pop(k)
            kv_down = model_state_dict.pop(prefix + 'linear_kv_down_proj.weight')
            new_key = prefix + 'fused_a.weight'
            model_state_dict[new_key] = torch.concatenate([q_down, kv_down], dim=0)


DEEPSEEK_DICT = {
    # ATTENTION
    'decoder.layers.self_attention.fused_a.weight': TRTLLMLayers.attention_fused_a_weight,
    'decoder.layers.self_attention.linear_q_up_proj.weight': TRTLLMLayers.attention_q_up_weight,
    'decoder.layers.self_attention.linear_kv_up_proj.weight': TRTLLMLayers.attention_kv_up_weight,
    'decoder.layers.self_attention.linear_q_up_proj.layer_norm_weight': TRTLLMLayers.attention_q_layernorm_weight,
    'decoder.layers.self_attention.linear_kv_up_proj.layer_norm_weight': TRTLLMLayers.attention_kv_layernorm_weight,
    # SHARE EXPERTS
    'decoder.layers.mlp.shared_experts.linear_fc1.weight': TRTLLMLayers.mlp_share_expert_fc,
    'decoder.layers.mlp.shared_experts.linear_fc2.weight': TRTLLMLayers.mlp_share_expert_proj,
    'decoder.layers.mlp.router.expert_bias': TRTLLMLayers.mlp_router_expert_bias,

    'preprocess_weight': deepseek_preprocessing_weight,
}