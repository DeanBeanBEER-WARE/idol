"""DINOv3 for the image encoder: Meta's gated repo when this account can read it, otherwise timm's ungated copy of
the same original weights, converted to the transformers model."""
import os
from typing import Dict, Optional

import torch

# Meta repo -> (timm copy, transformers config of the model; public in onnx-community/<model>-ONNX)
TIMM_COPIES = {
    'facebook/dinov3-vith16plus-pretrain-lvd1689m': ('timm/vit_huge_plus_patch16_dinov3.lvd1689m', dict(
        hidden_size=1280, intermediate_size=5120, num_attention_heads=20, num_hidden_layers=32,
        num_register_tokens=4, patch_size=16, image_size=224, hidden_act='silu', use_gated_mlp=True,
        layer_norm_eps=1e-5, layerscale_value=1.0, mlp_bias=True, query_bias=True, key_bias=False, value_bias=True,
        proj_bias=True, rope_theta=100.0, pos_embed_rescale=2.0)),
}


def can_read(repo: str) -> bool:
    """Whether the logged-in Hugging Face account can download ``repo``; offline or on network errors, whether its
    config is in the local cache."""
    from huggingface_hub import auth_check, try_to_load_from_cache
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if os.environ.get('HF_HUB_OFFLINE', '0').lower() not in ('1', 'true', 'yes'):
        try:
            auth_check(repo)
            return True
        except (GatedRepoError, RepositoryNotFoundError):
            return False
        except Exception:
            pass
    return isinstance(try_to_load_from_cache(repo, 'config.json'), str)


def convert_timm_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """timm DINOv3 ViT state dict -> transformers ``DINOv3ViTModel`` state dict.

    The fused qkv projection is split into q, k and v; the original q and v biases are all zero (timm drops them),
    and the mask token is only used for masked training. Raises ValueError when a timm tensor is left over."""
    state = dict(state)
    width = state['cls_token'].shape[-1]
    out = {
        'embeddings.cls_token': state.pop('cls_token'),
        'embeddings.register_tokens': state.pop('reg_token'),
        'embeddings.mask_token': torch.zeros(1, 1, width),
        'embeddings.patch_embeddings.weight': state.pop('patch_embed.proj.weight'),
        'embeddings.patch_embeddings.bias': state.pop('patch_embed.proj.bias'),
        'norm.weight': state.pop('norm.weight'),
        'norm.bias': state.pop('norm.bias'),
    }
    depth = 1 + max(int(key.split('.')[1]) for key in state if key.startswith('blocks.'))
    renames = {'norm1': 'norm1', 'norm2': 'norm2', 'attn.proj': 'attention.o_proj', 'mlp.fc1_g': 'mlp.gate_proj',
               'mlp.fc1_x': 'mlp.up_proj', 'mlp.fc2': 'mlp.down_proj'}
    for i in range(depth):
        src, dst = f'blocks.{i}.', f'model.layer.{i}.'
        q, k, v = state.pop(src + 'attn.qkv.weight').chunk(3, dim=0)
        out.update({dst + 'attention.q_proj.weight': q, dst + 'attention.k_proj.weight': k,
                    dst + 'attention.v_proj.weight': v,
                    dst + 'attention.q_proj.bias': torch.zeros(width), dst + 'attention.v_proj.bias': torch.zeros(width),
                    dst + 'layer_scale1.lambda1': state.pop(src + 'gamma_1'),
                    dst + 'layer_scale2.lambda1': state.pop(src + 'gamma_2')})
        for old, new in renames.items():
            for kind in ('weight', 'bias'):
                out[f'{dst}{new}.{kind}'] = state.pop(f'{src}{old}.{kind}')
    state.pop('rope.periods', None)  # transformers derives the RoPE frequencies from the config
    if state:
        raise ValueError(f'unconverted timm tensors: {sorted(state)}')
    return out


def load_dinov3(repo: str, dtype: torch.dtype, prefer_timm: Optional[bool] = None):
    """The transformers DINOv3 model of ``repo`` in ``dtype``, in eval mode.

    prefer_timm: None loads the timm copy only when ``repo`` is not readable for this account and timm hosts it;
    True / False force the timm copy / the Meta repo. Raises ValueError when the timm copy is forced but unknown."""
    from transformers import AutoModel

    if prefer_timm is None:
        prefer_timm = repo in TIMM_COPIES and not can_read(repo)
    if not prefer_timm:
        return AutoModel.from_pretrained(repo, torch_dtype=dtype).eval()
    if repo not in TIMM_COPIES:
        raise ValueError(f'no timm copy known for {repo}')

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    timm_repo, config = TIMM_COPIES[repo]
    model = DINOv3ViTModel(DINOv3ViTConfig(**config))
    model.load_state_dict(convert_timm_state_dict(load_file(hf_hub_download(timm_repo, 'model.safetensors'))),
                          strict=True)
    return model.to(dtype).eval()
