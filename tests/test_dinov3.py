"""DINOv3 from timm's ungated copy: weight conversion and the access check."""
import pytest
import torch

from idol import conditioning
from idol.dinov3 import TIMM_COPIES, convert_timm_state_dict

timm = pytest.importorskip('timm')
DINOV3 = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
FLUX2 = 'black-forest-labs/FLUX.2-dev'


@torch.no_grad()
def test_converted_timm_weights_give_the_same_features_as_timm():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    torch.manual_seed(0)
    source = timm.create_model('vit_huge_plus_patch16_dinov3', pretrained=False, embed_dim=64, depth=2,
                               num_heads=2, img_size=64).eval()
    for parameter in source.parameters():
        parameter.normal_(0, 0.2)
    config = dict(TIMM_COPIES[DINOV3][1], hidden_size=64, intermediate_size=256, num_attention_heads=2,
                  num_hidden_layers=2, image_size=64)
    model = DINOv3ViTModel(DINOv3ViTConfig(**config)).eval()
    model.load_state_dict(convert_timm_state_dict(source.state_dict()), strict=True)
    pixels = torch.randn(1, 3, 64, 64)
    torch.testing.assert_close(model(pixel_values=pixels).last_hidden_state, source.forward_features(pixels),
                               atol=1e-4, rtol=1e-4)


def test_conversion_rejects_unknown_tensors():
    source = timm.create_model('vit_huge_plus_patch16_dinov3', pretrained=False, embed_dim=64, depth=1,
                               num_heads=2, img_size=64)
    state = dict(source.state_dict(), extra=torch.zeros(1))
    with pytest.raises(ValueError, match='extra'):
        convert_timm_state_dict(state)


@pytest.fixture
def gated(monkeypatch):
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    def auth_check(repo, **kwargs):
        response = httpx.Response(403, request=httpx.Request('GET', f'https://huggingface.co/{repo}'))
        raise GatedRepoError(f'{repo} is gated', response=response)

    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    monkeypatch.setattr(huggingface_hub, 'auth_check', auth_check)
    monkeypatch.setattr(conditioning, '_ACCESSIBLE_REPOS', set())


def test_a_gated_dinov3_does_not_stop_the_run(gated):
    conditioning.require_encoder_access([DINOV3])


def test_a_gated_flux2_vae_still_stops_the_run(gated):
    with pytest.raises(conditioning.EncoderAccessError, match='FLUX.2-dev'):
        conditioning.require_encoder_access([DINOV3, FLUX2])
