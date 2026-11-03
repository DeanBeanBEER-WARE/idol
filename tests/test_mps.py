"""The models run on the Apple GPU (MPS) and match the CPU, checked with small randomly initialised models."""
import copy

import pytest
import torch
from tensordict import TensorDict
from torchvision.ops import deform_conv2d

from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
from idol.models.decoder import systematic_sample

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs the Apple GPU (MPS)')

MPS = torch.device('mps')
REPRESENTATION = {
    'num_gaussians': 32,
    'lr': {'_xyz': 1.0, '_features_dc': 1.0, '_opacity': 1.0, '_scaling': 1.0, '_rotation': 0.1},
    'perturbe_size': 1.5, '3d_filter_kernel_size': 0.0009, 'scaling_bias': 0.004, 'opacity_bias': 0.1,
}
FP16_TOLERANCE = {'atol': 5e-3, 'rtol': 5e-3}
BF16_TOLERANCE = {'atol': 0.1, 'rtol': 0.05}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def on_mps(module):
    return copy.deepcopy(module).to(MPS)


def octree_decoder():
    return OctreeProbabilityFixedlenDecoder(model_channels=128, cond_channels=16, num_blocks=1, num_heads=2).eval()


@torch.no_grad()
def test_denoiser_matches_cpu():
    model = LatentSeqMMFlowModel(q_token_length=64, in_channels=16, cam_channels=5, model_channels=128,
                                 cond_channels=32, cond2_channels=8, num_blocks=2, num_refiner_blocks=2,
                                 num_heads=2, num_head_channels=64).eval()
    model.pos_pe.copy_(torch.rand_like(model.pos_pe))
    x = TensorDict({'latent': torch.randn(1, 64, 16), 'camera': torch.randn(1, 1, 5)}, batch_size=1)
    t = torch.tensor([500.0])
    cond = TensorDict({'feature1': torch.randn(1, 21, 32), 'feature2': torch.randn(1, 21, 8)}, batch_size=1)
    expected = model(x, t, cond)
    actual = on_mps(model)(x.to(MPS), t.to(MPS), cond.to(MPS))
    for key in ('latent', 'camera'):
        torch.testing.assert_close(actual[key].cpu(), expected[key], **FP16_TOLERANCE)


@torch.no_grad()
def test_octree_decoder_matches_cpu():
    model = octree_decoder()
    coords, latent = torch.rand(1, 8, 3), torch.randn(1, 64, 16)
    level = torch.full((1,), 4, dtype=torch.long)
    expected = model(coords, level, latent)
    actual = on_mps(model)(coords.to(MPS), level.to(MPS), latent.to(MPS))
    torch.testing.assert_close(actual.cpu(), expected, **FP16_TOLERANCE)


@torch.no_grad()
def test_octree_sampling_places_every_point_and_follows_the_seed():
    model = on_mps(octree_decoder())
    latent = torch.randn(1, 64, 16, device=MPS)

    def place(seed):
        return OctreeProbabilityFixedlenDecoder.sample(model, latent, num_points=1000, level=8,
                                                       generator=torch.Generator().manual_seed(seed)).cpu()

    points = place(1)
    assert points.shape == (1, 1000, 3)
    assert points.min() >= 0 and points.max() <= 1
    assert torch.equal(points, place(1))
    assert not torch.equal(points, place(2))


def test_systematic_sampling_keeps_the_counts():
    probs = torch.softmax(torch.randn(6, 8), dim=-1).to(MPS)
    counts = torch.tensor([0, 1, 7, 7, 100, 3], device=MPS)
    split = systematic_sample(probs, counts, torch.Generator().manual_seed(0))
    assert torch.equal(split.sum(-1).cpu(), counts.cpu())


@torch.no_grad()
def test_gaussian_decoder_matches_cpu():
    model = GaussianFixedlenDecoder(in_channels=3, model_channels=128, cond_channels=16, num_blocks=1, num_heads=2,
                                    representation_config=REPRESENTATION).eval()
    anchors, latent = torch.rand(1, 32, 3), torch.randn(1, 64, 16)
    expected = model(anchors, latent)
    mps_model = on_mps(model)
    actual = mps_model(anchors.to(MPS), latent.to(MPS))
    torch.testing.assert_close(actual.cpu(), expected, **FP16_TOLERANCE)
    gaussian = mps_model.to_representation(anchors.to(MPS), actual)[0]
    assert gaussian.xyz.shape == (32 * 32, 3)


def test_birefnet_deformable_convolution_matches_cpu():
    inp, offset = torch.randn(1, 8, 16, 16), torch.randn(1, 18, 16, 16) * 2
    mask, weight = torch.rand(1, 9, 16, 16) * 2, torch.randn(4, 8, 3, 3)
    expected = deform_conv2d(inp, offset, weight, padding=1, mask=mask)
    actual = deform_conv2d(inp.to(MPS), offset.to(MPS), weight.to(MPS), padding=1, mask=mask.to(MPS))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_dinov3_runs_in_bf16():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    config = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                             image_size=64, patch_size=16, num_register_tokens=4)
    model = DINOv3ViTModel(config).eval()
    pixels = torch.randn(1, 3, 64, 64)
    expected = model(pixel_values=pixels).last_hidden_state
    actual = on_mps(model).to(torch.bfloat16)(pixel_values=pixels.to(MPS, torch.bfloat16)).last_hidden_state
    torch.testing.assert_close(actual.float().cpu(), expected, **BF16_TOLERANCE)


@torch.no_grad()
def test_flux2_vae_encodes_in_bf16():
    from diffusers.models import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 32, 32), latent_channels=8, norm_num_groups=8,
                             layers_per_block=1).eval()
    images = torch.rand(1, 3, 64, 64) * 2 - 1
    expected = vae.encode(images).latent_dist.mean
    mps_vae = on_mps(vae).to(torch.bfloat16)
    actual = mps_vae.encode(images.to(MPS, torch.bfloat16)).latent_dist.mean
    torch.testing.assert_close(actual.float().cpu(), expected, **BF16_TOLERANCE)
    seeded = mps_vae.encode(images.to(MPS, torch.bfloat16)).latent_dist.sample(generator=torch.Generator())
    assert seeded.device.type == 'mps'
