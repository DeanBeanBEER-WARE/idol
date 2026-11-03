"""PLY export of the Gaussian representation."""
import numpy as np
import pytest
import torch
from plyfile import PlyData

from idol import Gaussian

PLY_PROPERTIES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                  'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
COUNT = 64


@pytest.fixture
def gaussian():
    generator = torch.Generator().manual_seed(0)
    return Gaussian(raw_xyz=torch.rand(COUNT, 3, generator=generator),
                    features_dc=torch.randn(COUNT, 1, 3, generator=generator),
                    raw_scaling=torch.randn(COUNT, 3, generator=generator),
                    raw_rotation=torch.randn(COUNT, 4, generator=generator),
                    raw_opacity=torch.randn(COUNT, 1, generator=generator),
                    scaling_bias=0.004, opacity_bias=0.1, minimum_kernel_size=0.0009)


@pytest.fixture
def vertex(gaussian, tmp_path):
    path = tmp_path / 'object.ply'
    gaussian.save_ply(str(path))
    return PlyData.read(str(path))['vertex']


def test_writes_the_standard_3dgs_properties(vertex):
    assert [prop.name for prop in vertex.properties] == PLY_PROPERTIES
    assert vertex.count == COUNT
    for name in PLY_PROPERTIES:
        assert np.isfinite(vertex[name]).all()


def test_turns_the_z_up_object_into_y_up(gaussian, vertex):
    xyz = gaussian.xyz.numpy()
    np.testing.assert_array_equal(vertex['x'], xyz[:, 0])
    np.testing.assert_array_equal(vertex['y'], -xyz[:, 2])
    np.testing.assert_array_equal(vertex['z'], xyz[:, 1])


def test_stores_colour_and_inverse_activations(gaussian, vertex):
    colour = np.stack([vertex[f'f_dc_{i}'] for i in range(3)], axis=1)
    np.testing.assert_array_equal(colour, gaussian.features_dc[:, 0].numpy())
    np.testing.assert_allclose(1 / (1 + np.exp(-vertex['opacity'])), gaussian.opacity[:, 0].numpy(), rtol=1e-5)
    scale = np.stack([vertex[f'scale_{i}'] for i in range(3)], axis=1)
    np.testing.assert_allclose(np.exp(scale), gaussian.scaling.numpy(), rtol=1e-5)
    rotation = np.stack([vertex[f'rot_{i}'] for i in range(4)], axis=1)
    np.testing.assert_allclose(np.linalg.norm(rotation, axis=1), 1.0, rtol=1e-5)
