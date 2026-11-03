"""Device selection and seeded random draws."""
import pytest
import torch

from idol.pipeline import select_device
from idol.rng import rand, randn

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.delenv('IDOL_DEVICE', raising=False)


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.delenv('IDOL_DEVICE', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert select_device() == torch.device('cuda')


def test_auto_picks_mps_without_cuda(no_cuda, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert select_device() == torch.device('mps')


def test_auto_falls_back_to_cpu(no_cuda, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert select_device('auto') == torch.device('cpu')


def test_environment_variable_overrides_auto(no_cuda, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setenv('IDOL_DEVICE', 'cpu')
    assert select_device() == torch.device('cpu')


def test_argument_overrides_environment_variable(monkeypatch):
    monkeypatch.setenv('IDOL_DEVICE', 'mps')
    assert select_device('cpu') == torch.device('cpu')


@pytest.mark.parametrize('draw', [rand, randn])
def test_a_cpu_generator_gives_the_same_draws_on_every_device(draw):
    draws = [draw((3, 5), torch.Generator().manual_seed(7), device) for device in DEVICES]
    for device, values in zip(DEVICES, draws):
        assert values.device.type == device
        assert values.dtype == torch.float32
        assert torch.equal(values.cpu(), draws[0])
