"""Random draws on any device, optionally from a CPU generator so that a seed gives the same draws on every device."""
from typing import Optional, Sequence, Union

import torch


def randn(shape: Sequence[int], generator: Optional[torch.Generator],
          device: Union[str, torch.device]) -> torch.Tensor:
    """Standard normal float32 draws of ``shape`` on ``device``, from ``generator`` (a CPU generator) or, when it is
    None, from the default generator of ``device``."""
    if generator is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=generator).to(device)


def rand(shape: Sequence[int], generator: Optional[torch.Generator],
         device: Union[str, torch.device]) -> torch.Tensor:
    """Uniform float32 draws in [0, 1) of ``shape`` on ``device``, from ``generator`` (a CPU generator) or, when it
    is None, from the default generator of ``device``."""
    if generator is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=generator).to(device)
