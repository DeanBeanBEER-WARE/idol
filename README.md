# idol

**One image of an object in, a 3D Gaussian splat out, on Apple Silicon.**

This repository is a fork of [speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk) that adds support for the Apple GPU through PyTorch's MPS backend. Upstream targets NVIDIA GPUs with CUDA. The model, its weights and the results are Speridlabs' work; the local changes are listed in [UPSTREAM.md](UPSTREAM.md).

Idol takes a render, a generated image or a photo, removes the background, frames the object and generates up to 1,048,576 Gaussians, including the sides the picture never showed. The result is a standard 3D Gaussian Splatting `.ply` file.

Idol runs on the Apple GPU (MPS), on NVIDIA GPUs (CUDA) or on the CPU. The device is picked automatically in that order: CUDA, then MPS, then CPU. `--device` or the `IDOL_DEVICE` environment variable override it.

## Requirements

- macOS 14 or newer on Apple Silicon for the Apple GPU (or Linux / Windows with an NVIDIA GPU)
- Python 3.11 or newer (developed with 3.13)
- A Hugging Face account with access to the two gated image encoders (see below)
- About 7 GB of disk space for the model downloads, which go to the Hugging Face cache (`~/.cache/huggingface`)

## Install

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # add ",demo" for the local Gradio demo
```

## Hugging Face access

The weights download on first use. Two of the image encoders are gated:

1. Accept the terms on [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) and [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (only the VAE is downloaded).
2. Log in: `.venv/bin/hf auth login`, or set `HF_TOKEN`.

Without access, Idol stops before downloading anything and says what is missing.

## Usage

```bash
.venv/bin/python -m idol.sample --image photo.jpg --mode real
.venv/bin/python -m idol.sample --image renders/ --mode synthetic --out outputs
```

For each image this writes `<name>.ply` (open it in any Gaussian splat viewer), `<name>_input.png` (the image the model saw, after background removal and cropping) and `<name>_camera.json` (the predicted camera of the input).

| `--mode` | for |
|---|---|
| `synthetic` (default) | computer generated images |
| `real` | real world photographs |

| flag | default | |
|---|---|---|
| `--seed` | 42 | same seed, same object |
| `--steps` | 50 | sampling steps |
| `--cfg_strength` | 3.0 | classifier-free guidance |
| `--rescale_t` | 3.0 | time shift: higher spends more steps at high noise |
| `--num_gaussians` | 262144 | 32,768 to 1,048,576, a multiple of 32 |
| `--crop_margin` | by mode | square crop side = object bounding box x this |
| `--dinov3_dtype` | bf16 | `fp32` runs the DINOv3 encoder in full precision |
| `--weights` | `speridlabs/idol` | a Hugging Face repo or a local folder |
| `--device` | `IDOL_DEVICE`, else `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `--no_optimize` | optimized | turn off the faster CUDA sampling |

Python API:

```python
from idol import IdolPipeline

pipeline = IdolPipeline.from_pretrained("speridlabs/idol")
output = pipeline("photo.jpg", mode="real", seed=42, num_gaussians=262_144)
output.gaussian.save_ply("photo.ply")
output.preprocessed_image.save("photo_input.png")
```

## On the Mac

Measured on an M3 Max (64 GB) with PyTorch 2.15 and the default settings:

| Stage | Time | Peak memory |
|---|---|---|
| Background removal (BiRefNet, fp32) | 0.5 s | 8.5 GB |
| Sampling (50 steps with guidance) | 6.5 min | 3.0 GB |
| Decoding 262,144 Gaussians | 3 s | 1.7 GB |
| Decoding 1,048,576 Gaussians | 16.5 s | 3.8 GB |

Peak memory is the MPS driver allocation of that stage, its weights included. Sampling dominates the run time and scales linearly with `--steps`. The image encoders (DINOv3 and the FLUX.2 VAE) are not in the table yet.

Off CUDA every random draw comes from a CPU generator seeded with `--seed`, so the Apple GPU and the (much slower) CPU get the same draws. The same seed gives a different object on CUDA.

## Development

```bash
.venv/bin/ruff check .
.venv/bin/pytest
```

The tests use small randomly initialised models and never download weights.

## Model

| | |
|---|---|
| Output | 3D Gaussians: 32,768 to 1,048,576 (default 262,144), standard 3DGS `.ply` |
| DiT | 370M parameters |
| Image conditioning | DINOv3 ViT-H+ and FLUX.2 VAE |
| Decoder | 288M parameters |
| Background removal | BiRefNet |
| Sampler | Euler, 50 steps, classifier-free guidance 3.0, time shift 3.0 |

| Component | Parameters | Precision |
|---|---|---|
| DiT (denoiser) | 370.6M | FP16 transformer blocks; FP32 embeddings and norms |
| Octree decoder | 50.8M | FP16 transformer blocks; FP32 embeddings and norms |
| Gaussian decoder | 237.3M | FP16 transformer blocks; FP32 embeddings and norms |
| BiRefNet (background removal) | 220.7M | FP16 on CUDA |
| DINOv3 ViT-H+/16 | 840.6M | bf16 |
| FLUX.2 VAE | 84.1M | bf16 |

Benchmarks and CUDA timings are in the [upstream README](https://github.com/speridlabs/idol-sdk#benchmark).

## Credits and license

Idol is made by [Speridlabs](https://speridlabs.com) and builds on [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat). It uses [DINOv3](https://github.com/facebookresearch/dinov3), the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) VAE and [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

Code and weights are licensed under Apache-2.0, see [LICENSE](LICENSE).
