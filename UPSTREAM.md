# Upstream

This repository contains the Idol SDK by Speridlabs, extended to run on Apple Silicon.

| | |
|---|---|
| Repository | https://github.com/speridlabs/idol-sdk |
| Commit | `4702a7f00850e34b6e83a2170cc123c3e9570744` ("initial commit", 2026-10-29) |
| License | Apache-2.0, see [LICENSE](LICENSE) |
| Weights | https://huggingface.co/speridlabs/idol |

Copied verbatim from that commit: `idol/`, `demo/`, `assets/`, `LICENSE` and `pyproject.toml`. Upstream's `README.md` and `.gitignore` are replaced by this repository's own files; the README keeps upstream's model description.

## Local modifications

Every modified upstream file starts with the line `Modified in DeanBeanBEER-WARE/idol, see UPSTREAM.md.` (Apache-2.0 Section 4(b)).

| File | Change | Since |
|---|---|---|
| `pyproject.toml` | Local version `0.1.0+mac.N`; `dev` extra (ruff, pytest); ruff limited to the classic rule set with per-file ignores for upstream code; pytest settings | `0.1.0+mac.1` |
| `pyproject.toml` | Upper bounds on torch, torchvision, transformers and diffusers (newest versions tested on MPS); torchvision at least 0.29 for the MPS kernel of `deform_conv2d` | `0.1.0+mac.2` |
| `idol/__init__.py` | Local version | `0.1.0+mac.1` |
| `idol/pipeline.py` | `select_device`: `IDOL_DEVICE`, then CUDA, MPS, CPU; off CUDA every random draw comes from a CPU generator seeded with `seed`; `decode` takes that generator | `0.1.0+mac.2` |
| `idol/conditioning.py` | `ImageEncoder.__call__` takes the generator for the FLUX.2 VAE posterior sample | `0.1.0+mac.2` |
| `idol/models/decoder.py` | Octree `sample` and `systematic_sample` take the generator | `0.1.0+mac.2` |
| `idol/sample.py` | `--device` flag; `--no_optimize` help says CUDA | `0.1.0+mac.2` |
| `demo/model.py` | Device picked automatically outside ZeroGPU | `0.1.0+mac.2` |

Added files: `idol/rng.py` (random draws from an optional CPU generator), `tests/`.

## Syncing with upstream

Only when the repository owner asks for it.

1. Clone upstream into a scratch directory outside this repository.
2. Review what changed since the recorded commit: `git -C <clone> diff <recorded commit> <new commit> -- idol demo assets LICENSE pyproject.toml README.md`.
3. Apply those changes here. Files listed under Local modifications are merged by hand, keeping the local changes.
4. Record the new commit above, set the version to the upstream version with the local segment `+mac.1`, and run the verification commands.
