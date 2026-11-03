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
| `idol/__init__.py` | Local version | `0.1.0+mac.1` |

## Syncing with upstream

Only when the repository owner asks for it.

1. Clone upstream into a scratch directory outside this repository.
2. Review what changed since the recorded commit: `git -C <clone> diff <recorded commit> <new commit> -- idol demo assets LICENSE pyproject.toml README.md`.
3. Apply those changes here. Files listed under Local modifications are merged by hand, keeping the local changes.
4. Record the new commit above, set the version to the upstream version with the local segment `+mac.1`, and run the verification commands.
