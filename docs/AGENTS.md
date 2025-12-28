# Repository Guidelines

## Project Structure & Module Organization

- `inference_gradio.py`: Gradio Web UI for HF-format checkpoints (`trust_remote_code=True`).
- `inference_commandline_hf.py`: CLI inference for HF-format checkpoints.
- `inference_commandline.py`: CLI inference for `.pth` bundles (non-HF).
- `inference_tts_utils.py`: Shared TTS pipeline (tokenization, decoding, normalization).
- `hf_export/`: Hugging Face wrapper code used by exported checkpoints (model/config).
- `models/`: Core model implementation (training/inference internals).
- `data/`: Tokenizers and dataset utilities.
- `steps/`, `main.py`: Training entrypoints and training pipeline.
- `scripts/`, `examples/`, `figures/`: Helper scripts, examples, and assets.

## Build, Test, and Development Commands

- `sh serve.sh`: Run the Gradio demo (uses `uv run` and sets common env vars).
- `python inference_gradio.py --model_dir Aratako/T5Gemma-TTS-2b-2b --port 7860`: Run Gradio directly.
- `python inference_commandline_hf.py --model_dir <repo_or_path> ...`: One-shot CLI inference (HF).
- `python inference_commandline.py --model_root . --model_name trained ...`: One-shot CLI inference (`.pth`).
- `docker compose up --build`: Containerized run (see `README.md` for env overrides).
- Quick sanity check: `python -m py_compile inference_gradio.py inference_tts_utils.py`.

## Coding Style & Naming Conventions

- Python 3.12+ (see `pyproject.toml`); prefer type hints for public helpers.
- Keep changes minimal and localized; follow existing naming (snake_case for functions/vars).
- No enforced formatter/linter in this repo; keep imports grouped and lines reasonably short.

## Testing Guidelines

- No formal test suite is currently configured. For changes, run:
  - `python -m py_compile <touched_files...>`
  - A quick manual smoke test via `sh serve.sh` or CLI inference.

## Commit & Pull Request Guidelines

- Commit messages follow short, imperative, sentence-case summaries (e.g., “Optimize inference loop…”).
- PRs should include: what changed, how to reproduce (exact command), and (for perf work) before/after latency or `tokens/s`.

## Performance & Configuration Tips

- `--max_batch` controls the UI batch slider limit; larger values increase VRAM usage.
- Env toggles used by Gradio:
  - `T5GEMMA_DETERMINISTIC=1` for determinism (usually slower).
  - `T5GEMMA_EMPTY_CACHE=1` to call `torch.cuda.empty_cache()` (low-VRAM, often slower).
