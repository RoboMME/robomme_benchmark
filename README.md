---
title: RoboMME Oracle Planner
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
python_version: 3.10.13
---

This Space runs the RoboMME Gradio interface with the Gradio SDK and Hugging Face ZeroGPU.

Deployment notes:

- This repository targets a native Hugging Face Gradio Space, not a Docker Space.
- The Space entrypoint is root `app.py` as declared by `app_file: app.py`.
- Python dependencies are installed from root `requirements.txt`.
- Debian/system dependencies are installed from root `packages.txt`.
- ZeroGPU is enabled through the `spaces` package and `@spaces.GPU` on heavy functions.

Local development uses `uv`:

```bash
uv sync
uv run python app.py
```

The local `uv` workflow is only for development. Hugging Face Spaces does not run `uv` to build this app; it installs dependencies from `requirements.txt` and starts `python app.py`.

The app keeps a local CPU fallback for development. On Hugging Face ZeroGPU, the runtime preserves the GPU environment and requests GPU only around the heavy environment calls via `@spaces.GPU`.

```bash
uv run python app.py
```
