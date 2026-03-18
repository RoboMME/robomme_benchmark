---
title: RoboMME Oracle Planner
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
python_version: 3.10.13
---

This Space runs the RoboMME Gradio interface with the Gradio SDK and Hugging Face ZeroGPU.

Local development uses `uv`:

```bash
uv sync
uv run python app.py
```

The app keeps a local CPU fallback for development. On Hugging Face ZeroGPU, GPU is requested only around the heavy environment calls via `@spaces.GPU`.

```bash
uv run python app.py
```
