# ZeroGPU Fix Notes

## Problem

Hugging Face ZeroGPU runs failed during episode initialization with:

```text
RuntimeError: Failed to find a supported physical device "cpu"
```

The root cause was that the app still defaulted `ROBOMME_RENDER_BACKEND` to `cpu` in Spaces, so ManiSkill/SAPIEN attempted to create a render device with `sapien.Device("cpu")`. That works for local CPU fallback, but it is not a valid ZeroGPU rendering path.

## Adjustments

1. `gradio-web/main.py`
   - Keep local development on the existing CPU fallback.
   - In Hugging Face Spaces, default `ROBOMME_RENDER_BACKEND` to `cuda` instead of `cpu`.
   - Preserve explicit render backend overrides such as `pci:...`.
   - Stop auto-injecting the `llvmpipe` CPU Vulkan ICD in Spaces when `VK_ICD_FILENAMES` is unset.

2. `src/robomme/env_record_wrapper/episode_config_resolver.py`
   - Make environment creation default to `cuda` rendering in Spaces even if the builder is used outside the normal `main.py` startup path.
   - Keep local default behavior as `cpu`.

3. Tests
   - Updated runtime tests to verify:
     - local runs still force CPU fallback
     - Spaces preserve explicit render backend overrides
     - Spaces default to `cuda` render backend when unset
   - Added an environment-builder test for the Spaces default render backend.

## Local Verification

Use the local `uv` environment:

```bash
uv run pytest gradio-web/test/test_main_launch_config.py gradio-web/test/test_episode_builder_cpu_backend.py
```

Run the app locally with the existing CPU fallback:

```bash
uv run python app.py
```

## Expected ZeroGPU Behavior

- Local development: CPU fallback rendering remains the default.
- Hugging Face ZeroGPU: heavy environment calls wrapped by `@spaces.GPU` can now request GPU and construct the renderer with `cuda` instead of `cpu`.
