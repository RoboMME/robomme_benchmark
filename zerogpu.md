# ZeroGPU Fix Notes

## Problem

Hugging Face ZeroGPU runs failed during episode initialization with:

```text
RuntimeError: Failed to find a supported physical device "cpu"
```

The root cause was that the app still defaulted `ROBOMME_RENDER_BACKEND` to `cpu` in Spaces, so ManiSkill/SAPIEN attempted to create a render device with `sapien.Device("cpu")`. That works for local CPU fallback, but it is not a valid ZeroGPU rendering path.

After switching the default to `cuda`, the next failure became:

```text
RuntimeError: Failed to find a supported physical device "cuda:0"
```

That exposed a second issue specific to ZeroGPU: the app imports RoboMME and SAPIEN in the main Space process before `@spaces.GPU` allocates a real GPU worker. Hugging Face ZeroGPU officially targets "most PyTorch-based GPU Spaces" and warns that compatibility can be limited. In this project, SAPIEN also needs Vulkan/EGL runtime configuration in the worker process, not just PyTorch CUDA access.

From the deployed Space logs, the allocated ZeroGPU worker exposes:

- `CUDA_VISIBLE_DEVICES=MIG-...`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- SAPIEN error: `Your GPU driver does not support Vulkan`

That means the worker can execute CUDA workloads, but it does not expose the graphics/Vulkan capability SAPIEN needs for true GPU rendering. So the practical fix is:

- use ZeroGPU for the heavy compute section
- automatically fall back to software Vulkan rendering when GPU Vulkan is unavailable
- skip GPU Vulkan attempts entirely when the worker advertises only `compute,utility` without `graphics`

## Adjustments

1. `gradio-web/main.py`
   - Keep local development on the existing CPU fallback.
   - In Hugging Face Spaces, default `ROBOMME_RENDER_BACKEND` to `cuda` instead of `cpu`.
   - Preserve explicit render backend overrides such as `pci:...`.
   - Stop auto-injecting the `llvmpipe` CPU Vulkan ICD in Spaces when `VK_ICD_FILENAMES` is unset.
   - Mark runtime-injected render backend defaults with `ROBOMME_RENDER_BACKEND_AUTO=1` so downstream code can still treat them as fallbackable defaults rather than hard user overrides.

2. `src/robomme/env_record_wrapper/episode_config_resolver.py`
   - Preserve the existing local default behavior.
   - Add a dedicated software-render mode via `ROBOMME_FORCE_SOFTWARE_RENDER_MODE=1`.
   - In software-render mode, force llvmpipe and bind SAPIEN candidates in this order:
     - `pci:0000:00:00.0`
     - `cpu`
   - In compute-only Spaces workers, skip GPU Vulkan render candidates entirely.

3. `gradio-web/software_render_session.py`
   - Add a clean spawned subprocess backend for software rendering.
   - The subprocess sets `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` before importing `oracle_logic`, `robomme`, `sapien`, or `mani_skill`.
   - It runs a small SAPIEN software-render self-test first, then owns the real `OracleSession`.
   - The parent process talks to it over a simple RPC pipe and receives state snapshots after each call.

4. `gradio-web/process_session.py`
   - Split `ProcessSessionProxy` into two runtime modes:
     - local/in-process session
     - software-render subprocess session for compute-only ZeroGPU Spaces
   - Keep the public proxy interface unchanged for callbacks/UI.
   - Return a clear unsupported message when even llvmpipe software rendering is unavailable.

5. Tests
   - Updated runtime tests to verify:
     - local runs still force CPU fallback
     - Spaces preserve explicit render backend overrides
     - Spaces default to `cuda` render backend when unset
   - Added environment-builder coverage for forced software-render mode.
   - Added `ProcessSessionProxy` coverage for compute-only Spaces selecting the software-render subprocess backend and returning explicit unsupported messages.

6. `gradio-web/zerogpu_runtime.py`
   - Log the effective ZeroGPU worker runtime once per worker process.
   - Defer any SAPIEN render probing so the worker does not poison Vulkan initialization before the chosen render runtime is configured.

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
- Hugging Face ZeroGPU compute-only workers no longer attempt SAPIEN GPU Vulkan rendering.
- Instead, they use a clean llvmpipe software-render subprocess that owns the real `OracleSession`.
- If llvmpipe works, the Space runs in the requested "ZeroGPU for compute + CPU software rendering for images" mode.
- If llvmpipe still fails, the UI now returns an explicit unsupported-environment error instead of only surfacing the raw SAPIEN device exception.
