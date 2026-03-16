#!/bin/sh
set -eu

if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=1
fi

export CUDA_VISIBLE_DEVICES=-1
export NVIDIA_VISIBLE_DEVICES=void
unset NVIDIA_DRIVER_CAPABILITIES
unset SAPIEN_RENDER_DEVICE
unset MUJOCO_GL
if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -f /usr/share/vulkan/icd.d/lvp_icd.x86_64.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
fi

echo "[entrypoint] Starting RoboMME Gradio app in CPU-only mode"
echo "[entrypoint] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[entrypoint] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[entrypoint] NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
echo "[entrypoint] SAPIEN_RENDER_DEVICE=${SAPIEN_RENDER_DEVICE:-<unset>}"
echo "[entrypoint] VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
exec "$@"
