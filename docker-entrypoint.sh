#!/bin/sh
set -eu

if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=1
fi

export CUDA_VISIBLE_DEVICES=-1
export NVIDIA_VISIBLE_DEVICES=void
export SAPIEN_RENDER_DEVICE=cpu
unset NVIDIA_DRIVER_CAPABILITIES
unset VK_ICD_FILENAMES
unset MUJOCO_GL

echo "[entrypoint] Starting RoboMME Gradio app in CPU-only mode"
echo "[entrypoint] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[entrypoint] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[entrypoint] NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
echo "[entrypoint] SAPIEN_RENDER_DEVICE=$SAPIEN_RENDER_DEVICE"
exec "$@"
