#!/bin/sh
set -eu

if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=1
fi

export ROBOMME_RENDER_BACKEND="${ROBOMME_RENDER_BACKEND:-pci:0}"
unset SAPIEN_RENDER_DEVICE
unset MUJOCO_GL
if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -f /usr/share/vulkan/icd.d/lvp_icd.x86_64.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
fi

echo "[entrypoint] Starting RoboMME Gradio app"
echo "[entrypoint] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[entrypoint] ROBOMME_RENDER_BACKEND=$ROBOMME_RENDER_BACKEND"
echo "[entrypoint] SAPIEN_RENDER_DEVICE=${SAPIEN_RENDER_DEVICE:-<unset>}"
echo "[entrypoint] VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
exec "$@"
