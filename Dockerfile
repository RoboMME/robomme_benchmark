FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libvulkan1 \
    vulkan-tools \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=automatic \
    PYTHONPATH=/app/src \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    SAPIEN_RENDER_DEVICE=cuda \
    XDG_RUNTIME_DIR=/tmp/runtime-root

RUN mkdir -p "${XDG_RUNTIME_DIR}" /runs

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group server --no-install-project --python 3.11

COPY src ./src
COPY scripts ./scripts
COPY readme.md ./

RUN uv sync --frozen --no-dev --group server --python 3.11

CMD ["bash"]

