FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    gnupg \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libvulkan1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3 \
    && rm -f /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OMP_NUM_THREADS=1 \
    PORT=7860

WORKDIR /home/user/app
RUN mkdir -p /home/user/app/temp_demos \
    && chown -R user:user /home/user

COPY --chown=user:user requirements.txt ./

USER user

RUN python3 -m pip install --user --upgrade pip setuptools wheel \
    && python3 -m pip install --user -r requirements.txt

COPY --chown=user:user docker-entrypoint.sh ./
COPY --chown=user:user gradio-web ./gradio-web
COPY --chown=user:user src ./src

RUN chmod +x /home/user/app/docker-entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["./docker-entrypoint.sh"]

CMD ["python3", "gradio-web/main.py"]
