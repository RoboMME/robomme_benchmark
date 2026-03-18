# Docker Installation for RoboMME (Ubuntu)

This guide sets up Docker and (optionally) NVIDIA GPU support so you can build and run the RoboMME image.

## 1) Install Docker Engine

Follow Docker’s official instructions for Ubuntu:
- Docker Engine install guide: `https://docs.docker.com/engine/install/ubuntu/`

After installing, make sure the service is running:

```bash
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

### Optional: run Docker without `sudo`

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world
```

## 2) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html) (GPU support)

Install the toolkit (Ubuntu):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure Docker to use the NVIDIA runtime and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 3) Build the RoboMME Docker image

From the repository root:

```bash
docker build -t robomme -f Dockerfile .
```

Quick test
```bash
docker run --rm -it --gpus all robomme nvidia-smi
```