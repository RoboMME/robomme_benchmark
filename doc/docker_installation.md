# Docker Installation for RoboMME

This guide sets up Docker and NVIDIA GPU support so you can build and run the RoboMME image.

## 1) Install Docker Engine 
> Skip this if you already installed docker.

Follow Docker’s official instructions for Ubuntu:
- Docker Engine install guide: `https://docs.docker.com/engine/install/ubuntu/`

After installing, make sure the service is running:

```bash
docker run --rm hello-world
```

## 2) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html) (GPU support)

> Skip this if you already installed nvidia-ctk.

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
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## 3) Build the RoboMME Docker image

From the repository root:

```bash
docker build -t robomme:cuda12.8 .
```
Enter the docker 
```bash
# download the robomme_data_h5 from https://huggingface.co/Yinpei/robomme_data_h5
export robomme_data_path=<robomme_data_h5_path>

docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  -v "$robomme_data_path:/app/data/robomme_data_h5:ro" \
  robomme:cuda12.8
```
Run sample scripts
```
uv run ./scripts/run_example.py
```


## 4) Others

To stop the docker
```bash
docker ps
docker stop <container_id_or_name>
```

To rebuild the docker image
```bash
docker build --no-cache -t robomme:cuda12.8 .
```