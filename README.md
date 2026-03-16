---
title: RoboMME Oracle Planner
sdk: docker
app_port: 7860
---

This Space runs the RoboMME Gradio interface with the Docker SDK.

The container entrypoint is defined by the root `Dockerfile` and launches:

```bash
python3 gradio-web/main.py
```

`app_file` is intentionally not set here because this is a Docker Space; the application entrypoint comes from Docker `CMD`, while `app_port: 7860` is the external port published by the Space.

Local CPU Docker run:

```bash
docker build -t robomme-gradio:cpu .
docker run --rm -p 7860:7860 robomme-gradio:cpu
```
