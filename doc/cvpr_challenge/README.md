# RoboMME Challenge @ CVPR 2026 FMEA Workshop

### [FMEA Workshop](https://foundation-models-meet-embodied-agents.github.io/cvpr2026/) | [RoboMME Challenge]()

RoboMME is a challenge in the **Foundation Models Meet Embodied Agents** workshop at CVPR 2026. Submissions are hosted on eval.ai (challenge link: _TBD_).

## Submission format

We use **remote evaluation**: our server connects to the host and ports you provide.  
Each team submits one JSON file (see the `doc/cvpr_challenge/submit_example.json`) with:

| Field               | Type      | Description                                                                                               |
|---------------------|-----------|-----------------------------------------------------------------------------------------------------------|
| `model_name`        | string    | Name of your model.                                                                                       |
| `action_space`      | string    | One of `"joint_angle"`, `"ee_pose"`, or `"waypoint"`.                                                     |
| `use_depth`         | boolean   | Whether to return depth images.                                                                           |
| `use_camera_params` | boolean   | Whether to return camera intrinsics and extrinsics.                                                       |
| `github_link`       | string    | URL of the GitHub repository containing your code.                                                        |
| `paper_link`        | string    | URL of your report or paper (if available).                                                               |
| `email`             | string    | Contact email address.                                                                                    |
| `host`              | string    | Public host IP or domain where your policy server is running.                                            |
| `port`              | list[int] | Candidate ports we can use to reach your server. Provide **at least 2** ports to speed up evaluation.   |
| `api_key`           | string    | (Optional) API key required to query your endpoint, if applicable.                                       |

## Development

You can use the open-sourced [**val/test set**](https://github.com/RoboMME/robomme_benchmark/blob/0ac6cba0cbfe8ed1612dfbf37b7bedeb4b15a90c/scripts/evaluation.py#L83) as a testbed to develop and debug your models.

We provide two reference scripts:

- `scripts/challenge_eval_policy.py`: evaluation script that queries a remote policy server (used by the organizers).
- `scripts/challenge_serve_policy.py`: example script for hosting your policy server (used by participants).

Implement your policy in `src/remote_evaluation/policy.py` and host it via `scripts/challenge_serve_policy.py`.  
We provide an example [here](https://github.com/RoboMME/robomme_policy_learning/blob/cvpr26challenge/scripts/challenge_serve_policy.py) using our MME-VLA (framesamp + modulation) as the serving policy.

## Docker evaluation image

We also provide a Docker image path for the organizer-side evaluation client in [`scripts/challenge_eval_policy.py`](../../scripts/challenge_eval_policy.py). This image only runs the eval client and does **not** replace the existing remote policy server protocol.

### Host requirements

- Linux host with NVIDIA GPU access enabled for Docker.
- NVIDIA Driver installed on the host.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed so `docker run --gpus all ...` works.
- CPU-only execution is not the primary supported path for this image.

The container is configured for headless GPU rendering and writes evaluation videos to `/app/test_videos` inside the container.

### Build locally

```bash
docker build -t robomme-challenge:cvpr26challenge-docker .
```

### Run locally

The container entrypoint is `scripts/challenge_eval_policy.py`, so any extra arguments are forwarded directly to the eval script.

```bash
docker run --rm --gpus all \
  -v "$(pwd)/test_videos:/app/test_videos" \
  robomme-challenge:cvpr26challenge-docker \
  --host 141.212.115.116 \
  --port 8001 \
  --action_space joint_angle
```

Example with optional flags enabled:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/test_videos:/app/test_videos" \
  robomme-challenge:cvpr26challenge-docker \
  --host <policy-server-host> \
  --port 8001 \
  --action_space waypoint \
  --use_depth \
  --use_camera_params
```

Supported forwarded arguments remain the same as the Python script:

- `--host`
- `--port`
- `--action_space`
- `--use_depth`
- `--use_camera_params`

If the policy server is unreachable or a CLI argument is invalid, the error is emitted directly to the container logs via stdout/stderr.

### Manual publish to Docker Hub

Recommended repository naming:

```text
<dockerhub-username>/robomme-challenge
```

Example manual tagging and push flow:

```bash
docker tag robomme-challenge:cvpr26challenge-docker \
  <dockerhub-username>/robomme-challenge:cvpr26challenge-docker

docker push <dockerhub-username>/robomme-challenge:cvpr26challenge-docker
```

You can additionally publish a traceable version tag or `latest` manually if needed:

```bash
docker tag robomme-challenge:cvpr26challenge-docker \
  <dockerhub-username>/robomme-challenge:latest

docker push <dockerhub-username>/robomme-challenge:latest
```

## Evaluation 

For the official challenge, we use a **separate held-out test set**.  The full evaluation consists of **800 episodes** in total (50 per task). We only eval one model seed for simplicity in this challenge. 

Before running the full evaluation, we first run a **small subset of episodes** to check that your endpoint is stable and reaches at least **20% average success**.

If API calls fail (e.g., connection errors or repeated timeouts), we will contact you by email and ask you to update or fix your endpoint.

If your policy underperforms (<20%), we will also notify you that we will not run the full evaluation. Please use the open-sourced test/val set to evaluate your policy before submitting.

After evaluation, we will update the **challenge leaderboard** on eval.ai with your results.


## Timeline
...

## Acknownledgement
...
