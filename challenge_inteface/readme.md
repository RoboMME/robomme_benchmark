# RoboMME Challenge @ CVPR 2026 FMEA Workshop

### [FMEA Workshop](https://foundation-models-meet-embodied-agents.github.io/cvpr2026/) | [MME-VLA Submission Example]()


RoboMME is a challenge in the **Foundation Models Meet Embodied Agents** workshop at CVPR 2026. Submissions are hosted on eval.ai (challenge [link]()).

All challenge-related files are stored in the [`challenge_inteface`]() directory.

```
challenge_inteface/
├── client.py               Used by organizers
├── msgpack_numpy.py        Used by participants
├── policy.py               Used by participants (participants must modify the Policy class)
├── server.py               Used by participants
└── scripts
    ├── deploy.py           Used by participants
    └── phase1_eval.py      Used by organizers
```

## What participants will do
1. Copy the `challenge_inteface/` directory into your policy repository.
2. Implement the `Policy` [class](https://github.com/RoboMME/robomme_benchmark/blob/edc8e8008718d9bf545cfcc2dd3dc2264c903239/src/remote_evaluation/policy.py#L23) by overriding **`infer`** and **`reset`**, and adapt `challenge_inteface/scripts/deploy.py` to your needs.
3. Verify your policy locally:
```
# terminal 0
uv run python -m  challenge_inteface.scripts.deploy  --port 8001
# terminal 1
uv run python -m  challenge_inteface.scripts.phase1_eval  --port 8001
```
4. Submit your policy

Go to the EvalAI website to provide the required participant information. We provide two options for hosting your policy:

Way 1: via Docker (recommended)
Participants need to build a Docker image to wrap their policy. The organizers will pull the image and host it on their machine. We provide an [example]() Dockerfile for an MME-VLA model.

Way 2: via remote API
Participants need to deploy their policy on their machine as a server with a public IP. The organizers will query the hostname and evaluate remotely. This approach is used for more complicated systems that are difficult to package into a single Docker image.


## Timeline
- Before May 15: participants will develop their policies and test the policy server locally.
- May 15–25: Phase 1 evaluation. This phase primarily checks whether the Docker image or remote host is stable and correct. We will contact you if we find any issues.
- May 26–June 2: Phase 2 evaluation.
- June 3: winner announcement at the FMEA workshop.


## How should participants get started?

1. Get familiar to [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark) and [MME-VLA policy learning](https://github.com/RoboMME/robomme_policy_learning) repo.
2. Use the open-sourced [**val/test set**](https://github.com/RoboMME/robomme_benchmark/blob/0ac6cba0cbfe8ed1612dfbf37b7bedeb4b15a90c/scripts/evaluation.py#L83) as a testbed to develop and debug your models.
3. Wrap up your policy into the [challenge interface](challenge_inteface/policy.py) and test the policy server via `challenge_inteface/scripts`.


## Acknowledgement

We greatly appreciate the Foundation Models Meet Embodied Agents workshop at CVPR 2026 for hosting our challenge!
