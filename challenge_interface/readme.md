# RoboMME Challenge @ CVPR 2026 FMEA Workshop

### [FMEA Workshop](https://foundation-models-meet-embodied-agents.github.io/cvpr2026/) | [Submission Portal]() | [Submission Example](https://github.com/RoboMME/robomme_policy_learning/tree/main?tab=readme-ov-file#robomme-challenge)


The RoboMME challenge is part of the **Foundation Models Meet Embodied Agents** workshop at CVPR 2026. Submissions are hosted on [EvalAI](https://eval.ai/).

All challenge-related files are stored in the current directory.
```
challenge_interface/
├── client.py               Used by organizers
├── msgpack_numpy.py        Used by participants
├── policy.py               Used by participants (participants must modify the Policy class)
├── server.py               Used by participants
└── scripts
    ├── deploy.py           Used by participants
    └── phase1_eval.py      Used by organizers
```

## What participants will do
1. Copy the `challenge_interface/` directory into your policy repository.
2. Implement the `Policy` [class](https://github.com/RoboMME/robomme_benchmark/blob/edc8e8008718d9bf545cfcc2dd3dc2264c903239/src/remote_evaluation/policy.py#L23) by overriding **`infer`** and **`reset`**, and adapt `challenge_interface/scripts/deploy.py` to your needs.
3. Verify your policy locally:
```
uv sync --group server

# terminal 0
uv run python -m  challenge_interface.scripts.deploy --port 8001
# terminal 1
uv run python -m  challenge_interface.scripts.phase1_eval --port 8001
```
4. Submit your policy

Go to the [EvalAI website]() to provide the required participant information (deadline May 15). We provide two options for hosting your policy:

Way 1: via Docker (recommended)  
Participants need to build a Docker image to package their policy. The organizers will pull the image and host it on their machine. We provide a submission example [here](https://github.com/RoboMME/robomme_policy_learning/tree/main?tab=readme-ov-file#robomme-challenge) using the MME-VLA model.

Way 2: via remote API  
Participants need to deploy their policy on their own machine as a server with a public IP. The organizers will query the hostname and evaluate remotely. This approach is used for more complicated systems that are difficult to package into a single Docker image.


## Timeline
- **March-May 15**: Develop your policy and test the policy server locally.
- **May 15**: Deadline to submit the required information on the EvalAI portal.
- **May 15–May 25**: Phase 1 partial evaluation (stability & correctness).
  - We verify that your Docker image / remote server is stable and runs as expected.
  - If we find issues, we will contact you. You are allowed to update your models or Docker images at most three times during this period.
  - We will choose the top 5-10 teams to move on, depending on the total number of participants.
- **May 26**: Deadline to finalize fixes for your models and deployment.
- **May 26–June 2**: Phase 2 full evaluation.
  - We evaluate on held-out episodes for selected teams.
- **June 3**: Winner announcement at the FMEA workshop.


## How should participants get started?

1. Get familiar with the [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark) and the [MME-VLA policy learning](https://github.com/RoboMME/robomme_policy_learning) repo.
2. Use the open-source [**val/test set**](https://github.com/RoboMME/robomme_benchmark/blob/0ac6cba0cbfe8ed1612dfbf37b7bedeb4b15a90c/scripts/evaluation.py#L83) as a testbed to develop and debug your models.
3. Wrap up your policy following the [challenge interface](challenge_interface/policy.py) and test the policy server locally via `challenge_interface/scripts`.
4. Submit on EvalAI.


## Acknowledgement

We greatly appreciate the Foundation Models Meet Embodied Agents workshop at CVPR 2026 for hosting our challenge, and we thank Figure AI for sponsoring it!
