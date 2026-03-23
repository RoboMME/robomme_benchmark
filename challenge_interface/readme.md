# RoboMME Challenge @ CVPR 2026 FMEA Workshop

### [FMEA Workshop](https://foundation-models-meet-embodied-agents.github.io/cvpr2026/) | [Submission Example](https://github.com/RoboMME/robomme_policy_learning/tree/main?tab=readme-ov-file#robomme-challenge) | [Evaluate Your Policy](https://docs.google.com/forms/d/e/1FAIpQLSdMg_oEDU4kNsPM_dkuSsY6M7lTmtXQSyGzFO0MqwNmvvPGng/viewform?usp=publish-editor) 


The RoboMME challenge is part of the **Foundation Models Meet Embodied Agents** workshop at CVPR 2026.   
Evaluate your policy by filling this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdMg_oEDU4kNsPM_dkuSsY6M7lTmtXQSyGzFO0MqwNmvvPGng/viewform?usp=publish-editor).

All challenge-related files are stored in the current directory.
```
challenge_interface/
├── client.py               Used by organizers (websocket version)
├── client_http.py          Used by organizers (http version)
├── msgpack_numpy.py        Used by participants
├── policy.py               Used by participants (participants must modify the Policy class)
├── server.py               Used by participants (websocket version)
├── server_http.py          Used by participants (http version)
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
4. Submit your policy.

Submit the required information via this [link](https://docs.google.com/forms/d/e/1FAIpQLSdMg_oEDU4kNsPM_dkuSsY6M7lTmtXQSyGzFO0MqwNmvvPGng/viewform?usp=publish-editor) (deadline May 15). We provide three options for evaluating your policy.

**Choose one way to host your policy:**

**Option 1 (Recommended): via Docker**
- Participants build a Docker image packaging their policy.
- Organizers pull the image and host it on their machine.
- Submission example: [here](https://github.com/RoboMME/robomme_policy_learning/challenge_interface/docs/submission_guidance_docker.md) (MME-VLA model).

> In this option, we run your model on our servers. We can provide at most 80 GB of GPU memory; if your total system requires more, please choose Option 2.

**Option 2: via remote API**
- Participants deploy their policy on their own machine as a server with a public IP.
- Organizers query the host machine and run evaluation remotely.
- Submission example: [here](https://github.com/RoboMME/robomme_policy_learning/challenge_interface/docs/submission_guidance_remote.md) (MME-VLA model).

> This option may have unstable connections; we route to the closest node (US, mainland China, or Singapore) to reduce latency. Choose the `transport` type based on your setup; WebSocket with direct numeric IP access is recommended.

**Option 3: via GitHub Repo**
- Participants provide a link to their code repo.
- Organizers `git clone` the repo and follow the instructions to launch your policy server locally.
- Your server must follow the challenge interface (implement `Policy` with `infer()` and `reset()`).

> Like Option 1, your model runs on our servers (<= 80 GB GPU memory). If you need more GPU memory, or your system is very complex (e.g., multiple-model pipelines running simultaneously), choose Option 2 (remote API).


## Timeline
- **March–May 15**: Develop your policy and test your policy server.
- **May 15**: Deadline to submit your participant information.
- **May 15–May 22 (Phase 1 Validation)**: We verify stability & correctness for your Docker image / remote server / code repo.
  - If we find issues, we will contact you, and you can update your models/deployment during this window (up to **3 times**).
- **May 23**: Deadline to finalize your models and deployment.
- **May 23–June 2 (Phase 2 Full Evaluation)**: We evaluate on held-out episodes for teams that passed Phase 1.
- **June 3**: Winner announcement at the FMEA workshop.


## How should participants get started?

1. Get familiar with the [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark) and the [MME-VLA policy learning](https://github.com/RoboMME/robomme_policy_learning) repo.
2. Use the open-source [**val/test set**](https://github.com/RoboMME/robomme_benchmark/blob/0ac6cba0cbfe8ed1612dfbf37b7bedeb4b15a90c/scripts/evaluation.py#L83) as a testbed to develop and debug your models.
3. Wrap up your policy following the [challenge interface](https://github.com/RoboMME/robomme_benchmark/challenge_interface/policy.py) and test the policy server locally via `challenge_interface/scripts`.
4. [Submit your policy](https://docs.google.com/forms/d/e/1FAIpQLSdMg_oEDU4kNsPM_dkuSsY6M7lTmtXQSyGzFO0MqwNmvvPGng/viewform?usp=publish-editor).


## Acknowledgement

We greatly appreciate the Foundation Models Meet Embodied Agents workshop at CVPR 2026 for hosting our challenge!


## Contact

If you have anything to ask, please email robomme2026@gmail.com. We also provide wechat, discord channel, and google group link on our [website](https://robomme.github.io/challenge.html), you can join for latest news and discussion.
