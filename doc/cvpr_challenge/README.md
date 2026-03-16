# RoboMME Challenge @ CVPR 2026 FMEA Workshop

### [FMEA Workshop](https://foundation-models-meet-embodied-agents.github.io/cvpr2026/) | [RoboMME Challenge]()

RoboMME is a challenge in the **Foundation Models Meet Embodied Agents** workshop at CVPR 2026. Submissions are hosted on eval.ai (challenge link: _TBD_).

## Submission format

We use **remote evaluation**: our server connects to the host and ports you provide.  
Each team submits one JSON file (see the [example](doc/cvpr_challenge/submit_example.json)) with:

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

## Evaluation 

For the official challenge, we use a **separate held-out test set**.  The full evaluation consists of **800 episodes** in total (50 per task).

Before running the full evaluation, we first run a **small subset of episodes** to check that your endpoint is stable and reaches at least **20% average success**.

If API calls fail (e.g., connection errors or repeated timeouts), we will contact you by email and ask you to update or fix your endpoint.

If your policy underperforms (<20%), we will also notify you that we will not run the full evaluation. Please use the open-sourced test/val set to evaluate your policy before submitting.

After evaluation, we will update the **challenge leaderboard** on eval.ai with your results.


## Timeline
...

## Acknownledgement
...
