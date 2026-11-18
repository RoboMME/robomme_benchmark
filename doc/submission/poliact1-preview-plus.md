# PoliAct1 Preview

**QJ Robot** · [Code and frozen release](https://github.com/XinSu-QJRobots/PoliAct1-Preview/tree/v1.0.0-eval.7) · [Results](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/results/README.md) · [Episode execution records](https://github.com/XinSu-QJRobots/PoliAct1-Preview/tree/v1.0.0-eval.7/results/trajectories)

The code and result repository is currently private and shared with designated reviewers. Repository access is required to open these links; a public release may follow.

## Introduction

PoliAct1 Preview is a code-policy framework for memory-dependent robotic manipulation. A multimodal coding model uses public RGB-D observations and interaction history to operate a persistent, sandboxed Python kernel. Task memory combines frozen experience summaries with executable perception and manipulation skills, while the RoboMME-Plus simulator supplies observations, control-step accounting and terminal success labels. The evaluated release contains one selected strategy per task, its required skills and adapters, and episode-level execution records.

## Results

**Track:** Agentic Track.

**Evaluated by:** QJ Robot. **Results snapshot:** 2026-11-18.

**Setting:** FS — task-specific prompt/memory and skill-code adaptation, without model-weight updates.

**API Cost — Agentic Track:** **Estimated Qwen list-price equivalent: US$2,018.11 total; US$2.52/episode** across the reported 800 episodes, including unsuccessful episodes. This equals **CNY 13,455.14**, using `qwen3.8-max` [China list prices](https://help.aliyun.com/zh/model-studio/model-pricing) checked 2026-11-18: CNY 12/million input and CNY 36/million output tokens. Conversion uses [2026-11-18 reference FX](https://api.frankfurter.dev/v1/2026-11-18?base=USD&symbols=CNY), US$1 = CNY 6.6672. Recorded usage covers 793/800 episodes: 1,040,659,635 input tokens (including 984,161,664 cache-read tokens) and 26,867,293 output tokens. Cached input is counted once at the full input price; no cache discount is assumed. **Actual combined-model spending is incomplete:** GPT visual-plan review and requests without usage records are not priced. See [cost calculation](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/docs/COST.md).

RoboMME-Plus TEST comprises **16 tasks × 50 episodes = 800 episodes**, using **`waypoint`** actions. Only simulator success counts. Failures, model stops, timeouts, initialization errors and the abandoned episode remain in the denominator. All recorded execution-step counts are within the supplied `ceil(1.5 * oracle_policy_step) + 200` budget.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 26/50 | 52.00 |
| Counting | PickXtimes | 43/50 | 86.00 |
| Counting | SwingXtimes | 48/50 | 96.00 |
| Counting | StopCube | 16/50 | 32.00 |
| Permanence | VideoUnmask | 38/50 | 76.00 |
| Permanence | VideoUnmaskSwap | 40/50 | 80.00 |
| Permanence | ButtonUnmask | 37/50 | 74.00 |
| Permanence | ButtonUnmaskSwap | 13/50 | 26.00 |
| Reference | PickHighlight | 32/50 | 64.00 |
| Reference | VideoRepick | 47/50 | 94.00 |
| Reference | VideoPlaceButton | 48/50 | 96.00 |
| Reference | VideoPlaceOrder | 42/50 | 84.00 |
| Imitation | MoveCube | 1/50 | 2.00 |
| Imitation | InsertPeg | 18/50 | 36.00 |
| Imitation | PatternLock | 33/50 | 66.00 |
| Imitation | RouteStick | 10/50 | 20.00 |
| **Overall** | **All tasks** | **492/800** | **61.50** |

### Training Details

- **Adaptation:** official RoboMME-Plus TRAIN examples and interaction feedback were used to refine prompts, task memories and executable skills. No model weights were updated. Final strategy revisions, memory hashes and available TRAIN validation results are in the [task catalog](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/src/poliact1/catalog.json); these validation rows are not a complete count of historical development interactions.
- **Frozen strategies:** each task loads its selected memory and task adapter from the linked release. Strategies are not updated by the evaluation entry point. See the [strategy files](https://github.com/XinSu-QJRobots/PoliAct1-Preview/tree/v1.0.0-eval.7/src/poliact1/strategies) and [method description](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/docs/METHOD.md) for selection scope and exceptions.
- **Policy inputs:** public front/wrist RGB-D, camera calibration, robot state, task instruction and task-provided demonstration/history observations. Oracle scoring and subtask diagnostics are not policy inputs.
- **MoveCube:** the evaluated implementation uses externally generated GPT visual-plan files, reusing them only when conditioning/current images match. Unmatched images require an external plan provider; the release does not autonomously call a GPT API for those plans. See [the visual-plan interface](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/docs/EVALUATION.md#movecube-visual-plans). The very low MoveCube success rate (1/50, 2.00%) indicates overfitting to the training set; this task-specific strategy may therefore be omitted from optional reproduction runs, while its recorded outcomes remain included in the reported 800-episode aggregate.

### Reproduce

- **Code:** `v1.0.0-eval.7`, commit `2c0c096dc48c` in the linked reviewer-access repository.
- **Benchmark:** `RoboMME/robomme_benchmark@5166f74b735c6b76f077a89426ef2cc4f8839b57`; ManiSkill revision `07be6fbc66350ddca200abfb0a11b692f078f7fd`.
- **Model:** API alias `qwen3.8-max`, reasoning `max`. An immutable provider checkpoint ID is unavailable. Maximum response tokens: 16,384, or 32,768 for PatternLock, InsertPeg and RouteStick. Reviewers must provide their own compatible Qwen API endpoint and key; neither a hosted API service nor credentials are distributed.
- **Seed:** fixed benchmark episode seeds; no additional inference seed was set.
- **Hardware:** NVIDIA A800-SXM4-80GB, PhysX CPU simulation and NVIDIA Vulkan GPU rendering; Python 3.11 on Linux.
- **History and context:** public observation history is processed by the frozen [task adapters](https://github.com/XinSu-QJRobots/PoliAct1-Preview/tree/v1.0.0-eval.7/src/poliact1/backends). Conditioning samplers use task-specific frame sampling and image construction rather than one uniform image budget; their settings are fixed in that release. The API service determines the effective model context limit, which is not recorded as an immutable checkpoint property.
- **Instructions:** [installation, API configuration and evaluation guide](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/docs/EVALUATION.md). Existing RoboMME installations can be used through `ROBOMME_ROOT` and `POLIACT_SIM_PYTHON`.

```bash
git clone --depth 1 --branch v1.0.0-eval.7 https://github.com/XinSu-QJRobots/PoliAct1-Preview.git
cd PoliAct1-Preview
uv sync --frozen --extra report

# Point to the existing, pinned RoboMME installation and its interpreter.
export ROBOMME_ROOT=/path/to/robomme_benchmark
export POLIACT_SIM_PYTHON=/path/to/robomme/simulator-env/bin/python

# Supply your own compatible Qwen endpoint and credentials.
export OPENAI_BASE_URL=https://your-provider.example/v1
read -rs -p 'Your Qwen API key: ' OPENAI_API_KEY
export OPENAI_API_KEY

uv run --no-sync poliact1 verify
uv run --no-sync python tools/preflight.py
uv run --no-sync poliact1 doctor
uv run --no-sync poliact1 eval --tasks all --dataset test --episodes all \
  --gpus 0,1,2,3 --lanes-per-gpu 2 --videos --output runs/plus-full800
```

GitHub access must be authenticated before cloning the private repository. The full-suite command includes MoveCube and therefore retains its external visual-plan dependency. The guide also provides a separate 15-task command that excludes MoveCube; that subset does not replace the 800-episode table.

**Result artifacts:** [per-task scores](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/results/README.md), [800 episode outcomes](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/results/episodes.json), [trajectory index](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/results/trajectory-index.json) and [execution JSON files](https://github.com/XinSu-QJRobots/PoliAct1-Preview/tree/v1.0.0-eval.7/results/trajectories). The records contain 41,143 commands: 793 available command sequences, one partial abandoned sequence and six reset failures with no actions. Complete model conversations and original RGB-D/video binaries are not included in this compact repository.

### Released Checkpoints

No new learned checkpoint is produced. The release provides [16 frozen task memories](https://github.com/XinSu-QJRobots/PoliAct1-Preview/tree/v1.0.0-eval.7/src/poliact1/strategies), [task/model configuration](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/src/poliact1/catalog.json) and the required executable skills. The underlying model is accessed through the API alias identified above.

### Citations

- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).
- [RoboMME benchmark and RoboMME-Plus submission template](https://github.com/RoboMME/robomme_benchmark/blob/main/doc/submission/model_example_plus.md).
- [PoliAct1 Preview implementation credits](https://github.com/XinSu-QJRobots/PoliAct1-Preview/blob/v1.0.0-eval.7/ACKNOWLEDGEMENTS.md).
