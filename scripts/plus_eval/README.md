# Running RoboMME-Plus baselines

This directory contains setup and launch instructions for the baseline adapters.
For evaluation settings, permitted training experience and benchmark rules, start
with the [evaluation guide](../../src/robomme_plus/README.md#evaluation-protocol).

Run commands from the repository root after the [simulator installation](../../readme.md#-installation).
Initialize only the submodule needed by your model.

| Method | Entry point | Inference |
| --- | --- | --- |
| [Embodied Astra](agentic_methods/astra/embodied/README.md) | `agentic_methods/astra/embodied/eval_plus.py` | Astra API; one simulator GPU |
| [One-shot Astra](agentic_methods/astra/embodied/README.md#one-shot) | `agentic_methods/astra/embodied/eval_plus_fs.py` | Astra API + one training example |
| [3-tier Astra](agentic_methods/astra/three_tier/README.md) | `agentic_methods/astra/three_tier/eval_plus.py` | Astra API + local monitor and VLA |
| [FrameSamp+Modul](non_agentic_methods/framesamp_modul/README.md) | `non_agentic_methods/framesamp_modul/eval_plus.py` | Local VLA server |
| [SimpleMemVLA](non_agentic_methods/simplememvla/README.md) | `non_agentic_methods/simplememvla/eval_plus.py` | Local model |
| [PonderPounce](non_agentic_methods/ponderpounce/README.md) | `non_agentic_methods/ponderpounce/eval_plus.py` | Local Ponder and Pounce |

## Layout

Each method directory contains a pinned upstream Git submodule, its adapter and
README. ASPIRE/RAX integration patches are applied only to separate runtime copies.
`agentic_methods/` groups Astra, ASPIRE and RAX. `non_agentic_methods/` groups
FrameSamp+Modul, SimpleMemVLA and PonderPounce. Method helpers stay beside their
adapters; shared rollout code, `benchmark_io/`, and `results/` stay at this level.
Large external logs/videos are linked through ignored `METHOD/runs/` directories.
Run artifacts are excluded from Git. ASPIRE/RAX `solutions/` are tracked copies of
frozen programs, not links to experiment directories.

| Agentic method | Frozen policy rerun | Development |
| --- | --- | --- |
| [ASPIRE](agentic_methods/aspire/README.md) | `agentic_methods/aspire/eval_plus.py` | Native coordinator, workers and verifier |
| [RAX](agentic_methods/rax/README.md) | `agentic_methods/rax/eval_plus.py` | Native exploration, rehearsal and submission |

See the [adaptation protocol](../../doc/plus_agentic_evaluation.md),
[shared setup](benchmark_io/README.md), and [overall results](results/README.md).
The run options below apply to the six model adapters above; agentic adapters
have their own documented options.

## Run options

All adapters accept `--tasks BinFill --episodes 0 --output-dir runs/METHOD`.
Omit task and episode filters for all **800 test episodes**, evaluated sequentially.
Use `--help` for model-specific options.

| Option | Behavior |
| --- | --- |
| `--resume` | Continue missing/interrupted episodes; retry errors and incompatible saved outcomes |
| `--retry-errors` | Retry selected saved errors only |
| `--rerun` | Replace every selected episode |
| `--fail-fast` | Stop on the first episode error |

Use one continuation mode at a time. Keep inference settings unchanged when
continuing; use a new output directory for another model, checkpoint or prompt.
Replacements archive old episodes under `.history/`. Only one evaluator may
write to an output directory at a time.

## Outputs

[`evaluation.py`](evaluation.py) handles rollout execution, scoring and recording
according to the [benchmark evaluation rules](../../src/robomme_plus/README.md#evaluate).

Each run saves `config.json`, `summary.json` and `run.log`. Episode directories
contain `result.json`, `metadata.json`, `episode.mp4`, `actions.npz` and policy-call
artifacts. Results include task failure reasons, timing, usage and stop source.
Credentials and private paths are redacted from recorded artifacts.
Rebuild summaries without inference with:

```bash
.venv/bin/python scripts/plus_eval/evaluation.py summarize runs/METHOD
```

Zero-shot evaluation needs no training data. One-shot Astra uses the
[H5 training dataset](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5);
see the [data format and seed mapping](../../doc/plus_training_data_format.md).
Use [`dataset_replay.py`](dataset_replay.py) to replay training actions and save
front/wrist videos; see the [replay command](../../doc/plus_training_data_format.md#replay).
Report zero-shot and few-shot results separately using the
[submission template](../../doc/submission/model_example_plus.md).
