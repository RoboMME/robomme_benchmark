# RoboMME-Plus evaluation

Run commands from the repository root after the [simulator installation](../../readme.md#-installation).
Initialize only the submodule needed by your model.

| Method | Entry point | Inference |
| --- | --- | --- |
| [Embodied Astra](embodied-astra/README.md) | `embodied-astra/eval_plus.py` | Astra API; one simulator GPU |
| [One-shot Astra](embodied-astra/README.md#one-shot) | `embodied-astra/eval_plus_fs.py` | Astra API + one training example |
| [3-tier Astra](3-tier-astra/README.md) | `3-tier-astra/eval_plus.py` | Astra API + local monitor and VLA |
| [FrameSamp+Modul](robomme_vla/README.md) | `robomme_vla/eval_plus.py` | Local VLA server |
| [SimpleMemVLA](simplememvla/README.md) | `simplememvla/eval_plus.py` | Local model |
| [PonderPounce](ponderpounce/README.md) | `ponderpounce/eval_plus.py` | Local Ponder and Pounce |

## Layout and agentic baselines

Each method directory contains a pinned upstream Git submodule, its adapter and
README. ASPIRE/RAX integration patches are applied only to separate runtime copies.
Shared rollout code, `benchmark_io/`, and `results/` live outside method directories.
Large external logs/videos are linked through ignored `METHOD/runs/` directories.

| Agentic method | Frozen policy rerun | Development |
| --- | --- | --- |
| [ASPIRE](aspire/README.md) | `aspire/eval_plus.py` | Native coordinator, workers and verifier |
| [RAX](rax/README.md) | `rax/eval_plus.py` | Native exploration, rehearsal and submission |

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

## Protocol and outputs

[`evaluation.py`](evaluation.py) supplies the shared rollout, scoring and recording
protocol. Only environment `status == "success"` counts as success. Failures,
timeouts, model stops and errors remain in the requested-episode denominator.
The execution limit is `ceil(1.5 * oracle_policy_step) + 200` control steps;
reset demonstrations do not consume it. Test oracle annotations are never policy inputs.

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
