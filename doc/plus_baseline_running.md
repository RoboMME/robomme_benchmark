# Running model baselines

Use the [method instructions](../scripts/plus_eval/README.md) for dependencies and
launch commands. These shared options apply to Embodied Astra, one-shot Astra,
3-tier Astra, FrameSamp+Modul, SimpleMemVLA and PonderPounce. ASPIRE/RAX document
their own frozen-policy runners. Benchmark rules are in the
[evaluation guide](../src/robomme_plus/README.md#evaluation-protocol).

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

[`evaluation.py`](../scripts/plus_eval/evaluation.py) handles rollout execution, scoring and recording
according to the [benchmark evaluation rules](../src/robomme_plus/README.md#evaluate).

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
see the [data format and seed mapping](plus_training_data_format.md).
Use [`dataset_replay.py`](../scripts/plus_eval/dataset_replay.py) to replay training actions and save
front/wrist videos; see the [replay command](plus_training_data_format.md#replay).
Report zero-shot and few-shot results separately using the
[submission template](submission/model_example_plus.md).
