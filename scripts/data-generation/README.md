# RoboMME No-Patch Dataset Generation

This README covers only how to run the complete 16 × 9 generation, the three independent modules called by the generator, the main parameters, and the locations of the final artifacts.

## Complete 16 × 9 Generation

Run this from the repository root. The output directory must be inside the repository and must not exist or must be empty before the run.

~~~bash
cd /data/hongzefu/robomme_benchmark-restore-DataGen

uv run --locked scripts/data-generation-v2-noPatch/generate_dataset.py \
  --output-dir artifacts/generated/no-patch-full-16x9-<run-id> \
  --env all \
  --episodes 9 \
  --workers 9 \
  --gpus 0,1
~~~

This generates episode_0 through episode_8 for each of the fixed 16 environments, for a total of 144 trajectories. Replace <run-id> with a new name to avoid reusing an existing output directory.

## Three Post-Generation Calls

generate_dataset.py generates and merges the data. After each task's HDF5 and metadata are written, it directly calls the following three independent modules in the same Python process:

~~~text
generate_dataset.py
    ├── validate_generated_dataset_contract.py
    │   └── validate_generated_dataset_contract(...)
    ├── compare_joint_actions.py
    │   └── compare_joint_actions(...)
    └── write_generation_report.py
        └── write_generation_report(...)
~~~

1. The validator checks the fixed contract for generated data, current train metadata, and official reference data.
2. The comparator compares generated action/joint_action values element by element with data/robomme_data_h5. The default maximum absolute-difference threshold is 1e-8.
3. The writer summarizes the preceding results and writes JSON and Markdown reports.

All three are in-process function calls and do not call the legacy scripts/data-generation/ directory. If generation or post-generation auditing fails, the writer still writes a failure report whenever the output directory exists.

## Parameters

The complete 16 × 9 command uses the following parameters:

| Parameter | Value for this complete run | Meaning |
| --- | --- | --- |
| --output-dir | artifacts/generated/no-patch-full-16x9-<run-id> | Generated-data directory; it must be inside the repository and must not exist or must be empty. |
| --env | all | All 16 fixed environments. |
| --episodes | 9 | Generate episode_0 through episode_8 for every environment. |
| --workers | 9 | Run at most nine generation workers concurrently. |
| --gpus | 0,1 | Assign workers to GPU 0 and GPU 1 in round-robin order. |

The generator always uses the original seed and difficulty from the current train metadata and attempts each episode only once.

## Recompare an Existing Complete Output

Existing data does not need to be generated again. The following command read-only revalidates an existing 16 × 9 output, reruns the validator and comparator, and refreshes the central report:

~~~bash
uv run --locked scripts/data-generation-v2-noPatch/write_generation_report.py \
  --output-dir artifacts/generated/no-patch-full-16x9 \
  --env all \
  --episodes 9 \
  --workers 9 \
  --gpus 0,1 \
  --max-abs-diff 1e-8
~~~

--workers and --gpus are recorded only as report parameters by this revalidation command; it does not start workers.

## Final Artifacts

The complete run writes generated data to --output-dir:

~~~text
artifacts/generated/no-patch-full-16x9-<run-id>/
├── record_dataset_<Task>.h5
└── record_dataset_<Task>_metadata.json
~~~

A complete 16 × 9 run produces 16 HDF5 files and 16 metadata JSON files. Temporary worker files are removed at the end.

The authoritative comparison reports are always written to this directory, rather than to the generated-data output directory:

~~~text
scripts/data-generation-v2-noPatch/reports/
├── no_patch_generation_report.json
└── no_patch_generation_report.md
~~~

JSON retains the complete results; Markdown provides a summary. This directory retains only the latest report from a generation or revalidation run, and each new run overwrites the same JSON and Markdown files.

The current complete 16 × 9 acceptance result is: official and generated final completion are both 144/144; the comparison covers 73,907 joint vectors and 591,256 elements; the maximum absolute difference is 5.661269342205344e-09, which is below 1e-8.
