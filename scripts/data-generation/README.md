# RoboMME No-Patch Dataset Generation

This directory contains the complete 16-task × 100-episode RoboMME dataset generation and validation workflow. A complete run produces 1,600 trajectories, uses the original train-metadata seed and difficulty once per trajectory, runs 20 workers, and is locked to physical GPU 0.

## Complete 16 × 100 Generation

Run the following command from the repository root. The output directory must be inside this repository and must not exist or must be empty.

~~~bash
cd /data/hongzefu/robomme_benchmark-restore-DataGen

env CUDA_VISIBLE_DEVICES=0 uv run --locked scripts/data-generation/generate_dataset.py \
  --output-dir artifacts/generated/no-patch-full-16x100 \
  --env all \
  --episodes 100 \
  --workers 20 \
  --gpus 0
~~~

The complete run generates `episode_0` through `episode_99` for each of the fixed 16 environments. `--gpus` must be exactly `0`; any other GPU or multi-GPU value is rejected.

## Post-Generation Calls

`generate_dataset.py` generates and merges the data, then calls three independent modules in the same Python process:

~~~text
generate_dataset.py
    ├── validate_generated_dataset_contract.py
    │   └── validate_generated_dataset_contract(...)
    ├── compare_joint_actions.py
    │   └── compare_joint_actions(...)
    └── write_generation_report.py
        └── write_generation_report(...)
~~~

1. The validator checks the generated HDF5 and metadata contract against the current train metadata and official reference data.
2. The comparator checks every generated `action/joint_action` element against `data/robomme_data_h5`; the maximum allowed absolute difference is `1e-8`.
3. The writer records generation provenance, validation results, the runtime environment, final file sizes, and SHA-256 manifests in JSON and Markdown.

The generator removes its `.workers` directory after completion or failure. A failed generation or audit still writes a failure report when the output directory exists.

## Revalidate an Existing Complete Output

The following command performs a read-only revalidation, preserves generation provenance from the existing report, reruns the validator and comparator, refreshes the SHA-256 manifest, and atomically replaces the central report:

~~~bash
env CUDA_VISIBLE_DEVICES=0 uv run --locked scripts/data-generation/write_generation_report.py \
  --output-dir artifacts/generated/no-patch-full-16x100 \
  --env all \
  --episodes 100 \
  --workers 20 \
  --gpus 0 \
  --prior-report scripts/data-generation/reports/generation_report.json \
  --max-abs-diff 1e-8
~~~

`--workers` and `--gpus` are recorded as provenance by the read-only command; it does not start generation workers.

## Final Artifacts

The complete generated-data directory contains exactly 16 HDF5 files and 16 metadata JSON files:

~~~text
artifacts/generated/no-patch-full-16x100/
├── record_dataset_<Task>.h5
└── record_dataset_<Task>_metadata.json
~~~

The authoritative reports are written only to:

~~~text
scripts/data-generation/reports/
├── generation_report.json
└── generation_report.md
~~~

The JSON report contains complete per-trajectory provenance, contract audits, the element-wise comparison, hardware and software details, and the file manifests. The Markdown report provides the corresponding English summary. Each generation or revalidation atomically replaces these two report files.
