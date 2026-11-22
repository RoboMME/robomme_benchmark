# Frozen RAX R10 policies

32 evaluated task policies: `interaction-only/R10/` and `hybrid/R10/`, each with
16 tasks. These are exact copies from the current `r10_20261121` evaluation.
`manifest.json` records source and published hashes; `index.json` lists the selections.
Only interpreter caches are omitted. Original development history is retained externally.

Run from the repository root:

```bash
python scripts/plus_eval/agentic_methods/rax/eval_plus.py \
  --setting hybrid --budget 10 --gpu 0 --output-dir "<RUN_ROOT>/rax-hybrid-R10"
```

Use `--setting interaction-only` for the other setting, `--tasks TASK --episodes 0 1`
for a subset, and `--artifacts logs` to omit videos. See the [adapter README](../README.md).
