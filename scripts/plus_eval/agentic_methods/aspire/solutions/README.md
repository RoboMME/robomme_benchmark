# Frozen RoboMME-Plus solutions

The archive contains nine configurations, 16 tasks each: `demo-only/R0`, `interaction-only/R1,R3,R5,R10`,
`demo+interaction/R1,R3,R5,R10`. [index.json](index.json) lists all 144 selections.

Use R10 for the main interaction results; R1/R3/R5 are optional.

Executable policy logic matches the full evaluation. Each task's `manifest.json`
records published file hashes, evaluated source hashes and policy provenance.
`publication_annotations` records setting-name edits to comments, docstrings and
documentation; Python logic was checked for AST equivalence. RAX also
includes the portable native runner settings. `runtime_policy.json` records the
shared execution limits. Development histories remain in the external run logs.

See [the method README](../README.md#setup-and-run) for evaluation commands.
