# Frozen RoboMME-Plus solutions

Nine configurations, 16 tasks each: `AfD/R0`, `AfI/R1,R3,R5,R10`,
`AfD+I/R1,R3,R5,R10`. [index.json](index.json) lists all 144 selections.

Files match the programs used for the full evaluation; each task's
`manifest.json` records evaluated hashes and policy provenance. RAX also
includes the portable native runner settings. `runtime_policy.json` records the
shared execution limits. Development histories remain in the external run logs.

See [the method README](../README.md#run-and-inspect) for evaluation commands.

Generated Python bytecode is excluded from the distributable files. When present
in an evaluated bundle, its original hash remains in `source_policy_files`;
`omitted_generated_files` records the exclusion. Policy source bytes are unchanged.
