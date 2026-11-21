# Agentic methods

- [Astra](astra/README.md): Embodied, one-shot and 3-tier evaluators.
- [ASPIRE](aspire/README.md): native skill-library development and frozen programs.
- [RAX](rax/README.md): native exploration, rehearsal and frozen bundles.

Each adapter keeps its pinned upstream submodule separate. ASPIRE/RAX include
copied `solutions/` for `demo-only` (0 attempts), `interaction-only` and
`demo&interaction` (at most 10 attempts). R1/R3/R5 results are optional.
Development and rollout artifacts live in ignored `runs/` directories.
Follow the [evaluation protocol](../../../src/robomme_plus/README.md#evaluation-protocol)
and [shared ASPIRE/RAX setup](../benchmark_io/README.md).
