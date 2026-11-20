# Agentic methods

- [Astra](astra/README.md): Embodied, one-shot and 3-tier evaluators.
- [ASPIRE](aspire/README.md): native skill-library development and frozen programs.
- [RAX](rax/README.md): native exploration, rehearsal and frozen bundles.

Each adapter keeps its pinned upstream submodule separate. ASPIRE/RAX include
copied `solutions/` for demonstrations-only (0 attempts) and both interaction
settings (5 or 10 attempts). The directory/CLI keys remain `AfD`, `AfI` and
`AfD+I`. Archived R1/R3 solutions are retained for reproduction.
Development and rollout artifacts live in ignored `runs/` directories.
Follow the [evaluation protocol](../../../src/robomme_plus/README.md#evaluation-protocol)
and [shared ASPIRE/RAX setup](../benchmark_io/README.md).
