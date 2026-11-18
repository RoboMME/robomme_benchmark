# Agentic methods

- [Astra](astra/README.md): Embodied, one-shot and 3-tier evaluators.
- [ASPIRE](aspire/README.md): native skill-library development and frozen programs.
- [RAX](rax/README.md): native exploration, rehearsal and frozen bundles.

Each adapter keeps its pinned upstream submodule separate. ASPIRE/RAX include
copied `solutions/` for AfD R0 and AfI/AfD+I R1, R3, R5 and R10.
Development and rollout artifacts live in ignored `runs/` directories.
See the [shared protocol and setup](../benchmark_io/README.md).
