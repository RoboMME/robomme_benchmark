# Astra evaluators

- [Embodied and one-shot Astra](embodied/README.md)
- [3-tier Astra](three_tier/README.md)

Embodied and 3-tier Astra use `Zero-shot`; one-shot Astra uses
`Demo-only`, with no development rollouts. All follow the
[evaluation protocol](../../../../src/robomme_plus/README.md#evaluation-protocol).

Each directory contains its adapter and pinned upstream submodule.
`image_sampling.py` supplies the shared image-budget helper;
`requirements.txt` lists API-client dependencies. Local rollouts are linked under
each evaluator's ignored `runs/` directory.
