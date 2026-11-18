# Astra evaluators

- [Embodied and one-shot Astra](embodied/README.md)
- [3-tier Astra](three_tier/README.md)

Each directory contains its adapter and pinned upstream submodule.
`image_sampling.py` supplies the shared image-budget helper;
`requirements.txt` lists API-client dependencies. Local rollouts are linked under
each evaluator's ignored `runs/` directory.
