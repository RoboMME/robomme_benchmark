# RoboMME OOD probing split

`ood_test` probes whether policies trained on regular RoboMME generalize to
configurations beyond its hard settings. It extends counts, sequence lengths,
scene layouts and object geometry while retaining each task's mechanism,
instruction template and success criteria. This is a diagnostic study of
extrapolation; failures can reflect memory, perception or control limitations.

## Split and configurations

The split contains **20 held-out episodes for each of 16 tasks (320 total)**,
all with difficulty `ood`. It provides testing episodes only: there is no OOD
training split or fine-tuning data. The original train, validation and test
splits remain unchanged. Demonstration videos and visual cues shown within an
episode remain part of the task's observations.

Each `config_ood` inherits the hard configuration in the
[task implementations](../src/robomme/robomme_env/), with these changes:

| Task | OOD configuration |
|---|---|
| BinFill | 13–18 cubes; place 6–8 cubes across 2–3 colors. |
| ButtonUnmask / VideoUnmask | Expand the sampling region from 0.4 × 0.4 m to 0.5 × 0.5 m; retain two pickups and the hard bin cap of 15. Actual bin counts depend on placement. |
| ButtonUnmaskSwap / VideoUnmaskSwap | 4–8 swaps; retain four bins and two pickups. |
| InsertPeg | Four pegs. |
| MoveCube | Cube dimensions scaled to 80%; peg shortened from 0.10 m to 0.08 m. |
| PatternLock | 9–16 nodes on a 6 × 6 grid. |
| PickHighlight | 8–12 cubes and 4–5 highlights in a 0.5 × 0.5 m sampling region. |
| PickXtimes | 6–10 pick-and-place repetitions. |
| RouteStick | Route length 8–12; retain backtracking. |
| StopCube | Stop on the 6th–10th crossing, with ten available crossings. |
| SwingXtimes | 4–8 back-and-forth repetitions. |
| VideoPlaceButton / VideoPlaceOrder | 2–4 swaps; retain four targets. |
| VideoRepick | 4–5 pickup/drop cycles; retain the hard clutter setting. |

Counts and geometry change; the requested operation and its success check do
not. In particular, the unmask tasks keep the original pickup goal, and the
video-placement tasks keep four targets.

## Episode verification

Candidate seeds are generated deterministically and exclude the original
train, validation and test seeds. An episode is retained only after both the
native expert and an end-effector-action replay complete it successfully.
The [episode metadata](../src/robomme/env_metadata/ood_test/) records each seed,
selection attempt, verification outcomes and demonstration-file checksum.

All 320 selected episodes passed this screen. Replay screening used a
1,300-step limit, so the set is conditioned on expert replay solvability within
that budget; it does not represent every possible configuration draw.

## Baseline evaluation

Evaluation of the released FrameSamp+Modul, SimpleMemVLA and PonderPounce
checkpoints is in progress. Final aggregate and per-task results will be added
after all three complete; no OOD fine-tuning is performed.

Fresh `ood_test` runs default to 1,500 steps. The ongoing exploratory evaluation
retains completed episodes at 1,300 steps and uses 1,500 for the remaining cases.
Its final report will disclose these budgets; a uniform-budget comparison should
use 1,500 steps for every method.

## Run the probe

Follow the environment and checkpoint instructions for
[FrameSamp+Modul](../scripts/plus_eval/robomme_vla/README.md),
[SimpleMemVLA](../scripts/plus_eval/simplememvla/README.md), or
[PonderPounce](../scripts/plus_eval/ponderpounce/README.md). From the RoboMME
repository root, select regular RoboMME's OOD split explicitly. For example:

```bash
CUDA_VISIBLE_DEVICES=0 "$SIMPLEMEMVLA_PYTHON" \
  scripts/plus_eval/simplememvla/eval_plus.py \
  --benchmark robomme --dataset ood_test \
  --checkpoint "$SIMPLEMEMVLA_CHECKPOINT" --attn-implementation sdpa \
  --output-dir runs/simplememvla-ood
```

The other two runners accept the same `--benchmark robomme --dataset ood_test`
flags. Omit task/episode filters to evaluate all 320 episodes. FrameSamp+Modul
also requires its policy server, as described in its guide. Use a fresh output
directory and retain the generated configuration, episode results and summary.
For subsequent comparisons, use the same seeds and a uniform step budget across
methods, and report per-task results alongside aggregate success.
