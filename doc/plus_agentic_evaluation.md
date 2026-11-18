# RoboMME-Plus adaptation protocol

The release has **50 training episodes** (three per task, five for MoveCube) and
**800 test episodes** (50 per task). Training and test seeds are disjoint. Current
metadata enables both perceptual and mechanism variation. The original RoboMME
[`ood_test` split](ood_split.md) is a separate evaluation.

| Setting | Released training demos | Simulator starts per task |
| --- | --- | --- |
| ZS: direct transfer | No | 0 |
| AfD: demonstrations only | Yes | 0 |
| AfI: interaction only | No | R1, R3, R5 or R10 |
| AfD+I: demonstrations and interaction | Yes | R1, R3, R5 or R10 |

Released demos can be reused without limit. An episode's **conditioning video**
is part of its observation, including in ZS/AfI; it is distinct from an expert
training demonstration. Only the released training seeds may be used to adapt.
Every simulator start counts, including failed setup, previews, debugging,
verification, retries and frozen-bundle rehearsal. Training disables automatic
reset retries so each start must reserve another budget slot.

Freeze weights, programs, prompts, tools and libraries before testing. Reset
adaptive memory between test episodes. Test feedback cannot revise a policy or
select among candidates. Task source, oracle queries and privileged object state
are forbidden inputs. Report test-time optimization separately.

## ASPIRE and RAX

Both retain their original LIBERO workflows and use Opus 5.5 through AWS Bedrock
with high reasoning effort. ASPIRE runs a multi-task coordinator with per-task
workers and a shared verified skill library. RAX develops and submits a bundle
separately for each task. AfD/AfD+I receive released RGB demonstration sequences;
live interaction uses the native RGB-D, calibration and robot-state interfaces.
Neither implementation uses our separate Code-as-Policies baseline.

R1/R3/R5/R10 are **independent development runs**, not snapshots taken from R10.
Each task gets up to 150 minutes, followed by a read-only wrap-up. All verification
and selection interactions share R. AfD allows offline submission without a
simulator rehearsal. RAX's optional Gemini service is disabled.

Frozen evaluation runs the selected Python policies and local providers, with no
frontier-model API calls. The simulator enforces each episode's control-step
limit. Setup may retry up to three times with identical configuration; policy
execution is never retried automatically. Videos sent across worker boundaries
use at most 512 timestamps, both cameras, and a separate initial live frame.
Provider, RPC, no-progress and controller wall-time limits prevent unbounded waits.

## Reporting and reproduction

Report successes / **800**, per-task success / **50**, variation settings, R,
actual development attempts and simulator steps per task, and development/evaluation
API costs separately. Failures, timeouts, errors and absent policies stay in the
denominator. Usage comes from provider token records; dollar amounts are estimates
using the recorded rate card, not invoices. Missing usage is not zero cost.

Full evaluation media is retained for R0 and R10. R1/R3/R5 retain results and logs;
all development code, commands, transcripts and training rollouts are retained.

- [Setup, queue and artifact layout](../scripts/plus_eval/benchmark_io/README.md)
- [ASPIRE adapter and frozen solutions](../scripts/plus_eval/aspire/README.md)
- [RAX adapter and frozen solutions](../scripts/plus_eval/rax/README.md)
- [Results](robomme_plus_baseline_results.md)
- [Final training data format](plus_training_data_format.md)
