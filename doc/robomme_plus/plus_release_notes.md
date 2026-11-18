# RoboMME-Plus final release

- Based on upstream main `582ab30`, retaining regular RoboMME's `ood_test` split.
- Current task code, perceptual/mechanism variations, 50 training records and
  800 test records. Only the 55 referenced MoveCube recordings are packaged.
- Setup retries are bounded; budgeted development counts every simulator start.
- Each baseline keeps its core submodule separate from its adapter. ASPIRE/RAX
  include explicit integration patches and 144 frozen task solutions each.
- Shared result summaries and external artifact links live under `scripts/plus_eval`.
  Logs, videos and provider weights are stored outside Git.

See the [evaluation protocol](../../src/robomme_plus/README.md#evaluation-protocol),
[baseline results and running instructions](robomme_plus_baseline_results.md),
[compliance audit skill](../skills/audit_robomme_plus_compliance.md), and
[data/source pins](plus_release_manifest.json).
