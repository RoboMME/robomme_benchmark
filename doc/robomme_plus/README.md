# RoboMME-Plus: start here

RoboMME-Plus measures robot memory generalization and adaptation across 16 tasks,
with 50 released training demonstrations and 800 fixed test episodes.

1. **Understand the benchmark and choose a setting.** Read the
   [overview and evaluation protocol](../../src/robomme_plus/README.md#evaluation-protocol):
   zero-shot, demo-only, interaction-only or demo+interaction. Interaction settings
   allow at most **10 Plus development starts per task across all training seeds
   combined**. Demonstrations are offline recordings; simulator replay counts as
   interaction and is forbidden in demo-only.
2. **Prepare your environment and inputs.** Follow the
   [installation instructions](../../readme.md#-installation) and
   [training-data guide](plus_training_data_format.md). Record the benchmark and
   dataset revisions, and expose only the resources permitted by your setting.
3. **Connect and audit your agentic harness before development.** Follow the
   [development interface](agentic_development.md) to integrate your own method,
   select training seeds and share one persistent rollout counter across workers.
   Then use the [harness-audit procedure and example prompt](auditing.md#1-review-the-harness).
4. **Develop and preserve the complete history.** Keep every solution revision
   and its associated prompts, responses, commands/tool outputs, rollout records
   and API costs, including unsuccessful and interrupted work. Link each rollout
   to its saved code revision. Freeze the final policy, prompts, tools and harness
   before testing.
5. **Audit the development record.** Follow the
   [development-log audit procedure and example prompt](auditing.md#2-review-the-completed-development-record).
   The two audit skills instruct a reviewer to inspect code and evidence; they do
   not automatically certify compliance.
6. **Evaluate and submit.** Run the frozen method on
   [all 800 test episodes](../../src/robomme_plus/README.md#evaluate), then use the
   [submission template](submission/model_example_plus.md) to share results,
   code, development history and both audit reports.

The harness and development-log audits apply to agentic development. To run an
existing baseline, use its [setup and running instructions](../../scripts/plus_eval/README.md).

## Reference

- [Baseline results](robomme_plus_baseline_results.md)
- [ASPIRE/RAX setup and development](../../scripts/plus_eval/benchmark_io/README.md)
- Audit skills: [harness](../skills/audit_robomme_plus_harness.md) · [development logs](../skills/audit_robomme_plus_development.md)
- Submission examples: [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md>) · [RAX](<submission/rax-plus (eval_by_RoboMME_team).md>)
- [Release manifest: data and source revisions](plus_release_manifest.json)
