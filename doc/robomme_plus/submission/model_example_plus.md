# RoboMME-Plus submission templates

Copy the template for each setting you report. Replace bracketed placeholders
and fill in the results table. Both Non-Agentic and Agentic tracks can use any
setting. For interaction settings, report 5 and 10 rollouts per task in the same table;
mark unevaluated columns N/R.

All 50 released demonstrations are available for **Few-shot demonstrations** and **Few-shot
demonstrations+interactions**, with unlimited reuse. State which demos and modalities were used.
**Zero-shot** and **Few-shot interactions** do not use released demonstrations.

Each result uses **800 fixed test episodes: 16 tasks × 50 episodes**. Report
success rates as percentages with two decimal places. API costs are in USD;
report the evaluation total for all 800 episodes, and development cost separately.
Use N/A when no paid API is used. Label estimates or incomplete costs.


**Track (select one):**

- [ ] Non-Agentic Track: fine-tuned robot policies (e.g., VLAs, WAMs) that directly predict actions.
- [ ] Agentic Track: systems using pretrained foundation models for planning, action, or coding.


## 1. Zero-shot

Use the original policy, or a policy trained on original RoboMME, without Plus adaptation.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Total evaluation API cost (USD):** [COST / N/A]
- **Results:** [ARTIFACTS LINK]

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | ___/50 |  |
| Counting | PickXtimes | ___/50 |  |
| Counting | SwingXtimes | ___/50 |  |
| Counting | StopCube | ___/50 |  |
| Permanence | VideoUnmask | ___/50 |  |
| Permanence | VideoUnmaskSwap | ___/50 |  |
| Permanence | ButtonUnmask | ___/50 |  |
| Permanence | ButtonUnmaskSwap | ___/50 |  |
| Reference | PickHighlight | ___/50 |  |
| Reference | VideoRepick | ___/50 |  |
| Reference | VideoPlaceButton | ___/50 |  |
| Reference | VideoPlaceOrder | ___/50 |  |
| Imitation | MoveCube | ___/50 |  |
| Imitation | InsertPeg | ___/50 |  |
| Imitation | PatternLock | ___/50 |  |
| Imitation | RouteStick | ___/50 |  |
| **Overall** | **All tasks** | **___/800** |  |


## 2. Few-shot demonstrations

Report this if you choose **Few-shot demonstrations**. Adapt using released offline demonstrations, without simulator interactions.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Demo use:** [SELECTION AND MODALITIES]
- **Development API cost (USD):** [COST / N/A]
- **Total evaluation API cost (USD):** [COST / N/A]
- **Results:** [ARTIFACTS LINK]

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | ___/50 |  |
| Counting | PickXtimes | ___/50 |  |
| Counting | SwingXtimes | ___/50 |  |
| Counting | StopCube | ___/50 |  |
| Permanence | VideoUnmask | ___/50 |  |
| Permanence | VideoUnmaskSwap | ___/50 |  |
| Permanence | ButtonUnmask | ___/50 |  |
| Permanence | ButtonUnmaskSwap | ___/50 |  |
| Reference | PickHighlight | ___/50 |  |
| Reference | VideoRepick | ___/50 |  |
| Reference | VideoPlaceButton | ___/50 |  |
| Reference | VideoPlaceOrder | ___/50 |  |
| Imitation | MoveCube | ___/50 |  |
| Imitation | InsertPeg | ___/50 |  |
| Imitation | PatternLock | ___/50 |  |
| Imitation | RouteStick | ___/50 |  |
| **Overall** | **All tasks** | **___/800** |  |


## 3. Few-shot interactions

Report this if you choose **Few-shot interactions**. Adapt through training-seed interactions, without released offline demonstrations.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Usage log:** [LINK TO ATTEMPTS AND SIMULATOR STEPS PER TASK, BY BUDGET]
- **Development API cost (USD):** 5 rollouts [COST / N/A]; 10 rollouts [COST / N/A]
- **Total evaluation API cost (USD):** 5 rollouts [COST / N/A]; 10 rollouts [COST / N/A]
- **Results:** [ARTIFACTS LINK]

| Suite | Task | 5 rollouts (%) | 10 rollouts (%) |
| --- | --- | --- | --- |
| Counting | BinFill |  |  |
| Counting | PickXtimes |  |  |
| Counting | SwingXtimes |  |  |
| Counting | StopCube |  |  |
| Permanence | VideoUnmask |  |  |
| Permanence | VideoUnmaskSwap |  |  |
| Permanence | ButtonUnmask |  |  |
| Permanence | ButtonUnmaskSwap |  |  |
| Reference | PickHighlight |  |  |
| Reference | VideoRepick |  |  |
| Reference | VideoPlaceButton |  |  |
| Reference | VideoPlaceOrder |  |  |
| Imitation | MoveCube |  |  |
| Imitation | InsertPeg |  |  |
| Imitation | PatternLock |  |  |
| Imitation | RouteStick |  |  |
| **Overall** | **All tasks** |  |  |


## 4. Few-shot demonstrations+interactions

Report this if you choose **Few-shot demonstrations+interactions**. Adapt using released offline demonstrations and training-seed interactions.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Demo use:** [SELECTION AND MODALITIES]
- **Usage log:** [LINK TO ATTEMPTS AND SIMULATOR STEPS PER TASK, BY BUDGET]
- **Development API cost (USD):** 5 rollouts [COST / N/A]; 10 rollouts [COST / N/A]
- **Total evaluation API cost (USD):** 5 rollouts [COST / N/A]; 10 rollouts [COST / N/A]
- **Results:** [ARTIFACTS LINK]

| Suite | Task | 5 rollouts (%) | 10 rollouts (%) |
| --- | --- | --- | --- |
| Counting | BinFill |  |  |
| Counting | PickXtimes |  |  |
| Counting | SwingXtimes |  |  |
| Counting | StopCube |  |  |
| Permanence | VideoUnmask |  |  |
| Permanence | VideoUnmaskSwap |  |  |
| Permanence | ButtonUnmask |  |  |
| Permanence | ButtonUnmaskSwap |  |  |
| Reference | PickHighlight |  |  |
| Reference | VideoRepick |  |  |
| Reference | VideoPlaceButton |  |  |
| Reference | VideoPlaceOrder |  |  |
| Imitation | MoveCube |  |  |
| Imitation | InsertPeg |  |  |
| Imitation | PatternLock |  |  |
| Imitation | RouteStick |  |  |
| **Overall** | **All tasks** |  |  |


<details>
<summary>Evaluation rules — apply to all four templates</summary>

1. **Count development rollouts.** A budget of 5 or 10 rollouts caps attempts
   per task, including development, tuning, failures, timeouts, early stops, and
   retries. A budget is a total: 10 rollouts does not mean ten more attempts after
   5. Unused attempts cannot transfer between tasks. Offline reuse is unlimited.
   Follow the official step limits. Agentic methods must retain full development
   logs and use the [compliance audit skill](../../skills/audit_robomme_plus_compliance.md).
2. **Use permitted resources.** Adapt only with released Plus demonstrations
   and provided training seeds, according to the chosen setting. Do not derive
   solutions from task source code, query oracles, or use privileged simulator
   information. Task-specific prompts and strategies must come from permitted training experience, not hand-written task solutions.
3. **Keep the model and agentic framework frozen.** Complete adaptation before
   testing. Keep weights, prompts, tools, and the harness fixed throughout
   evaluation. Reset memory between test episodes. Never adapt from test results.
   Test-time optimization is outside the standard protocol; report its results
   separately from these tables.
4. **Report usage.** Include the setting, demo use, rollout budget, actual
   attempts and simulator steps per task, and separate development/evaluation API costs
   in USD. Compare methods at the same setting and rollout budget.

Use a shared model checkpoint or agentic framework across all 16 tasks. Keep
all 800 episodes in the denominator, including failures, timeouts, model stops,
and errors. Keep the default test variations and supplied execution limits
(`ceil(1.5 * oracle_policy_step) + 200` control steps). Report altered test
conditions separately. For repeated runs, report each inference seed and the
mean for the same setting and budget.

Task-provided test demonstration videos are normal observations, not adaptation
data. Original RoboMME training is outside the Plus rollout budget.

Reference: [evaluation guide](../../../src/robomme_plus/README.md) ·
[baseline results and running instructions](../robomme_plus_baseline_results.md) ·
[training data format and seed mapping](../plus_training_data_format.md).

</details>
