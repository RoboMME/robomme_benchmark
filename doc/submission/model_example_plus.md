# RoboMME-Plus submission templates

Copy the template for each setting you report. Replace bracketed placeholders
and fill in the results table. Both Standard and Agentic tracks can use any
setting. For interaction settings, report R1, R3, R5, and R10 in the same table;
mark unevaluated columns N/R.

All 50 released demonstrations are available for **AfD** and **AfD+I**, with
unlimited reuse. State which demos and modalities were used. **ZS** and **AfI**
do not use released demonstrations.

Each result uses **800 fixed test episodes: 16 tasks × 50 episodes**. Report
success rates as percentages with two decimal places. API costs are in USD;
report the evaluation total for all 800 episodes, and adaptation cost separately.
Use N/A when no paid API is used. Label estimates or incomplete costs.


## 1. Zero-shot transfer (ZS)

Report this if you choose **ZS**. Transfer directly from regular RoboMME, without Plus adaptation.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Standard / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Standard track):** [SIZE IN B / N/A]
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


## 2. Adapt from demos (AfD)

Report this if you choose **AfD**. Adapt using released offline demonstrations, without simulator interactions.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Standard / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Standard track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Demo use:** [SELECTION AND MODALITIES]
- **Adaptation API cost (USD):** [COST / N/A]
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


## 3. Adapt from interactions (AfI)

Report this if you choose **AfI**. Adapt through training-seed interactions, without released offline demonstrations.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Standard / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Standard track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Usage log:** [LINK TO ATTEMPTS AND SIMULATOR STEPS PER TASK, BY BUDGET]
- **Adaptation API cost (USD):** R1 [COST / N/A]; R3 [COST / N/A]; R5 [COST / N/A]; R10 [COST / N/A]
- **Total evaluation API cost (USD):** R1 [COST / N/A]; R3 [COST / N/A]; R5 [COST / N/A]; R10 [COST / N/A]
- **Results:** [ARTIFACTS LINK]

| Suite | Task | R1 (%) | R3 (%) | R5 (%) | R10 (%) |
| --- | --- | --- | --- | --- | --- |
| Counting | BinFill |  |  |  |  |
| Counting | PickXtimes |  |  |  |  |
| Counting | SwingXtimes |  |  |  |  |
| Counting | StopCube |  |  |  |  |
| Permanence | VideoUnmask |  |  |  |  |
| Permanence | VideoUnmaskSwap |  |  |  |  |
| Permanence | ButtonUnmask |  |  |  |  |
| Permanence | ButtonUnmaskSwap |  |  |  |  |
| Reference | PickHighlight |  |  |  |  |
| Reference | VideoRepick |  |  |  |  |
| Reference | VideoPlaceButton |  |  |  |  |
| Reference | VideoPlaceOrder |  |  |  |  |
| Imitation | MoveCube |  |  |  |  |
| Imitation | InsertPeg |  |  |  |  |
| Imitation | PatternLock |  |  |  |  |
| Imitation | RouteStick |  |  |  |  |
| **Overall** | **All tasks** |  |  |  |  |


## 4. Adapt from demos+interactions (AfD+I)

Report this if you choose **AfD+I**. Adapt using released offline demonstrations and training-seed interactions.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL) · [Code](CODE_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Standard / Agentic]
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Standard track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Demo use:** [SELECTION AND MODALITIES]
- **Usage log:** [LINK TO ATTEMPTS AND SIMULATOR STEPS PER TASK, BY BUDGET]
- **Adaptation API cost (USD):** R1 [COST / N/A]; R3 [COST / N/A]; R5 [COST / N/A]; R10 [COST / N/A]
- **Total evaluation API cost (USD):** R1 [COST / N/A]; R3 [COST / N/A]; R5 [COST / N/A]; R10 [COST / N/A]
- **Results:** [ARTIFACTS LINK]

| Suite | Task | R1 (%) | R3 (%) | R5 (%) | R10 (%) |
| --- | --- | --- | --- | --- | --- |
| Counting | BinFill |  |  |  |  |
| Counting | PickXtimes |  |  |  |  |
| Counting | SwingXtimes |  |  |  |  |
| Counting | StopCube |  |  |  |  |
| Permanence | VideoUnmask |  |  |  |  |
| Permanence | VideoUnmaskSwap |  |  |  |  |
| Permanence | ButtonUnmask |  |  |  |  |
| Permanence | ButtonUnmaskSwap |  |  |  |  |
| Reference | PickHighlight |  |  |  |  |
| Reference | VideoRepick |  |  |  |  |
| Reference | VideoPlaceButton |  |  |  |  |
| Reference | VideoPlaceOrder |  |  |  |  |
| Imitation | MoveCube |  |  |  |  |
| Imitation | InsertPeg |  |  |  |  |
| Imitation | PatternLock |  |  |  |  |
| Imitation | RouteStick |  |  |  |  |
| **Overall** | **All tasks** |  |  |  |  |


<details>
<summary>Evaluation rules — apply to all four templates</summary>

1. **Count all training-seed attempts.** R1, R3, R5, and R10 cap attempts per
   task, including development, tuning, failures, timeouts, early stops, and
   retries. Budgets are cumulative; R10 does not mean ten more attempts after
   R5. Unused attempts cannot transfer between tasks. Offline reuse is unlimited.
2. **Use permitted resources.** Adapt only with released Plus demonstrations
   and provided training seeds, according to the chosen setting. Do not derive
   solutions from task source code, query oracles, or use privileged simulator
   information. Task-specific prompts and strategies must come from permitted training experience, not hand-written task solutions.
3. **Keep the model and agentic framework frozen.** Complete adaptation before
   testing. Keep weights, prompts, tools, and the harness fixed throughout
   evaluation. Reset memory between test episodes. Never adapt from test results.
4. **Report test-time optimization separately.** It is outside the standard
   protocol; report its results separately from these tables.
5. **Report usage and cost.** Include the setting, demo use, R, actual attempts
   and simulator steps per task, and separate adaptation/evaluation API costs
   in USD. Compare methods at the same setting and R.

Use a shared model checkpoint or agentic framework across all 16 tasks. Keep
all 800 episodes in the denominator, including failures, timeouts, model stops,
and errors. Keep the default test variations and supplied execution limits
(`ceil(1.5 * oracle_policy_step) + 200` control steps). Report altered test
conditions separately. For repeated runs, report each inference seed and the
mean for the same setting and budget.

Task-provided test demonstration videos are normal observations, not adaptation
data. Original RoboMME training is outside the Plus rollout budget.

Reference: [evaluation guide](../../src/robomme_plus/README.md) ·
[baseline results and running instructions](../robomme_plus_baseline_results.md) ·
[training data format and seed mapping](../plus_training_data_format.md).

</details>
