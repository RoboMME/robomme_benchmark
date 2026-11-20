# RoboMME-Plus public protocol v4

This revision follows the website's **Few-shot demonstrations**, **Few-shot
interactions** and **Few-shot demonstrations+interactions** settings. Interaction
budgets are 5 or 10 simulator starts per task; demonstrations-only permits none.
Tool and artifact keys remain `AfD`, `AfI` and `AfD+I`.

Read BENCHMARK.md, DEMOS.md, your method's reference, PUBLIC_RULES.md and RUNTIME.md.
The observation contract, native method workflow and synthetic fixtures from v3
are unchanged. Fixtures document both cameras, RGB-D calibration, video phases,
control limits and terminal/error fields without task scenes or expert behavior.
They are generated through the packet serializers without starting a simulator.

Fresh studies pin version 4 and its file hashes. Existing studies keep their
original prompts, assignments and source snapshots. Installation rejects changes
to a prepared workspace; use a fresh run for the new protocol.
