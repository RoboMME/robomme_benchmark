# RoboMME-Plus public interface protocol v3

This revision clarifies the existing data contract; it changes no observation
capability, task, budget, policy or native method workflow. RAX retains its exact
bundle rehearsal requirement. ASPIRE's development and selection rules are
unchanged. There is no CaP policy code or task-specific solution in this bundle.

Read BENCHMARK.md, DEMOS.md, your method's reference, PUBLIC_RULES.md and RUNTIME.md.
For RAX, fixtures/README.md indexes complete synthetic packets: both cameras,
RGB-D calibration, initial video, live observations, limits and terminal/errors.
The field names and structure match the adapter; the images and sensor values
are invented and contain no task scene or expert behavior.

Compared with v2, RAX.md explicitly documents `position_m`,
`camera_intrinsics[view].matrix`, `attachments[].content_index`, depth scale,
workspace-relative paths and the terminal result. It includes small data-access
examples, not robot-control logic. Synthetic fixtures are generated and tested
against the real packet serializers without starting a simulator.

Fresh studies default to this bundle and record version 3 plus file hashes.
Existing studies keep their pinned bundle. Installation refuses an in-place
version change to an existing agent workspace; a protocol change belongs to a
new, explicitly versioned development run. Earlier bundles remain in their original experiment snapshots, outside the release.
