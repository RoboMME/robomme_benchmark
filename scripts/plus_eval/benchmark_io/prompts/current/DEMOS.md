# Released demonstration format

This reference describes the visual export actually provided to these adapters.
Read the supplied `manifest.json`; do not infer a format from the original dataset
name. `interaction-only` has no released-demonstration access.

The manifest has `schema_version: 1`, a `task`, and a `demonstrations` list. Each
demonstration has an `episode` index, `goal`, `control_hz`, and ordered `frames`.
Each frame records:

| Field | Meaning |
| --- | --- |
| `file` | JPEG path relative to the directory containing this manifest |
| `frame` | Zero-based index within this recording |
| `timestamp_s` | Recording time in seconds, measured from its first frame |
| `phase` | `video_demo` or `expert_execution` |

Each JPEG contains **front on the left, wrist on the right**. The two views have
equal width; split the image at half its width to inspect them separately. The
manifest also provides `ordered_images` as a convenience, but use `frames` for
phase and time information. Use each demonstration's own frame list and goal;
the top-level list spans multiple episodes.

`video_demo` is the task's initial conditioning footage. `expert_execution` is
what the expert robot subsequently did to solve that training episode. Some
tasks have no `video_demo` frames. Recording timestamps continue across the phase
boundary; they are not the new execution's control-step counter. The execution
scene can differ from the scene shown in the conditioning footage.

This export provides RGB images, goals, phase labels and recording times. It does
**not** provide depth, camera calibration, robot-state arrays or expert actions.
Those fields must not be assumed to exist because live observations have them.
Images may show the robot moving, but they are not numeric action commands.

For ASPIRE's full campaign, the manifests are
`/workspace/demonstrations/<TASK>/manifest.json`. A single-task sanity session uses
`/workspace/demonstrations/manifest.json`. For RAX, use the manifest path in the
native status reply's `demonstration_context`; the current adapter publishes it
at `runtime/demonstrations/manifest.json` under the agent workspace.

This is ordinary offline file reading in the development workspace:

```python
import json
from pathlib import Path
from PIL import Image

manifest_path = Path(manifest_path_from_session)
manifest = json.loads(manifest_path.read_text())
demo = manifest["demonstrations"][0]
record = demo["frames"][0]
image = Image.open(manifest_path.parent / record["file"]).convert("RGB")
mid = image.width // 2
front = image.crop((0, 0, mid, image.height))
wrist = image.crop((mid, 0, image.width, image.height))
phase, recording_time = record["phase"], record["timestamp_s"]
```

This example documents file access only. Generated runtime programs must follow
their own method's sandbox and API rules.
