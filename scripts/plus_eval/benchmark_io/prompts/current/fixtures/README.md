# Synthetic public-interface examples

These examples contain blank images and invented calibration/state values. They
do not contain a benchmark scene or a policy and require no simulator or API.

RAX examples are MCP result objects: decode the JSON text in `content[0].text`,
then use its image descriptors' `content_index` to select base64 PNG attachments.
`rax_rejected_action.json` instead demonstrates a full JSON-RPC error envelope.

| File | Meaning |
| --- | --- |
| `rax_observe.json` | Initial observation plus conditioning-sequence descriptor |
| `rax_live_observe.json` | Execution observation at step 1 |
| `rax_status.json` | State/action layouts and episode limits |
| `rax_pixel_to_position.json` | Center-pixel conversion using the synthetic camera |
| `rax_terminal_success.json`, `rax_terminal_failure.json` | Two independent ending-transition examples |
| `rax_rejected_action.json` | Malformed action rejected before execution |
| `rax_checks.json` | Expected colors, depth and projected point |
| `rax_workspace/` | Mock workspace containing published manifests and both RGB-D views |
| `aspire_observation.py` | Synthetic ASPIRE `Observation`-shaped object with read-only arrays |

For RAX path tests, treat `rax_workspace/` as the workspace root. Paths in the
packets/manifests are relative to that root, not the manifest directory. The
initial manifest has two `video_demo` frames and one `execution` frame. All
three have step zero; their phase/frame index/timestamp distinguishes them.

Both RGB views have constant, distinct colors. Depth is PNG uint16 millimeters
(front 500, wrist 800); calibration uses `matrix`, `position_m` and
`quaternion_wxyz`. The front center pixel has world position `[1, 2, 2.5]` meters.
These values test parsing and units, not scene understanding or control.
