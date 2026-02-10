import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "historybench/env_record_wrapper/rpy_util.py"
MODULE_SPEC = spec_from_file_location("rpy_util", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
# 通过文件路径直载模块，避免引入包级重依赖。
MODULE = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

align_quat_sign_with_prev_torch = MODULE.align_quat_sign_with_prev_torch
normalize_quat_wxyz_torch = MODULE.normalize_quat_wxyz_torch
quat_wxyz_to_rpy_xyz_torch = MODULE.quat_wxyz_to_rpy_xyz_torch
summarize_and_print_rpy_sequence = MODULE.summarize_and_print_rpy_sequence
unwrap_rpy_with_prev_torch = MODULE.unwrap_rpy_with_prev_torch


def test_normalize_quat_wxyz_torch_fallback_and_regular_case():
    # 前两条非法输入应回退为单位四元数，第三条应正常归一化。
    quat = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [float("nan"), 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float64,
    )
    out = normalize_quat_wxyz_torch(quat)

    expected_fallback = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(out[0], expected_fallback)
    torch.testing.assert_close(out[1], expected_fallback)

    norm = torch.linalg.norm(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    expected_regular = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64) / norm
    torch.testing.assert_close(out[2], expected_regular)


def test_align_quat_sign_with_prev_torch_flips_on_negative_dot():
    # 点积为负时，当前帧应翻转符号并与上一帧对齐。
    quat = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    prev = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    out = align_quat_sign_with_prev_torch(quat, prev)
    torch.testing.assert_close(out, prev)


def test_unwrap_rpy_with_prev_torch_cross_pi_continuity():
    # 从 +179° 到 -179°，展开后应是 +2° 的连续增量而非大跳变。
    prev = torch.tensor([[0.0, 0.0, np.deg2rad(179.0)]], dtype=torch.float64)
    curr = torch.tensor([[0.0, 0.0, np.deg2rad(-179.0)]], dtype=torch.float64)
    out = unwrap_rpy_with_prev_torch(curr, prev)

    delta_yaw = float((out - prev)[0, 2].item())
    assert np.isclose(delta_yaw, np.deg2rad(2.0), atol=1e-9)


def test_quat_wxyz_to_rpy_xyz_torch_identity():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    rpy = quat_wxyz_to_rpy_xyz_torch(quat)
    torch.testing.assert_close(rpy, torch.zeros((1, 3), dtype=torch.float64))


def test_summarize_and_print_rpy_sequence_empty_and_single_and_multi():
    # 覆盖空输入、单样本、多样本，并检查返回值可直接 JSON 序列化。
    empty_summary = summarize_and_print_rpy_sequence([])
    assert empty_summary["count"] == 0
    assert empty_summary["axis_min_rad"] is None
    assert empty_summary["prev_step_peak_transition"] is None
    json.dumps(empty_summary)

    single_summary = summarize_and_print_rpy_sequence([0.1, -0.2, 0.3])
    assert single_summary["count"] == 1
    assert single_summary["prev_step_peak_transition"] is None
    assert single_summary["axis_max_abs_delta_rad"] == [0.0, 0.0, 0.0]
    json.dumps(single_summary)

    multi_summary = summarize_and_print_rpy_sequence(np.array([0.0, 0.0, 0.0, 0.0, 0.1, -0.2]))
    assert multi_summary["count"] == 2
    assert multi_summary["prev_step_peak_transition"] == [0, 1]
    json.dumps(multi_summary)
