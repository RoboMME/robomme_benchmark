from historybench.env_record_wrapper.rpy_util import (
    quat_wxyz_to_rpy_xyz_torch,
    rpy_xyz_to_quat_wxyz_torch,
)
import torch
import numpy as np
from scipy.spatial.transform import Rotation

def test_conversion():
    print("Testing rpy_util scipy conversion...")
    
    # Test Case 1: Identity
    print("\n--- Test Case 1: Identity ---")
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0]) # wxyz
    rpy_identity = quat_wxyz_to_rpy_xyz_torch(quat_identity)
    print(f"Quat: {quat_identity}")
    print(f"RPY (expected 0,0,0): {rpy_identity}")
    assert torch.allclose(rpy_identity, torch.zeros(3), atol=1e-6)
    
    quat_back = rpy_xyz_to_quat_wxyz_torch(rpy_identity)
    print(f"Quat back: {quat_back}")
    assert torch.allclose(quat_back, quat_identity, atol=1e-6)

    # Test Case 2: 90 degrees around X (Roll)
    print("\n--- Test Case 2: 90 deg Roll ---")
    # Euler xyz: pi/2, 0, 0
    rpy_target = torch.tensor([np.pi/2, 0.0, 0.0])
    quat_mid = rpy_xyz_to_quat_wxyz_torch(rpy_target)
    print(f"RPY Input: {rpy_target}")
    print(f"Quat Output: {quat_mid}")
    
    # Check against scipy directly
    rot = Rotation.from_euler('xyz', [np.pi/2, 0, 0], degrees=False)
    quat_scipy_xyzw = rot.as_quat()
    quat_scipy_wxyz = np.roll(quat_scipy_xyzw, 1) # xyzw -> wxyz
    print(f"Scipy ground truth (wxyz): {quat_scipy_wxyz}")
    
    # Note: Quaternion sign might flip, but represented rotation is same
    # But for 90 deg simple rotation, it usually matches
    assert np.allclose(quat_mid.numpy(), quat_scipy_wxyz, atol=1e-6) or \
           np.allclose(quat_mid.numpy(), -quat_scipy_wxyz, atol=1e-6)

    rpy_back = quat_wxyz_to_rpy_xyz_torch(quat_mid)
    print(f"RPY Back: {rpy_back}")
    assert torch.allclose(rpy_back, rpy_target, atol=1e-6)

    # Test Case 3: Batch input
    print("\n--- Test Case 3: Batch Input ---")
    quats = torch.stack([quat_identity, quat_mid])
    rpys = quat_wxyz_to_rpy_xyz_torch(quats)
    print(f"Batch RPYs:\n{rpys}")
    assert rpys.shape == (2, 3)
    assert torch.allclose(rpys[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(rpys[1], rpy_target, atol=1e-6)
    
    quats_back = rpy_xyz_to_quat_wxyz_torch(rpys)
    print(f"Batch Quats back:\n{quats_back}")
    
    print("\nVerification Passed!")

if __name__ == "__main__":
    test_conversion()
