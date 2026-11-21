from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmark_io.protocol import Study, BudgetExhausted
from benchmark_io.episode import Episode, camera_to_world, public_frames
from services.robomme_plus.session import delta_to_native
from services.robomme_plus.tools import pixel_to_world
from cap_harness.contracts import ArmCommand, RobotAction, Observation, CameraObservation, RobotState, Pose
from cap_harness.robomme_plus.adapter import RoboMMEPlusAdapter


def records():
    return {"BinFill": [dict(seed=100+i, episode=i, max_steps=100, control_freq=20) for i in range(3)]}


def make_study(tmp_path, setting="interaction-only", method="rax", budget=10):
    return Study(tmp_path, method=method, setting=setting, budget=0 if setting=="demo-only" else budget, records=records())


def test_budget_atomic_and_persists_failed_attempts(tmp_path):
    study = make_study(tmp_path)
    def reserve(_):
        try:
            return Study.open(tmp_path).reserve("BinFill", purpose="concurrent_test")
        except BudgetExhausted:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = [x for x in pool.map(reserve, range(30)) if x]
    assert len(attempts) == 10
    assert sorted([sum(a["episode"]==i for a in attempts) for i in range(3)]) == [3,3,4]
    study.finish(attempts[0]["id"], {"error": "simulator startup failed"})
    with pytest.raises(BudgetExhausted):
        Study.open(tmp_path).reserve("BinFill", purpose="retry")


def test_settings_are_enforced_before_data_or_simulator_access(tmp_path):
    study = make_study(tmp_path / "demo-only", setting="demo-only")
    with pytest.raises(BudgetExhausted):
        study.reserve("BinFill", purpose="preview")
    with pytest.raises(PermissionError):
        make_study(tmp_path / "interaction-only").require_demos()
    make_study(tmp_path / "both", setting="demo+interaction").require_demos()
    with pytest.raises(ValueError):
        make_study(tmp_path / "interaction-only", budget=9)
    with pytest.raises(ValueError):
        study.reserve("BinFill", 50, purpose="test_leak")


def test_demo_export_denies_interaction_only_before_opening_files(tmp_path):
    from benchmark_io.demonstrations import export_task
    study = make_study(tmp_path / "study")
    with pytest.raises(PermissionError):
        export_task(study, "BinFill", tmp_path / "does_not_exist", tmp_path / "exports")
    assert not (tmp_path / "exports").exists()


def test_rax_r0_configuration_and_wrong_setting_rejection(tmp_path, monkeypatch):
    from services.controller.config import Configuration, Limits
    study = make_study(tmp_path / "study", setting="demo-only")
    monkeypatch.setenv("ROBOMME_PLUS_STUDY", str(study.root))
    monkeypatch.setenv("RAX_ARTIFACT_ROOT", str(tmp_path))
    limit = Limits(wall_seconds=600, memory_mb=1024, cpus=1., pids=128,
                   scratch_mb=128, artifact_bytes=1000000)
    kwargs = dict(root=tmp_path / "rax", image="sha256:" + "0"*64, task="BinFill",
                  exploration_seeds=(), formal_seed=0, eval_seed=0, formal_eval_seed=1,
                  sim_gpu="GPU-00000000-0000-0000-0000-000000000000", development=limit,
                  formal=limit, training=limit, benchmark="robomme_plus", agent_cli="claude",
                  agent_effort="medium", demonstration_context="completion_sequence")
    assert Configuration(**kwargs).exploration_seeds == ()
    assert Configuration(**{**kwargs, "agent_effort": "high", "formal_episodes": 50}).agent_effort == "high"
    with pytest.raises(ValueError):
        Configuration(**{**kwargs,"demonstration_context":"none"})


def test_failed_reset_consumes_attempt(tmp_path):
    study = make_study(tmp_path / "study")
    class Builder:
        def __init__(self, *args, **kwargs): pass
        def get_episode_metadata(self, i): return records()["BinFill"][i]
        def make_env_for_episode(self, *args, **kwargs): raise RuntimeError("render startup failed")
    with pytest.raises(RuntimeError, match="render startup failed"):
        Episode("BinFill", 0, study=study, output=tmp_path / "rollout", builder_factory=Builder)
    assert len(study.attempts()) == 1
    assert study.attempts()[0]["ended"] is not None


def test_rax_camera_roundtrip_with_rotated_translated_camera():
    cv_to_world = np.eye(4)
    cv_to_world[:3,:3] = Rotation.from_euler("xyz", [.4,-.7,.2]).as_matrix()
    cv_to_world[:3,3] = [.3,.8,.6]
    converted = camera_to_world(np.linalg.inv(cv_to_world)[:3])
    np.testing.assert_allclose(converted, cv_to_world, atol=1e-12)
    gl_rotation = converted[:3,:3] @ np.diag([1,-1,-1])
    q = Rotation.from_matrix(gl_rotation).as_quat()
    k = np.array([[150,0,128],[0,150,128],[0,0,1]])
    expected = cv_to_world @ np.r_[np.linalg.solve(k,[100,140,1])*.8,1]
    result = pixel_to_world(np.full((256,256),.8), {"matrix":k},
                           {"position_m":converted[:3,3],"quaternion_wxyz":q[[3,0,1,2]]},100,140)
    np.testing.assert_allclose(result["position_m"], expected[:3], atol=1e-12)


def test_rax_delta_rotation_world_frame_and_gripper():
    eef = np.array([.1,.2,.3,.4,.5,.6])
    action = [.2,-.4,.6,.1,-.2,.3,1]
    native = delta_to_native(action,eef)
    np.testing.assert_allclose(native[:3], [.11,.18,.33])
    expected = Rotation.from_rotvec(np.array(action[3:6])*.5).as_matrix() @ Rotation.from_euler("xyz",eef[3:6]).as_matrix()
    np.testing.assert_allclose(Rotation.from_euler("xyz",native[3:6]).as_matrix(),expected,atol=1e-12)
    assert native[6] == -1
    with pytest.raises(ValueError): delta_to_native([0]*6+[float("nan")],eef)


def test_reset_preserves_both_cameras_and_all_phases():
    obs = {f"{key}_list": [np.zeros(size) for _ in range(3)] for key,size in
           (("joint_state",7),("eef_state",6),("gripper_state",2))}
    info = {}
    for camera in ("front","wrist"):
        obs[f"{camera}_rgb_list"] = [np.full((2,2,3),i,np.uint8) for i in range(3)]
        obs[f"{camera}_depth_list"] = [np.full((2,2,1),1000,np.int16)]*3
        obs[f"{camera}_camera_extrinsic_list"] = [np.eye(4)[:3]]*3
        info[f"{camera}_camera_intrinsic"] = np.eye(3)
    frames = public_frames(obs,info,first_step=0,reset=True)
    assert [f["phase"] for f in frames] == ["video_demo","video_demo","execution"]
    assert frames[-1]["cameras"]["wrist"]["rgb"][0,0,0] == 2
    np.testing.assert_array_equal(frames[0]["cameras"]["front"]["depth_m"],np.ones((2,2)))


def test_aspire_joint_commands_are_absolute_and_gripper_not_inverted(tmp_path):
    study = make_study(tmp_path / "study", method="aspire")
    adapter = RoboMMEPlusAdapter(study_root=study.root, output_root=tmp_path / "out")
    sent = []
    def step(action):
        sent.append(action.copy())
    adapter.episode = SimpleNamespace(step=step,steps=1,terminated=False,truncated=False)
    robot_pose = Pose(np.zeros(3), np.array([1.,0.,0.,0.]), "robot_base")
    robot_state = RobotState(joint_positions={"primary":np.zeros(7)}, joint_velocities={"primary":np.zeros(7)},
        end_effector_poses={"primary":robot_pose}, gripper_positions={"primary":1.},
        base_frame="robot_base", embodiment="robomme_plus")
    observation = Observation(cameras={"front":CameraObservation(np.zeros((2,2,3),np.uint8),
        np.ones((2,2)),np.eye(3),"front",robot_pose)},robot_state=robot_state)
    adapter.get_observation = lambda: observation
    joints=np.arange(7)*.1
    for g in (0.,1.):
        result=adapter.native_step(RobotAction({"primary":ArmCommand("joint_position",joints,g,"robomme_plus")}))
        assert result.ok
    np.testing.assert_allclose(sent[0][:7],joints)
    assert sent[0][-1] == -1 and sent[1][-1] == 1
