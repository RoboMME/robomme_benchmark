"""Scene behavior shared by all memory manipulation tasks."""

from typing import Any, Dict

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from robomme_plus.scoring.failures import clear_failure, failure_message


class MemoryTask(BaseEnv):
    PERCEPTUAL_VARIATIONS = False

    def __init__(self, *args, _perceptual_variation=None, **kwargs):
        self._perceptual_variation = None
        if _perceptual_variation is not None and self.PERCEPTUAL_VARIATIONS:
            from robomme_plus.variation.runtime import PerceptualVariation
            self._perceptual_variation = PerceptualVariation(
                self.task_name, self.seed, _perceptual_variation)
            kwargs = self._perceptual_variation.sensor_options(kwargs)
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        if self._perceptual_variation is not None:
            self._perceptual_variation.reset(self)
        clear_failure(self)
        self.timestep = 0
        self._last_task_index = None
        self._timelimit_deadlines = {}
        self._timewindow_timers = {}
        if hasattr(self, "first_timestep"):
            del self.first_timestep
        return super().reset(*args, **kwargs)

    def _load_lighting(self, options):
        if self._perceptual_variation is not None and self._perceptual_variation.load_lighting(self):
            return
        return super()._load_lighting(options)

    def _get_obs_sensor_data(self, *args, **kwargs):
        if self._perceptual_variation is not None:
            self._perceptual_variation.apply(self)
        return super()._get_obs_sensor_data(*args, **kwargs)

    def close(self):
        try:
            return super().close()
        finally:
            if self._perceptual_variation is not None:
                self._perceptual_variation.close()

    def get_info(self):
        info = super().get_info()
        info.pop("failure_reason", None)
        if bool(info.get("fail", False)) and not bool(info.get("success", False)):
            info["failure_reason"] = failure_message(self)
        return info

    @property
    def _default_sensor_configs(self):
        camera_eye = [0.3, 0, 0.4]
        camera_target = [0, 0, -0.2]
        pose = sapien_utils.look_at(eye=camera_eye, target=camera_target)
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward * 0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)


class TableTask(MemoryTask):
    """Table, Panda configuration and reset state for the current task designs."""

    SUPPORTED_ROBOTS = ["panda_wristcam"]

    def __init__(self, *args, seed=0, robot_uids="panda_wristcam",
                 robot_init_qpos_noise=0, **kwargs):
        from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
        from robomme_plus.simulation.specification import SpecRecorder

        self.seed = seed
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.generator = torch.Generator().manual_seed(seed)
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
        self.task_list = []
        self.timestep = 0
        self._spec = SpecRecorder(None, self.task_name, {"seed": seed})
        config = PICK_CUBE_CONFIGS.get(robot_uids, PICK_CUBE_CONFIGS["panda"])
        for name in ("cube_half_size", "goal_thresh", "cube_spawn_half_size",
                     "cube_spawn_center", "max_goal_height", "sensor_cam_eye_pos",
                     "sensor_cam_target_pos", "human_cam_eye_pos", "human_cam_target_pos"):
            setattr(self, name, config[name])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options):
        from mani_skill.utils.scene_builder.table import TableSceneBuilder

        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

    def _initialize_episode(self, env_idx, options):
        from robomme_plus.control.reset import get_reset_panda_param

        self.table_scene.initialize(env_idx)
        self.agent.reset(get_reset_panda_param("qpos"))
        self.timestep = 0
        self._timelimit_deadlines = {}
        self._last_task_index = None

    def evaluate(self, **kwargs):
        return {"success": torch.tensor([False]), "fail": torch.tensor([False])}
