"""Public evaluation interface: episodes and budgets come from oracle metadata."""

from .evaluation.metadata import load_episode_metadata
from .evaluation.runtime import ACTION_SPACES, make_episode
from .registry import TASKS, get_task


class BenchmarkEnvBuilder:
    """Build episodes with the original RoboMME reset/step contract.

    The execution limit is always ceil(1.5 * oracle_policy_step) + 200.
    Episodes use the supplied benchmark metadata and reference timing.
    """

    def __init__(
        self,
        env_id,
        *,
        dataset="test",
        metadata_path=None,
        action_space="waypoint",
        gui_render=False,
        perceptual_variations=True,
        mechanism_variations=True,
    ):
        self.spec = get_task(env_id)
        if action_space not in ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ACTION_SPACES}")
        self.env_id, self.dataset = (
            env_id,
            dataset if metadata_path is None else "custom",
        )
        self.action_space, self.gui_render = action_space, gui_render
        if type(perceptual_variations) is not bool:
            raise TypeError("perceptual_variations must be a boolean")
        self.perceptual_variations = perceptual_variations
        if type(mechanism_variations) is not bool:
            raise TypeError("mechanism_variations must be a boolean")
        self.mechanism_variations = mechanism_variations
        self._records = load_episode_metadata(
            env_id, dataset, metadata_path=metadata_path
        )

    @staticmethod
    def get_task_list():
        return list(TASKS)

    def get_episode_num(self):
        return len(self._records)

    def get_episode_metadata(self, episode_idx, *, mechanism_variations=None):
        if (
            isinstance(episode_idx, bool)
            or not isinstance(episode_idx, int)
            or not 0 <= episode_idx < len(self._records)
        ):
            raise IndexError(f"Episode index must be in [0, {len(self._records)})")
        from .variation.mechanism import effective_metadata
        enabled = self.mechanism_variations if mechanism_variations is None else mechanism_variations
        return effective_metadata(self._records[episode_idx], enabled)

    def resolve_episode(self, episode_idx):
        record = self.get_episode_metadata(episode_idx)
        return record["seed"], record["difficulty"]

    def episode_max_steps(self, episode_idx):
        return self.get_episode_metadata(episode_idx)["max_steps"]

    def episode_variant(self, episode_idx):
        return self.get_episode_metadata(episode_idx).get("variant")

    def make_env_for_episode(self, episode_idx, *, perceptual_variations=None,
                             mechanism_variations=None, **observation_options):
        mechanisms = self.mechanism_variations if mechanism_variations is None else mechanism_variations
        record = self.get_episode_metadata(episode_idx, mechanism_variations=mechanisms)
        enabled = self.perceptual_variations if perceptual_variations is None else perceptual_variations
        return make_episode(
            self.env_id,
            seed=record["seed"],
            difficulty=record["difficulty"],
            variant=record.get("variant"),
            native_difficulty=record.get("native_difficulty"),
            episode_metadata=record,
            action_space=self.action_space,
            gui_render=self.gui_render,
            perceptual_variations=enabled,
            mechanism_variations=mechanisms,
            **observation_options,
        )
