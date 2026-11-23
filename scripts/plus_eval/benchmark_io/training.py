"""Small training interface for an operator's simulator service.

Expose these methods through the harness's existing RPC/tools. Do not pass this
Python object, the ledger or simulator filesystem into an untrusted agent process.
ASPIRE and RAX use the same Study/Episode backend through their native adapters.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .protocol import Study


class TrainingSession:
    def __init__(self, study_root, task, *, action_space="ee_pose", output_root=None):
        self._study = Study.open(study_root)
        self._study.select_episode(task)
        self._task = task
        if action_space not in ("ee_pose", "joint_angle"):
            raise ValueError("Use ee_pose or joint_angle for one-step training control")
        self._action_space = action_space
        self._output = Path(output_root) if output_root else self._study.root / "rollouts"
        self._episode = None

    def status(self):
        attempts = [a for a in self._study.attempts() if a["task"] == self._task]
        return dict(task=self._task, budget=self._study.config["budget"], used=len(attempts),
                    remaining=self._study.config["budget"] - len(attempts),
                    training_seeds=[r["seed"] for r in self._study.config["records"][self._task]])

    def reset(self, *, policy_revision, purpose="development", seed=None, episode=None):
        """Start one counted attempt; an omitted seed/index uses round-robin."""
        if not isinstance(policy_revision, str) or not policy_revision.strip():
            raise ValueError("Save the current code and supply its revision/hash before reset")
        self._study.select_episode(self._task, episode, seed=seed)
        self.close()
        from .episode import Episode
        self._episode = Episode(self._task, episode, study=self._study,
                                output=self._output / self._task / uuid4().hex,
                                action_space=self._action_space, purpose=purpose,
                                seed=seed, policy_revision=policy_revision)
        return self.observe()

    def observe(self):
        """Return public sensor frames and feedback; no simulator step or reset."""
        ep = self._episode
        if ep is None or ep.closed:
            raise RuntimeError("No open training episode; reset starts a counted attempt")
        return dict(frames=ep.frames, current=ep.current, goal=ep.instruction,
                    control_step=ep.steps, max_control_steps=ep.limit,
                    terminated=ep.terminated, truncated=ep.truncated, success=ep.success,
                    failure_reason=ep.failure_reason, error_message=ep.error_message)

    def step(self, action):
        if self._episode is None:
            raise RuntimeError("No training episode; reset starts a counted attempt")
        self._episode.step(action)
        return self.observe()

    def close(self):
        if self._episode is not None:
            self._episode.close()
            self._episode = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._episode is not None:
            self._episode.close(error=f"{exc_type.__name__}: {exc}" if exc_type else None)
            self._episode = None
