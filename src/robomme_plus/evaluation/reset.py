"""Bounded recovery of setup errors before a policy receives observations."""

from copy import deepcopy

import gymnasium as gym

from robomme_plus.common.logging_utils import logger


class ResetRetryWrapper(gym.Wrapper):
    """Try reset at most three times, rebuilding the same episode on failure.

    ``rebuild`` must construct the identical task, seed, variations and wrappers.
    It must not select another episode. No step or policy execution is retried.
    """

    MAX_ATTEMPTS = 3

    def __init__(self, env, *, rebuild):
        super().__init__(env)
        self._rebuild = rebuild
        self._needs_rebuild = False
        self.reset_attempts = 0
        self.max_attempts = self.MAX_ATTEMPTS

    def limit_reset_attempts(self, attempts):
        """Operator-only cap for budgeted training; each start must be counted."""
        if type(attempts) is not int or not 1 <= attempts <= self.MAX_ATTEMPTS:
            raise ValueError("Reset attempts must be an integer from 1 to 3")
        self.max_attempts = attempts

    def reset(self, **kwargs):
        self.reset_attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            self.reset_attempts = attempt
            try:
                if self._needs_rebuild:
                    self.env = self._rebuild()
                    self._needs_rebuild = False
                obs, info = self.env.reset(**deepcopy(kwargs))
            except (ValueError, TypeError, MemoryError):
                # Invalid arguments cannot recover; process memory exhaustion
                # requires an operator retry in a fresh process.
                raise
            except Exception as exc:
                self._needs_rebuild = True
                message = str(exc).lower()
                if any(token in message for token in (
                    "out of memory", "cuda_error", "device-side assert",
                    "illegal memory access", "device lost",
                )):
                    raise
                logger.warning("RoboMME-plus reset attempt %d/%d failed: %s: %s",
                               attempt, self.max_attempts, type(exc).__name__, exc)
                try:
                    self.env.close()
                except Exception:
                    logger.exception("Error closing the failed reset environment")
                if attempt == self.max_attempts:
                    raise
            else:
                info = dict(info, reset_attempts=attempt)
                return obs, info
