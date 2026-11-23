"""Bound Ponder's multimodal context by replaying complete retained segments.

Qwen3.5 also has recurrent attention state, so slicing full-attention KV tensors
alone cannot remove old observations. Rebuild both cache types when trimming.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ponderpounce.inference.append_context import SoftS2SessionContext


CONTEXT_POLICY = "task_demo_recent_complete_queries_v1"


@dataclass
class Query:
    segment: object
    total_queries: int
    index: int


class WindowedS2Context(SoftS2SessionContext):
    def __init__(self, *, context_tokens=None, **kwargs):
        cap = int(kwargs["system2"].effective_context_cap)
        if context_tokens is not None and not 0 < context_tokens <= cap:
            raise ValueError(f"Ponder context tokens must be in [1, {cap}]")
        self.context_cap = cap if context_tokens is None else int(context_tokens)
        self.truncation_count = 0
        self.last_truncation = None
        self._recording = False
        self._pending = []
        self._history = []
        self._last_plan = None
        self._last_plan_index = None
        super().__init__(**kwargs)

    @property
    def generation_reserve(self):
        # Upstream can append a trigger even when max_new_tokens is zero.
        return max(1, self.max_new_tokens) + int(self.s2.num_cognition_tokens)

    def reset(self):
        self._recording = False
        self._pending = []
        self._history = []
        self._last_plan = self._last_plan_index = None
        # The constructor normally prefills the demo before seeing an execution
        # observation. Bound even this first append and leave response space.
        if self._prefix.length + self.generation_reserve > self.context_cap:
            before = self._prefix.length
            prefix, indices = self._fit_prefix(self.context_cap - self.generation_reserve)
            self._prefix = prefix
            self._report_truncation({"reason": "demonstration_prefix", "tokens_before": before,
                "tokens_after": prefix.length, "dropped_queries": 0, "kept_queries": [],
                "demo_frame_indices": indices})
        super().reset()
        if self._prefix.length:
            self._task_prefix_pending = False

    def _fit_prefix(self, budget):
        task = self.compiler.obs([], qid=-1, task_prefix=self.session.task_text)
        if task.length > budget:
            raise ValueError("Ponder context cap cannot fit the task and current observation/response reserve")
        images = self.session.demo_images
        count = len(images)
        while count:
            indices = np.linspace(0, len(images) - 1, count, dtype=int).tolist()
            prefix = self.compiler.prefix(self.session.task_text, [images[i] for i in indices])
            if prefix.length <= budget:
                return prefix, indices
            image_tokens = max(1, prefix.length - task.length)
            count = min(count - 1, max(0, int(count * (budget - task.length) / image_tokens)))
        return task, []

    def _report_truncation(self, event):
        self.truncation_count += 1
        self.last_truncation = {**event, "operation": "truncate", "context_cap": self.context_cap,
            "generation_reserve": self.generation_reserve, "count": self.truncation_count}
        self.on_truncation(self.last_truncation)

    def on_truncation(self, event):
        """Recording adapters can attach the retained context to their call trace."""

    def _make_room(self, obs_pils):
        snapshot = self._prefix_snapshot
        context_len = snapshot.context_len if snapshot is not None else self._context_len
        pending = snapshot.task_prefix_pending if snapshot is not None else self._task_prefix_pending
        task_prefix = self.session.task_text if pending else None
        incoming = self.compiler.obs(obs_pils, qid=self.query_index, task_prefix=task_prefix)
        if context_len + incoming.length + self.generation_reserve <= self.context_cap:
            return
        observation = self.compiler.obs(obs_pils, qid=self.query_index)
        required = observation.length + self.generation_reserve
        task = self.compiler.obs([], qid=-1, task_prefix=self.session.task_text)
        available = self.context_cap - required
        if task.length > available:
            raise ValueError("Ponder context cap cannot fit the task and current observation/response reserve")
        before = self._context_len
        # Remove a quarter-window at a time instead of rebuilding every fire.
        budget = max(task.length, available - self.context_cap // 4)
        prefix, indices = self._fit_prefix(min(budget, max(task.length, self.context_cap // 2)))
        history = list(self._history)
        original_count = len(history)

        def plan_to_keep():
            if self._last_plan is None or getattr(self.s2, "current_obs_only", False):
                return None
            if any(q.index == self._last_plan_index for q in history):
                return None
            return self._last_plan

        while True:
            plan = plan_to_keep()
            length = prefix.length + sum(q.segment.length for q in history) + (plan.length if plan is not None else 0)
            if length <= budget:
                break
            if history:
                history.pop(0)
            else:
                # Prefer keeping the latest generated plan over extra demo
                # frames. A task+plan still cannot displace the current images.
                plan_length = plan.length if plan is not None else 0
                if task.length + plan_length > available:
                    plan = None
                    break
                budget = max(budget, task.length + plan_length)
                prefix, indices = self._fit_prefix(budget - plan_length)

        query_index, n_obs_frames = self.query_index, self.n_obs_frames
        self._recording = False
        self._pending = []
        self._prefix = prefix
        # Drop the old cache before allocating another. A fresh cache clears
        # both KV tensors and the hybrid model's convolution/recurrent state.
        self.cache = None
        self._prefix_snapshot = None
        super().reset()
        self._task_prefix_pending = False
        if plan is not None:
            self._append(plan, total_queries=0)
        for query in history:
            self._append(query.segment, total_queries=query.total_queries)
        self.query_index, self.n_obs_frames = query_index, n_obs_frames
        self._history = history
        self._report_truncation({"reason": "execution_history", "tokens_before": before,
            "tokens_after": self._context_len, "dropped_queries": original_count - len(history),
            "kept_queries": [q.index for q in history], "demo_frame_indices": indices,
            "latest_plan_preserved_separately": plan is not None,
            "incoming_observation_tokens": observation.length})

    @torch.no_grad()
    def fire(self, obs_pils):
        self._recording = False
        self._pending = []
        self._make_room(obs_pils)
        self._recording = True
        index = self.query_index
        try:
            result = super().fire(obs_pils)
            segment = self.compiler.concat([item[0] for item in self._pending])
            query = Query(segment, sum(item[1] for item in self._pending), index)
            if getattr(self.s2, "current_obs_only", False):
                self._history = []
            else:
                self._history.append(query)
            if result.subgoal_tokens:
                self._last_plan = self.compiler.generated_tokens(result.subgoal_tokens, qid=-1)
                self._last_plan_index = index
            return result
        finally:
            self._recording = False
            self._pending = []

    def _append(self, segment, *, total_queries):
        output = super()._append(segment, total_queries=total_queries)
        if self._recording:
            self._pending.append((segment, total_queries))
        return output

    def _snapshot(self):
        snapshot = super()._snapshot()
        snapshot.recorded_segment_count = len(self._pending)
        return snapshot

    def _restore(self, snapshot):
        super()._restore(snapshot)
        self._pending = self._pending[:snapshot.recorded_segment_count]
