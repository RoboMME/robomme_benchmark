"""Exercise the real multimodal compiler, cache append and rollback on CPU."""
from collections import deque
import contextlib
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "scripts/plus_eval/non_agentic_methods/ponderpounce"
sys.path[:0] = [str(HERE), str(HERE / "PonderPounce")]
try:
    HAS_DEPS = tuple(map(int, importlib.metadata.version("transformers").split(".")[:2])) >= (5, 13)
except importlib.metadata.PackageNotFoundError:
    HAS_DEPS = False

if HAS_DEPS:
    import torch
    from context_window import WindowedS2Context
    from ponderpounce.inference.append_context import SoftS2SessionContext
    from ponderpounce.inference.session import Session


class Tokenizer:
    special = {"<|fim_prefix|>": 1, "<|fim_pad|>": 2,
               "<|vision_start|>": 3, "<|image_pad|>": 4, "<|vision_end|>": 5}
    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return self.special[token]

    def convert_ids_to_tokens(self, token):
        return next(k for k, v in self.special.items() if v == token)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[10 + ord(c) % 30 for c in text])

    def decode(self, ids, **kwargs):
        return "".join("<|fim_prefix|>" if t == 1 else "go" for t in ids)


class Processor:
    tokenizer = Tokenizer()
    merge_size = 1
    grid_size = 2

    @property
    def image_processor(self):
        return self

    def __call__(self, *, images, **kwargs):
        # Four visual tokens per image. Keep image identity in pixel_values.
        return {"image_grid_thw": torch.tensor([[1, self.grid_size, self.grid_size]] * len(images)),
                "pixel_values": torch.tensor([[float(i)] for i in images]).repeat_interleave(self.grid_size ** 2, 0)}


class System2:
    """Small deterministic stand-in with both KV and recurrent cache state."""
    effective_context_cap = 16384
    num_cognition_tokens = 1
    _subgoal_trigger_id, _cognition_token_id = 1, 2
    current_obs_only = False

    def __init__(self):
        self.next_tokens = deque()
        self.created_caches = 0
        self.image_batches = []
        self.fail_next = False
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=1))
        model = SimpleNamespace(config=config, get_vision_position_ids=self.vision_positions)
        self.vlm = SimpleNamespace(config=config, model=model, lm_head=self.head)

    @staticmethod
    def vision_positions(position, grid, *args, device, **kwargs):
        return torch.arange(int(grid.prod()), device=device).repeat(3, 1) + position

    def head(self, hidden):
        logits = torch.full((1, 64), -100.)
        logits[0, self.next_tokens.popleft() if self.next_tokens else 2] = 100
        return logits

    def _new_static_cache(self):
        self.created_caches += 1
        return SimpleNamespace(layers=[
            SimpleNamespace(keys=torch.zeros(self.effective_context_cap), cumulative_length=torch.tensor(0),
                            max_cache_len=self.effective_context_cap),
            SimpleNamespace(conv_states=torch.zeros(1), recurrent_states=torch.zeros(1),
                            is_conv_states_initialized=True, is_recurrent_states_initialized=True,
                            has_previous_state=False)])

    def _build_inputs_embeds(self, *, input_ids, cognition_mask, pixel_values, image_grid_thw, s2_total_queries):
        assert int(cognition_mask.sum()) == s2_total_queries * self.num_cognition_tokens
        if pixel_values is not None:
            assert int((input_ids == 4).sum()) == len(pixel_values)
            assert int(image_grid_thw.prod(dim=1).sum()) == len(pixel_values)
            self.image_batches.append(pixel_values[:, 0].tolist())
        return input_ids.float().unsqueeze(-1)

    def _isolation_stash(self, *args, **kwargs):
        return contextlib.nullcontext()

    def _run_cached_language_model(self, *, inputs_embeds, past_key_values, attention_mask, **kwargs):
        full, linear = past_key_values.layers
        start = int(full.cumulative_length)
        ids = inputs_embeds[0, :, 0]
        full.keys[start:start + len(ids)] = ids
        full.cumulative_length.add_(len(ids))
        linear.conv_states.add_(ids.sum())
        linear.recurrent_states.add_(ids.sum())
        linear.has_previous_state = True
        assert attention_mask.shape[-1] == int(full.cumulative_length)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected forward failure")
        return SimpleNamespace(last_hidden_state=inputs_embeds)


@unittest.skipUnless(HAS_DEPS, "Run in the PonderPounce environment")
class PonderContextTest(unittest.TestCase):
    def make_context(self, cap=128, demos=(), current_only=False, native=False, max_new_tokens=4):
        model = System2()
        model.current_obs_only = current_only
        session = Session("test", None, task_text="move cube", demo_images=list(demos), s2_processor=Processor())
        kwargs = dict(system2=model, session=session, device="cpu", dtype=torch.float32, max_new_tokens=max_new_tokens)
        if native:
            return SoftS2SessionContext(**kwargs)
        return WindowedS2Context(context_tokens=cap, **kwargs)

    def assert_cache_consistent(self, ctx):
        full, linear = ctx.cache.layers
        self.assertEqual(int(full.cumulative_length), ctx._context_len)
        self.assertLessEqual(ctx._context_len, ctx.context_cap)
        expected = full.keys[:ctx._context_len].sum()
        torch.testing.assert_close(linear.recurrent_states[0], expected)
        torch.testing.assert_close(linear.conv_states[0], expected)
        self.assertEqual(ctx._cum_cm.shape[-1], ctx._context_len)

    def test_under_limit_matches_original_context(self):
        bounded, original = self.make_context(cap=1024, demos=[100, 101]), self.make_context(demos=[100, 101], native=True)
        for index in range(10):
            for ctx in (bounded, original):
                ctx.s2.next_tokens.extend([1, 20, 2] if index == 2 else [2])
            got, expected = bounded.fire([index, index + 100]), original.fire([index, index + 100])
            self.assertEqual((got.kind, got.subgoal_tokens, got.n_input_frames),
                             (expected.kind, expected.subgoal_tokens, expected.n_input_frames))
            torch.testing.assert_close(got.cognition, expected.cognition)
            torch.testing.assert_close(bounded._cum_qid, original._cum_qid)
        self.assertEqual(bounded.truncation_count, 0)
        self.assert_cache_consistent(bounded)

    def test_many_rollovers_keep_latest_images_and_rebuild_both_cache_types(self):
        ctx = self.make_context()
        for index in range(100):
            ctx.fire([index, index + 100])
            self.assertEqual(ctx.query_index, index + 1)
            self.assertEqual(ctx.n_obs_frames, index + 1)
            self.assertEqual(ctx.s2.image_batches[-1], [index] * 4 + [index + 100] * 4)
            self.assert_cache_consistent(ctx)
        self.assertGreater(ctx.truncation_count, 10)
        self.assertEqual(ctx.s2.created_caches, ctx.truncation_count + 1)
        self.assertGreater(ctx._history[0].index, 0)
        self.assertEqual(ctx._history[-1].index, 99)
        self.assertEqual(ctx.last_truncation["generation_reserve"], 5)
        self.assertGreater(ctx._prefix.length, 0)  # task survives removal of first query

    def test_oversized_demo_keeps_whole_uniformly_sampled_frames(self):
        ctx = self.make_context(demos=range(100))
        self.assertLessEqual(ctx._context_len + ctx.generation_reserve, 128)
        indices = ctx.last_truncation["demo_frame_indices"]
        self.assertGreater(len(indices), 1)
        self.assertEqual((indices[0], indices[-1]), (0, 99))
        self.assertLess(len(indices), 100)
        for _ in range(25):
            ctx.fire([1000, 1001])
            self.assert_cache_consistent(ctx)
        self.assertEqual(ctx.session.demo_images, list(range(100)))

    def test_keeps_latest_generated_plan_after_its_query_is_removed(self):
        ctx = self.make_context()
        ctx.s2.next_tokens.extend([1, 20, 2])
        self.assertEqual(ctx.fire([0, 100]).kind, "transition")
        for i in range(25):
            ctx.fire([i, i + 100])
        self.assertGreater(ctx._history[0].index, 0)
        self.assertTrue(ctx.last_truncation["latest_plan_preserved_separately"])
        full = ctx.cache.layers[0]
        torch.testing.assert_close(full.keys[ctx._prefix.length:ctx._prefix.length + 2], torch.tensor([1., 20.]))
        self.assert_cache_consistent(ctx)

    def test_incomplete_generation_is_rolled_back_before_retaining_query(self):
        ctx = self.make_context()
        ctx.s2.next_tokens.extend([1, 20, 21, 22])  # no cognition terminator in response budget
        result = ctx.fire([1, 2])
        self.assertIsNone(result.subgoal_tokens)
        self.assertNotIn(1, ctx._history[0].segment.input_ids.tolist())
        self.assertEqual(ctx._history[0].total_queries, 1)
        for i in range(20):
            ctx.fire([i, i + 100])
        self.assert_cache_consistent(ctx)

    def test_forward_failure_restores_cache_and_does_not_commit_history(self):
        ctx = self.make_context(demos=[99])
        ctx.fire([1, 2])
        length, index = ctx._context_len, ctx.query_index
        ctx.s2.fail_next = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            ctx.fire([3, 4])
        self.assertEqual((ctx._context_len, ctx.query_index, len(ctx._history)), (length, index, 1))
        self.assert_cache_consistent(ctx)
        ctx.fire([5, 6])
        self.assert_cache_consistent(ctx)

    def test_current_observation_only_uses_reset_length_for_budget(self):
        ctx = self.make_context(cap=50, demos=[99], current_only=True)
        for i in range(30):
            ctx.fire([i, i + 100])
        self.assertEqual(ctx.truncation_count, 0)
        self.assertEqual(ctx.query_index, 30)
        self.assertEqual(ctx._history, [])
        self.assert_cache_consistent(ctx)

    def test_reserves_worst_case_response_including_trigger_and_cognition(self):
        ctx = self.make_context(cap=70, max_new_tokens=20)
        for i in range(20):
            ctx.s2.next_tokens.extend([1] + [20] * 18 + [2])
            self.assertEqual(ctx.fire([i, i + 100]).kind, "transition")
            self.assert_cache_consistent(ctx)
        self.assertGreater(ctx.truncation_count, 0)

    def test_native_16384_limit_with_two_64_token_camera_images(self):
        bounded, original = self.make_context(cap=16384, max_new_tokens=320), self.make_context(native=True, max_new_tokens=320)
        bounded.session.s2_processor.grid_size = 8
        original.session.s2_processor.grid_size = 8
        # Reproduce the upstream native-cap failure with realistic 132-token
        # front+wrist observations, then pass the same boundary with trimming.
        from ponderpounce.model.system2 import ContextOverflowError
        with self.assertRaises(ContextOverflowError):
            for i in range(140):
                original.fire([i, i + 1000])
        for i in range(140):
            bounded.fire([i, i + 1000])
            self.assert_cache_consistent(bounded)
        self.assertGreater(bounded.truncation_count, 0)
        self.assertEqual(bounded.query_index, 140)

    def test_impossible_budget_and_invalid_cap_fail_clearly(self):
        for cap in (-1, 0, 16385):
            with self.assertRaisesRegex(ValueError, "context tokens"):
                self.make_context(cap=cap)
        ctx = self.make_context(cap=20)
        with self.assertRaisesRegex(ValueError, "cannot fit the task"):
            ctx.fire([1, 2])


if __name__ == "__main__":
    unittest.main()
