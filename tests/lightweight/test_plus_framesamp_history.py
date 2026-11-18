"""Long OOD executions retain valid, distinct FrameSamp position embeddings."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "scripts/plus_eval/non_agentic_methods/framesamp_modul/robomme_policy_learning"


class FrameSampHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(POLICY / "src"), str(POLICY / "packages/openpi-client/src")]
        try:
            import jax.numpy as jnp
            from mme_vla_suite.shared.mem_buffer import MemoryBuffer
        except ImportError as exc:
            raise unittest.SkipTest(f"FrameSamp server dependencies unavailable: {exc}")
        cls.jnp = jnp
        cls.MemoryBuffer = MemoryBuffer

    def test_frame_sampling_across_the_position_cache_boundary(self):
        for views in (1, 2):
            with self.subTest(views=views):
                memory = self.MemoryBuffer(
                    num_views=views, img_emb_dim=12, pos_emb_dim=12,
                    max_steps=4096, prepare_buffer=True,
                    vision_enc_fn=lambda images: self.jnp.zeros((len(images), views, 64, 12)),
                )
                boundary = 4096 // views
                original_cache = memory.pos_emb_dict
                for indices in ([0], [boundary - 1, boundary], [boundary + 1]):
                    images = np.zeros((len(indices), views, 16, 16, 3), dtype=np.uint8)
                    memory.add_buffer(images, np.zeros((len(indices), 8)), indices)
                selected = [0, boundary - 1, boundary, boundary + 1]
                for step in selected:
                    for size in (8, 4, 2):
                        positions = self.jnp.arange(step * views, (step + 1) * views)
                        expected = np.array(memory.pos_embedder(positions, size))
                        actual = memory._history_feats[step][f"pos_emb_{size}x{size}"]
                        np.testing.assert_array_equal(actual, expected)
                image, position, state, mask = memory._prepare_frame_sampling(
                    memory._history_feats, selected, token_budget=80 * views, token_per_image=16)
                self.assertEqual(position.shape, (80 * views, 12))
                self.assertEqual(image.shape, (80 * views, 12))
                self.assertEqual(state.shape, (80 * views, 8))
                self.assertEqual(int(mask.sum()), 64 * views)
                self.assertIs(memory.pos_emb_dict, original_cache)
                self.assertEqual(len(original_cache["8x8"]), 4096)
                self.assertFalse(np.array_equal(
                    memory._history_feats[boundary - 1]["pos_emb_4x4"],
                    memory._history_feats[boundary]["pos_emb_4x4"]))


if __name__ == "__main__":
    unittest.main()
