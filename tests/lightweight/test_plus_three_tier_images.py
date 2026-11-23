"""Three-tier limits count attached sheets and stop before oversized API calls."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/plus_eval'
sys.path.insert(0, str(SCRIPTS))
from evaluation import EpisodeStop
sys.path.insert(0, str(SCRIPTS / "agentic_methods/astra/three_tier"))

from three_tier_images import demonstration_indices, request_image_budget

spec = importlib.util.spec_from_file_location('ood_three_tier_adapter', SCRIPTS / 'agentic_methods/astra/three_tier/eval_plus.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ThreeTierImagesTest(unittest.TestCase):
    def owner(self):
        owner = object.__new__(adapter.ThreeTierAstraEvaluation)
        owner.image_budget = request_image_budget()
        owner.recorder = SimpleNamespace(call=lambda label, function, **kwargs: function(), event=lambda *args: None)
        return owner

    def test_small_demo_keeps_every_frame_and_counts_sheets(self):
        indices, report = demonstration_indices(33)
        self.assertEqual(indices, list(range(33)))
        self.assertEqual(report['input_images'], 5)  # Three sheets and two fixed views.
        self.assertFalse(report['applied'])

    def test_large_demo_preserves_endpoints_and_reserves_execution_memory(self):
        for memory in (0, 1, 17):
            with self.subTest(memory=memory):
                indices, report = demonstration_indices(25000, initial_memory_frames=memory)
                self.assertEqual((indices[0], indices[-1]), (0, 24999))
                self.assertEqual(len(indices), len(set(indices)))
                self.assertEqual(report['input_images'], 750)
                self.assertTrue(report['applied'])
                self.assertGreater(len(indices), 750)  # The API receives sheets, not individual frames.

    def test_oversized_request_stops_before_transport_or_api_started_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'request.json').write_text(json.dumps({'images': ['unused.jpg'] * 1501}))
            owner = self.owner()
            responder = adapter.recorded_responder(owner, 'sk-unit-test-only')
            with patch.object(adapter.urllib.request, 'urlopen') as send, self.assertRaises(EpisodeStop):
                responder(out)
            send.assert_not_called()
            self.assertFalse((out / 'api_started.json').exists())
            self.assertEqual(owner.image_budget['stop_source'], 'local')

    def test_api_image_rejection_is_an_episode_stop_instead_of_a_retryable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'request.json').write_text(json.dumps({'images': []}))
            (out / 'prompt.txt').write_text('test prompt')
            owner = self.owner()
            responder = adapter.recorded_responder(owner, 'sk-unit-test-only')
            error = urllib.error.HTTPError('https://example.invalid', 400, 'image limit', {},
                io.BytesIO(json.dumps({'error': {'code': 'max_images_per_request'}}).encode()))
            with patch.object(adapter.urllib.request, 'urlopen', side_effect=error) as send, \
                 self.assertRaises(EpisodeStop):
                responder(out)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(owner.image_budget['stop_source'], 'api')

    def test_independent_requests_have_no_cumulative_episode_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owner = self.owner()
            responder = adapter.recorded_responder(owner, 'sk-unit-test-only')
            class Response:
                status = 200
                headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self):
                    return json.dumps({'status': 'completed', 'model': 'gpt-6-astra',
                        'reasoning': {'effort': 'medium'}, 'output': []}).encode()
            with patch.object(adapter.urllib.request, 'urlopen', return_value=Response()):
                # Use a payload-level count stub without creating thousands of files.
                with patch.object(adapter, 'image_count', return_value=1500):
                    for i in range(2):
                        out = root / str(i)
                        out.mkdir()
                        (out / 'request.json').write_text('{"images": []}')
                        (out / 'prompt.txt').write_text('test prompt')
                        responder.next_request_at = 0
                        responder(out)
            self.assertFalse(owner.image_budget['stopped'])
            self.assertEqual(owner.image_budget['submitted_images'], 3000)


if __name__ == '__main__':
    unittest.main()
