"""Privacy contracts use synthetic secrets and require no model/API calls."""
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/plus_eval'))
from evaluation import EpisodeRecorder, capture_log, write_json
from privacy import RedactingWriter, refresh_secrets, sanitize, sanitize_tree, scrub


class PrivacyTests(unittest.TestCase):
    def tearDown(self):
        refresh_secrets()

    def test_structured_fields_paths_and_preserved_metrics(self):
        original = {'Authorization': 'arbitrary opaque credential', 'OPENAI_API_KEY': 'other opaque credential',
                    'checkpoint': Path('/home/private-person/models/weights'),
                    '/home/private-person/first': 'revision-one', '/home/private-person/second': 'revision-two',
                    'usage': {'input_tokens': 12, 'output_tokens': 3}, 'path': 'assets/frame.png',
                    'subgoal': 'pick at <55, 92>', 'source': 'https://example.org/home/public/readme',
                    'tokens': [1, 2], 'role': 'user'}
        clean = sanitize(original)
        text = json.dumps(clean)
        self.assertNotIn('opaque credential', text)
        self.assertNotIn('private-person', text)
        self.assertIn('revision-one', text)
        self.assertIn('revision-two', text)
        for key in ('usage', 'path', 'subgoal', 'source', 'tokens', 'role'):
            self.assertEqual(clean[key], original[key])
        self.assertEqual(sanitize(clean), clean)
        self.assertEqual(original['Authorization'], 'arbitrary opaque credential')

    def test_known_env_secrets_and_text_credentials(self):
        for value in ('../../scripts/plus_eval/README.md', '../data/file.json', 'assets/frame.png',
                      '/files/method/BinFill_ep0/episode.mp4'):
            self.assertEqual(scrub(value), value)
        with patch.dict('os.environ', {'HF_TOKEN': 'opaqueFixtureCredential2468'}):
            refresh_secrets()
            self.assertNotIn('opaqueFixtureCredential2468', scrub('error: opaqueFixtureCredential2468'))
        for value in ('Bearer fixture123', 'api_key=opaque-fixture', '"password": "opaque-fixture"',
                      'sk-fixture-secret', 'hf_abcdefghijklmnopqrstuvwxyz',
                      'https://private-user:password@example.org/api?api_key=opaque-fixture&limit=5',
                      'https://example.org/weights?X-Amz-Signature=opaque-fixture',
                      r'C:\Users\private-person\model', r'\\private-host\private-share\file',
                      '/nfs/private-person/model', '/custom-mount/private-person/model', 'file:///home/private-person/model',
                      'Cookie: session=opaque-fixture; auth=fixture123',
                      'https://example.org/api?%61pi_key=opaque-fixture',
                      '10.12.34.56', 'private-person@example.org'):
            with self.subTest(kind=value.split(':')[0][:10]):
                cleaned = scrub(value)
                for secret in ('fixture123', 'opaque-fixture', 'sk-fixture-secret', 'abcdefghijklmnopqrstuvwxyz',
                               'private-person', 'private-user', 'private-host', '10.12.34.56'):
                    self.assertNotIn(secret, cleaned)
                self.assertEqual(scrub(cleaned), cleaned)

    def test_split_writes_and_flushes_do_not_leak(self):
        output = io.StringIO()
        writer = RedactingWriter(output)
        self.assertEqual(writer.write('Authorization: Bear'), 19)
        writer.flush()
        self.assertEqual(output.getvalue(), '')
        writer.write('er synthetic-credential\ncheckpoint=/home/private-person/model\n')
        writer.write('-----BEGIN PRIVATE KEY-----\nbody-of-secret\n-----END PRIVATE KEY-----\n')
        writer.write('trailing safe message')
        writer.finish()
        text = output.getvalue()
        for value in ('synthetic-credential', 'private-person', 'body-of-secret'):
            self.assertNotIn(value, text)
        self.assertIn('trailing safe message', text)

    def test_tracebacks_and_prints_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory, patch('sys.stdout', io.StringIO()), patch('sys.stderr', io.StringIO()):
            log = Path(directory) / 'episode.log'
            with capture_log(log):
                print('checkpoint=/home/private-person/model')
                try:
                    raise ValueError('Authorization: Bearer synthetic-credential')
                except ValueError:
                    logging.getLogger('privacy-test').exception('fixture')
            text = log.read_text()
            self.assertIn('Traceback', text)
            self.assertIn('ValueError', text)
            self.assertNotIn('synthetic-credential', text)
            self.assertNotIn('/home/', text)

    def test_recorder_sanitizes_without_changing_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            rec = EpisodeRecorder(Path(directory) / 'episode')
            output = {'subgoal': 'pick at [545, 227]', 'api_key': 'synthetic-credential'}
            answer = rec.call('ponder', lambda: output, inputs={'checkpoint': Path('/home/private-person/model')},
                              prompt='Task: pick; Authorization: Bearer synthetic-credential')
            self.assertIs(answer, output)
            self.assertEqual(answer['api_key'], 'synthetic-credential')
            for file in (rec.directory / 'calls/000000_ponder').iterdir():
                self.assertNotIn('synthetic-credential', file.read_text())
                self.assertNotIn('private-person', file.read_text())
            values = rec.artifact(np.array(['/home/private-person/model'] * 130))
            self.assertIsInstance(values, list)
            self.assertNotIn('private-person', json.dumps(values))

    def test_astra_native_exchange_is_unchanged_but_export_is_clean(self):
        spec = importlib.util.spec_from_file_location('privacy_astra', ROOT / 'scripts/plus_eval/agentic_methods/astra/three_tier/eval_plus.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            model = module.ThreeTierAstraEvaluation(SimpleNamespace())
            model.recorder = EpisodeRecorder(Path(directory) / 'episode')
            model.native_spool = tempfile.TemporaryDirectory()
            model.native_exported = {}
            source = Path(model.native_spool.name)
            def infer():
                (source / 'response.json').write_text(json.dumps({'text': 'pick', 'api_key': 'synthetic-credential'}))
                (source / 'prompt.txt').write_text('file /home/private-person/model')
                return json.loads((source / 'response.json').read_text())
            self.assertEqual(model.planner_call(infer)['api_key'], 'synthetic-credential')
            exported = model.recorder.directory / 'native/astra'
            self.assertNotIn('synthetic-credential', (exported / 'response.json').read_text())
            self.assertNotIn('private-person', (exported / 'prompt.txt').read_text())
            def fail():
                (source / 'response.json').write_text('{"password":"other-credential"}')
                raise RuntimeError('test error')
            with self.assertRaises(RuntimeError): model.planner_call(fail)
            self.assertNotIn('other-credential', (exported / 'response.json').read_text())
            model.end_episode({})
            self.assertFalse(source.exists())

    def test_migration_preserves_media_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{"api_key":"synthetic-credential","path":"/home/private-person/model"}')
            (root / 'events.jsonl').write_text('{"authorization":"synthetic-credential","input_tokens":5}\n')
            (root / 'frame.png').write_bytes(b'unchanged media')
            self.assertEqual(sanitize_tree(root)['files_needing_redaction'], 2)
            self.assertEqual(sanitize_tree(root, apply=True)['files_changed'], 2)
            self.assertEqual(sanitize_tree(root)['files_needing_redaction'], 0)
            self.assertEqual((root / 'frame.png').read_bytes(), b'unchanged media')
            self.assertEqual(json.loads((root / 'events.jsonl').read_text())['input_tokens'], 5)



if __name__ == '__main__':
    unittest.main()
