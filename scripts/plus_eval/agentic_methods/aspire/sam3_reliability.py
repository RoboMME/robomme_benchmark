"""Bounded SAM3 transport retries and operator-only service diagnostics."""
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import requests
from cap_harness.providers.sam3.client import Sam3Provider


class _RecordedSession:
    def __init__(self, session, record):
        self.session, self.record = session, record

    def request(self, method, url, **kwargs):
        started = time.monotonic()
        event = dict(event='http_attempt', method=method, path=urlsplit(url).path)
        try:
            response = self.session.request(method, url, **kwargs)
            event['status_code'] = response.status_code
            return response
        except requests.RequestException as exc:
            event.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            event['seconds'] = time.monotonic() - started
            self.record(event)


class ReliableSam3Provider(Sam3Provider):
    """Use upstream retry semantics: one initial request and at most two retries.

    Only transient transport failures are retried. Empty/incorrect detections and
    invalid prompts retain their original behavior. No simulator action/reset is
    performed by a retry; controller watchdogs remain in force.
    """
    def __init__(self, base_url, *, log_path, timeout_s=60, session=None):
        self.log_path = Path(log_path)
        self.infrastructure_failures = 0
        self._log_lock = threading.Lock()
        super().__init__(base_url, timeout_s=timeout_s, max_retries=2,
                         retry_backoff_s=1.0,
                         session=_RecordedSession(session or requests.Session(), self._record))

    def _record(self, event):
        with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), provider='sam3', **event)) + '\n')

    def _checked(self, result, operation):
        error = getattr(result, 'error', None)
        code = getattr(error, 'code', None)
        code = getattr(code, 'value', code)
        if code == 'provider_unavailable':
            with self._log_lock:
                self.infrastructure_failures += 1
            self._record(dict(event='request_failed', operation=operation,
                              code=code, details=dict(error.details)))
        return result

    def segment_text(self, *args, **kwargs):
        return self._checked(super().segment_text(*args, **kwargs), 'segment_text')

    def segment_points(self, *args, **kwargs):
        return self._checked(super().segment_points(*args, **kwargs), 'segment_points')
