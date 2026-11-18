"""Native ASPIRE workspace provisioning and isolated API/tool transport."""
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import shutil
import socketserver
import time

import requests

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE/'ASPIRE'


class AgentHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/status':
            self.reply(200, self.server.session.status())
        else:
            self.reply(404, {'error': 'Unknown endpoint'})

    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 64 * 1024**2:
                raise ValueError('Invalid request size')
            body = self.rfile.read(size)
            if self.path == '/api/run':
                self.reply(200, self.server.session.run(json.loads(body)))
                return
            if self.path.split('?')[0] != '/v1/messages':
                self.reply(404, {'error': 'Unknown endpoint'})
                return
            headers = {'Content-Type': 'application/json',
                       'Authorization': 'Bearer ' + self.server.session.token}
            for key in ('anthropic-version', 'anthropic-beta'):
                if key in self.headers:
                    headers[key] = self.headers[key]
            with requests.post(getattr(self.server.session.args, 'proxy_url', 'http://127.0.0.1:18177') + self.path, headers=headers,
                               data=body, timeout=(10, 700), stream=True) as response:
                self.send_response(response.status_code)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Connection', 'close')
                self.end_headers()
                for chunk in response.iter_content(chunk_size=4096):
                    self.wfile.write(chunk); self.wfile.flush()
                self.close_connection = True
        except Exception as exc:
            self.reply(500, {'error': {'type': 'api_error', 'message': f'{type(exc).__name__}: {exc}'}})

    def log_message(self, format, *args):
        with (self.server.session.root / 'bridge.log').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'message': format % args}) + '\n')



class AgentServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True



def provision(session):
    root, workspace = session.root, session.workspace
    for name in ('docs', '.claude/skills/iterative-debugging', 'campaign', 'published'):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    for name in ('api-reference.md', 'run-artifacts.md'):
        shutil.copy2(UPSTREAM / 'docs' / name, workspace / 'docs' / name)
    original = UPSTREAM / '.claude/skills/iterative-debugging'
    for name in ('main-agent-prompt.md', 'subagent-prompt.md', 'verifier-prompt.md'):
        shutil.copy2(original / name, workspace / '.claude/skills/iterative-debugging' / name)
    library = original / 'skills'
    shutil.copytree(library, workspace / '.claude/skills/iterative-debugging/skills')
    from benchmark_io.prompt_bundle import DEFAULT_BUNDLE
    protocol = Path(os.environ.get('ROBOMME_AGENTIC_PROMPTS', DEFAULT_BUNDLE))
    shutil.copy2(protocol/'ASPIRE.md', workspace/'docs/robomme-plus.md')
    home = root / 'claude-home'
    home.mkdir()
    model = 'us.anthropic.claude-opus-5-5'
    settings = {'permissions': {'defaultMode': 'bypassPermissions', 'deny': ['WebFetch', 'WebSearch']},
        'skipDangerousModePermissionPrompt': True, 'autoUpdates': False, 'cleanupPeriodDays': 3650,
        'env': {'ANTHROPIC_BASE_URL': 'http://127.0.0.1:17777', 'ANTHROPIC_AUTH_TOKEN': 'local-host-relay',
            'ANTHROPIC_MODEL': model, 'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
            'ANTHROPIC_DEFAULT_SONNET_MODEL': model, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': model,
            'CLAUDE_CODE_EFFORT_LEVEL': 'high', 'API_TIMEOUT_MS': '750000',
            'BASH_DEFAULT_TIMEOUT_MS': '600000', 'BASH_MAX_TIMEOUT_MS': '3700000',
            'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'ENABLE_TOOL_SEARCH': 'false'}}
    (home / 'settings.json').write_text(json.dumps(settings, indent=2))
    (home / '.claude.json').write_text(json.dumps({'hasCompletedOnboarding': True,
        'bypassPermissionsModeAccepted': True, 'projects': {'/workspace': {'hasTrustDialogAccepted': True}}}))
    (home / 'mcp.json').write_text('{"mcpServers":{}}')
    bridge = root / 'bridge'
    bridge.mkdir()
    shutil.copy2(HERE / 'agent_bridge.py', bridge / 'agent_bridge.py')
    shutil.copy2(HERE / 'agent_cli.py', bridge / 'cap-harness')
    (bridge / 'cap-harness').chmod(0o755)
    return model, home, bridge
