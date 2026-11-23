"""Operator-only Anthropic Messages transport for the native upstream CLIs.

No agent loop or policy logic. RAX's existing custom-provider relay reaches this
loopback endpoint; AWS credentials never enter its network-isolated container.
Requests, provider events and usage are kept outside the agent workspace.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid

from .costs import record_usage, RATE_CARD


MODEL = "us.anthropic.claude-opus-5-5"


def inline_media(value):
    """The model provider must not fetch URLs or files on an agent's behalf."""
    if isinstance(value, list):
        return all(inline_media(item) for item in value)
    if isinstance(value, dict):
        source = value.get('source')
        if isinstance(source, dict) and source.get('type') not in ('base64', 'text', 'content'):
            return False
        return all(inline_media(item) for item in value.values())
    return True


def bedrock_body(payload, beta_header="", expected_effort="high"):
    if payload.get("model") != MODEL:
        raise ValueError("This experiment permits only the configured Opus model")
    if any(key in payload for key in ('mcp_servers', 'container')):
        raise ValueError('Provider-side servers and containers are forbidden')
    local_tools = payload.get('tools', [])
    if not isinstance(local_tools, list) or any(
        not isinstance(tool, dict) or tool.get('type') not in (None, 'custom')
        or not isinstance(tool.get('name'), str) for tool in local_tools
    ):
        raise ValueError('Only client-executed tools are allowed; provider-side tools are forbidden')
    if not inline_media(payload.get('messages', [])) or not inline_media(payload.get('system', [])):
        raise ValueError('Only inline media is allowed; provider URL fetching is forbidden')
    body = {k: v for k, v in payload.items() if k not in ("model", "stream")}
    body["anthropic_version"] = "bedrock-2023-05-31"
    # InvokeModel represents Anthropic's beta header in the request body. Keep
    # these flags: Claude Code uses them for mid-conversation effort/thinking.
    if beta_header:
        # Bedrock exposes one-hour caching as GA and rejects this Anthropic-only
        # beta name. Retain the cache_control TTLs themselves unchanged.
        body["anthropic_beta"] = [v.strip() for v in beta_header.split(",")
                                  if v.strip() and v.strip() != "extended-cache-ttl-2025-04-11"]
    # Native Claude Code may send a side request without reasoning options.
    # Explicit reasoning requests must retain the experiment's configured effort.
    effort = body.get("output_config", {}).get("effort")
    if effort not in (None, expected_effort):
        raise ValueError(f"Reasoning effort must be {expected_effort}")
    if any(m.get("output_config", {}).get("effort") not in (None, expected_effort)
           for m in body.get("messages", [])):
        raise ValueError(f"Per-message reasoning effort must be {expected_effort}")
    return body


class BedrockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, client, root, token, max_calls, *, effort="high", context=None):
        self.client, self.root, self.token = client, Path(root), token
        self.calls, self.max_calls, self.lock = 0, max_calls, threading.Lock()
        self.effort, self.context = effort, context or {}
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = dict(effort=effort, context=self.context, rate_card=RATE_CARD)
        pin = self.root / "accounting_config.json"
        if pin.exists() and json.loads(pin.read_text()) != metadata:
            raise ValueError("Cannot change effort or cost attribution on an existing relay log")
        pin.write_text(json.dumps(metadata, indent=2))
        # A transport restart must preserve request identifiers and the call cap.
        ordinals = [int(p.name.split('_', 1)[0]) for p in self.root.iterdir()
                    if p.is_dir() and p.name.split('_', 1)[0].isdigit()]
        self.calls = max(ordinals, default=0)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def error(self, status, message):
        self.reply(status, dict(type="error", error=dict(type="api_error", message=message)))

    def do_POST(self):
        if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            self.error(401, "Invalid operator relay credential")
            return
        endpoint = urlsplit(self.path).path
        if endpoint not in ("/v1/messages", "/v1/messages/count_tokens"):
            self.error(404, "Unsupported endpoint")
            return
        started = False
        record = None
        usage, usage_complete, client_disconnected = {}, False, False
        try:
            size = int(self.headers.get("Content-Length", "-1"))
            if not 0 < size <= 128 * 1024 * 1024:
                raise ValueError("Invalid request length")
            payload = json.loads(self.rfile.read(size))
            body = bedrock_body(payload, self.headers.get("anthropic-beta", ""), self.server.effort)
            if endpoint.endswith("count_tokens"):
                if not hasattr(self.server.client, "count_tokens"):
                    self.error(501, "Installed Bedrock SDK has no token-count endpoint")
                    return
                result = self.server.client.count_tokens(modelId=MODEL, input={"invokeModel": {"body": json.dumps(body)}})
                self.reply(200, {"input_tokens": result["inputTokens"]})
                return
            with self.server.lock:
                if self.server.calls >= self.server.max_calls:
                    self.error(403, "Operator API-call limit reached; no automatic retries")
                    return
                self.server.calls += 1
                ordinal = self.server.calls
            record = self.server.root / f"{ordinal:04d}_{uuid.uuid4().hex[:8]}"
            record.mkdir()
            (record / "request.json").write_text(json.dumps(payload))
            (record / "bedrock_request.json").write_text(json.dumps(body))
            (record / "transport.json").write_text(json.dumps(dict(model=MODEL, started=time.time(),
                effort=self.server.effort, context=self.server.context,
                body_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest())))
            kwargs = dict(modelId=MODEL, body=json.dumps(body), contentType="application/json", accept="application/json")
            if payload.get("stream"):
                response = self.server.client.invoke_model_with_response_stream(**kwargs)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                self.close_connection = True
                with (record / "events.jsonl").open("w") as events:
                    for event in response["body"]:
                        if "chunk" not in event:
                            raise RuntimeError(f"Bedrock stream error: {event}")
                        data = json.loads(event["chunk"]["bytes"])
                        events.write(json.dumps(data) + "\n")
                        events.flush()
                        usage.update(data.get("message", {}).get("usage", {}))
                        usage.update(data.get("usage", {}))
                        usage_complete |= data.get("type") == "message_stop"
                        packet = f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()
                        if not client_disconnected:
                            try:
                                self.wfile.write(packet)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                client_disconnected = True  # Drain provider usage even if CLI disconnected.
            else:
                response = self.server.client.invoke_model(**kwargs)
                result = json.loads(response["body"].read())
                (record / "response.json").write_text(json.dumps(result))
                usage.update(result.get("usage", {}))
                usage_complete = "usage" in result
                self.reply(200, result)
            (record / "completed.json").write_text(json.dumps(dict(ended=time.time(), ok=True, client_disconnected=client_disconnected)))
        except Exception as exc:
            if record:
                (record / "error.json").write_text(json.dumps(dict(type=type(exc).__name__, message=str(exc))))
            if not started:
                provider_code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
                status = {"ValidationException":400, "AccessDeniedException":403,
                          "ExpiredTokenException":401, "ThrottlingException":429}.get(
                              provider_code, 400 if isinstance(exc, ValueError) else 502)
                self.error(status, f"{type(exc).__name__}: {exc}")
            self.close_connection = True
        finally:
            if record:
                record_usage(record, usage, complete=usage_complete, context=self.server.context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--port", type=int, default=18177)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--effort", choices=("medium", "high"), default="high")
    parser.add_argument("--context", type=Path, help="Operator-owned method/setting/budget/phase JSON")
    args = parser.parse_args()
    import boto3
    from botocore.config import Config
    token = args.token_file.read_text().strip()
    if args.token_file.stat().st_mode & 0o077:
        raise ValueError("Relay token must be owner-readable only")
    client = boto3.Session(profile_name=args.profile, region_name=args.region).client("bedrock-runtime",
        config=Config(connect_timeout=10, read_timeout=600, retries={"max_attempts": 0}))
    BedrockServer(("127.0.0.1", args.port), client, args.output, token, args.max_calls,
        effort=args.effort, context=json.loads(args.context.read_text()) if args.context else {}).serve_forever()


if __name__ == "__main__":
    main()
