import json
import threading
import urllib.error
import urllib.request

import pytest

from benchmark_io.bedrock_proxy import BedrockServer, MODEL, bedrock_body


def payload(**kwargs):
    return dict(model=MODEL, max_tokens=64, output_config={"effort":"high"},
                messages=[{"role":"user","content":"test"}], **kwargs)


def test_preserves_native_messages_and_beta_controls():
    request = payload(stream=True)
    request["messages"].append(dict(role="system", content="instructions", output_config={"effort":"high"}))
    body = bedrock_body(request, "thinking-display-updates-2026-08-18, mid-conversation-output-config-2026-07-01, extended-cache-ttl-2025-04-11")
    assert body["messages"] == request["messages"]
    assert body["anthropic_beta"] == ["thinking-display-updates-2026-08-18", "mid-conversation-output-config-2026-07-01"]
    assert "stream" not in body and "model" not in body
    assert body["output_config"] == {"effort":"high"}
    with pytest.raises(ValueError):
        bedrock_body({**request,"model":"another-model"})
    request["messages"][-1]["output_config"]["effort"] = "medium"
    with pytest.raises(ValueError):
        bedrock_body(request)


def test_transport_auth_stream_logging_and_call_cap(tmp_path):
    events = [dict(type="message_start", message={"usage":{"input_tokens":12}}),
              dict(type="content_block_start", index=0,
                   content_block=dict(type="tool_use", id="tool_1", name="Read", input={})),
              dict(type="message_delta", usage={"output_tokens":3}), dict(type="message_stop")]
    class Client:
        calls = []
        def invoke_model_with_response_stream(self, **kwargs):
            self.calls.append(kwargs)
            return dict(body=[{"chunk":{"bytes":json.dumps(e).encode()}} for e in events])
    client = Client()
    server = BedrockServer(("127.0.0.1",0),client,tmp_path,"test-token",1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(token):
        return urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/messages", data=json.dumps(payload(stream=True)).encode(),
            headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"}), timeout=5)
    try:
        with pytest.raises(urllib.error.HTTPError) as denied:
            request("wrong")
        assert denied.value.code == 401 and not client.calls
        with request("test-token") as response:
            content = response.read().decode()
        assert '"id": "tool_1"' in content and "event: message_stop" in content
        with pytest.raises(urllib.error.HTTPError) as capped:
            request("test-token")
        assert capped.value.code == 403 and len(client.calls) == 1
        logged = next(tmp_path.glob("*/events.jsonl"))
        assert [json.loads(line) for line in logged.read_text().splitlines()] == events
        assert "test-token" not in "".join(p.read_text() for p in tmp_path.glob("*/*.json"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('extra', [
    {'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}]},
    {'mcp_servers': [{'url': 'https://example.com'}]},
    {'container': {}},
    {'messages': [{'role': 'user', 'content': [{'type': 'image',
        'source': {'type': 'url', 'url': 'https://example.com/image.png'}}]}]},
])
def test_provider_cannot_bypass_offline_agent_network(extra):
    request = payload()
    request.update(extra)
    with pytest.raises(ValueError):
        bedrock_body(request)


def test_proxy_restart_keeps_call_cap(tmp_path):
    (tmp_path / '0007_saved').mkdir()
    server = BedrockServer(('127.0.0.1', 0), object(), tmp_path, 'test-token', 10)
    try:
        assert server.calls == 7
    finally:
        server.server_close()


def test_timeout_with_null_provider_response_returns_retryable_error_and_keeps_cost_record(tmp_path):
    class ProviderTimeout(Exception):
        response = None
    class Client:
        def invoke_model_with_response_stream(self, **kwargs):
            raise ProviderTimeout('Provider read timed out')
    server = BedrockServer(('127.0.0.1', 0), Client(), tmp_path, 'test-token', 1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/messages',
        data=json.dumps(payload(stream=True)).encode(),
        headers={'Authorization': 'Bearer test-token', 'Content-Type': 'application/json'})
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        assert caught.value.code == 502
        assert 'Provider read timed out' in caught.value.read().decode()
    finally:
        server.shutdown();server.server_close();thread.join()
    assert json.loads(next(tmp_path.glob('*/error.json')).read_text())['type'] == 'ProviderTimeout'
    assert next(tmp_path.glob('*/cost.json')).is_file()
