"""
HTTP version of challenge policy server.

This mirrors the websocket interface:
- POST /infer  -> returns msgpack output of policy.infer(inputs)
- POST /reset  -> returns {"reset_finished": True}
- GET  /healthz
- GET  /metadata
"""

import traceback
from flask import Flask, Response, request

from . import msgpack_numpy
from .policy import Policy


class PolicyHTTPServer:
    def __init__(
        self,
        policy: Policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._app = Flask(__name__)
        self._register_routes()

    def _register_routes(self) -> None:
        @self._app.get("/healthz")
        def healthz() -> Response:
            return Response("OK\n", status=200, mimetype="text/plain")

        @self._app.get("/metadata")
        def metadata() -> Response:
            return Response(
                msgpack_numpy.packb(self._metadata),
                status=200,
                mimetype="application/msgpack",
            )

        @self._app.post("/reset")
        def reset() -> Response:
            try:
                self._policy.reset()
                payload = msgpack_numpy.packb({"reset_finished": True})
                return Response(payload, status=200, mimetype="application/msgpack")
            except Exception:
                return Response(traceback.format_exc(), status=500, mimetype="text/plain")

        @self._app.post("/infer")
        def infer() -> Response:
            try:
                if not request.data:
                    return Response("Empty request body", status=400, mimetype="text/plain")
                inputs = msgpack_numpy.unpackb(request.data)
                outputs = self._policy.infer(inputs)
                payload = msgpack_numpy.packb(outputs)
                return Response(payload, status=200, mimetype="application/msgpack")
            except Exception:
                return Response(traceback.format_exc(), status=500, mimetype="text/plain")

    def serve_forever(self) -> None:
        print(f"Serving HTTP policy on {self._host}:{self._port}...")
        self._app.run(host=self._host, port=self._port, debug=False)

