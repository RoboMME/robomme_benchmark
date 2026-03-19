"""
Adapted from https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py

Participant do not need to modify this file
"""

import asyncio
import http
import traceback
import websockets.asyncio.server as _server
import websockets.frames

from .policy import Policy
from . import msgpack_numpy



class PolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

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

    def serve_forever(self) -> None:
        print(f"Serving policy on {self._host}:{self._port}...")
        print("Waiting for client to connect...")
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        print(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        
        while True:
            try:
                inputs = msgpack_numpy.unpackb(await websocket.recv())
                
                if inputs.get("reset", False):
                    self._policy.reset()
                    await websocket.send(packer.pack({"reset_finished": True}))
                else:
                    outputs = self._policy.infer(inputs)
                    await websocket.send(packer.pack(outputs))

            except websockets.ConnectionClosed:
                print(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
