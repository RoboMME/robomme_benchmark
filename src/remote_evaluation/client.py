
import time
from typing import Dict, Optional, Tuple
import websockets.sync.client

from . import msgpack_numpy



class PolicyClient:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        print(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers, 
                    ping_timeout=600, open_timeout=60, close_timeout=60
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
    
    
    def reset(self) -> None:
        data = self._packer.pack({"reset": True})
        self._ws.send(data)
        response = self._ws.recv()
        return msgpack_numpy.unpackb(response)
    
    
    def add_buffer(self, buffer: Dict):
        data = self._packer.pack(buffer)
        self._ws.send(data)
        response = self._ws.recv()
        return msgpack_numpy.unpackb(response)