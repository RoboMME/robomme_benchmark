"""
This is used by RoboMME challenge organizers to query the policy via HTTP.

Participants do not need to modify this file.
"""


import requests
from typing import Dict, Optional

from . import msgpack_numpy


class PolicyHTTPClient:
    """HTTP client with the same interface as PolicyClient."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = f"http://{host}"
        if port is not None:
            self._base_url += f":{port}"
        self._timeout = timeout
        self._headers = {
            "Content-Type": "application/msgpack",
            "Accept": "application/msgpack",
        }
        if api_key:
            self._headers["Authorization"] = f"Api-Key {api_key}"
        self._server_metadata = self._get_server_metadata()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _get_server_metadata(self) -> Dict:
        response = requests.get(
            f"{self._base_url}/metadata",
            headers={"Accept": "application/msgpack"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return msgpack_numpy.unpackb(response.content)

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        payload = msgpack_numpy.packb(obs)
        response = requests.post(
            f"{self._base_url}/infer",
            data=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return msgpack_numpy.unpackb(response.content)

    def reset(self) -> Dict:
        payload = msgpack_numpy.packb({"reset": True})
        response = requests.post(
            f"{self._base_url}/reset",
            data=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return msgpack_numpy.unpackb(response.content)

