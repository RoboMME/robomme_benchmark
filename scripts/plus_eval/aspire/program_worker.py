"""Run upstream ASPIRE ProgramExecutor inside the simulator-free container."""
import json
from pathlib import Path
import socket
import sys

from cap_harness.registry import ToolRegistry, SHARED_PUBLIC_TOOL_NAMES, ROBOMME_PLUS_PUBLIC_TOOL_NAMES
from cap_harness.runtime import ProgramExecutor
from wire import dumps, read_reply


def call(name, *args, **kwargs):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect('/run/aspire/tool.sock')
        stream = connection.makefile('rwb')
        stream.write(dumps({'name': name, 'args': args, 'kwargs': kwargs}) + b'\n')
        stream.flush()
        result = read_reply(stream, video=name == 'robomme_plus.get_conditioning_video')
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result['value']


def main():
    registry = ToolRegistry(public_extension_allowlist=ROBOMME_PLUS_PUBLIC_TOOL_NAMES)
    for name in sorted(SHARED_PUBLIC_TOOL_NAMES | ROBOMME_PLUS_PUBLIC_TOOL_NAMES):
        def function(*args, _name=name, **kwargs):
            return call(_name, *args, **kwargs)
        registry.register(function, name=name, layer='shared_atomic', capability='remote_public_api', public=True)
    result = ProgramExecutor(registry).execute_program(Path('/program/program.py').read_text())
    sys.stdout.buffer.write(dumps(result) + b'\n')


if __name__ == '__main__':
    main()
