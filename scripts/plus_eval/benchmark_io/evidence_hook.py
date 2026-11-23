"""Container-side audit transport; no host filesystem or simulator capability."""
import json
import socket
import sys


def main():
    event = json.load(sys.stdin)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(110)
            connection.connect('/run/robomme-audit/audit.sock')
            connection.sendall(json.dumps(event).encode()+b'\n')
            with connection.makefile('rb') as stream:
                response = json.loads(stream.readline())
            if not response.get('ok'):
                raise RuntimeError('Operator could not archive development evidence')
    except Exception:
        # PreToolUse exit 2 prevents the unrecorded tool call. On Stop/SubagentStop,
        # exit 2 would forbid the agent from ever stopping, so other events report
        # a non-blocking error; reconciliation must flag the missing record.
        print('Development audit recorder unavailable; preserve this failure.', file=sys.stderr)
        return 2 if event.get('hook_event_name') == 'PreToolUse' else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
