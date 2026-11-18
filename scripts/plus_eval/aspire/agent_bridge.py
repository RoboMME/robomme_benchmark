"""Loopback-to-Unix socket bridge in an offline coding-agent container."""
import os
import select
import socket
import socketserver
import subprocess
import sys
import threading


class Bridge(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX) as peer:
            peer.connect('/run/aspire/agent.sock')
            sockets = [self.request, peer]
            while True:
                readable, _, _ = select.select(sockets, [], [], 3600)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(262144)
                    if not data:
                        return
                    (peer if source is self.request else self.request).sendall(data)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    with Server(('127.0.0.1', 17777), Bridge) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        result = subprocess.call(sys.argv[1:])
        server.shutdown()
    raise SystemExit(result)


if __name__ == '__main__':
    main()
