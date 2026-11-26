import http.client
import http.server
import queue
import socket
import socketserver
import sys
import threading
from unittest.mock import patch

import pytest


class OfflineHTTPServer(http.server.HTTPServer):
    """Run the real HTTP handler over local socket pairs with no TCP bind."""

    def __init__(self, address, handler):
        assert address == ('127.0.0.1', 0)
        self.server_name = address[0]
        self.server_port = 41234
        self.requests = queue.Queue()
        self.closed = threading.Event()
        socketserver.BaseServer.__init__(self, (self.server_name, self.server_port), handler)
        servers[self.server_port] = self

    def serve_forever(self):
        try:
            while True:
                request = self.requests.get(timeout=8)
                if request is None:
                    break
                self.process_request(request, (self.server_name, self.server_port))
        finally:
            self.closed.set()

    def shutdown(self):
        self.requests.put(None)
        assert self.closed.wait(timeout=8)

    def server_close(self):
        servers.pop(self.server_port, None)


servers = {}


def connect(connection):
    assert connection.host == '127.0.0.1'
    server = servers[connection.port]
    client, peer = socket.socketpair()
    client.settimeout(5)
    peer.settimeout(5)
    server.requests.put(peer)
    connection.sock = client


def deny_external(*args, **kwargs):
    raise OSError('network disabled for review')


with patch.object(http.server, 'HTTPServer', OfflineHTTPServer), \
        patch.object(http.client.HTTPConnection, 'connect', connect), \
        patch.object(socket, 'getaddrinfo', deny_external), \
        patch.object(socket.socket, 'connect', deny_external), \
        patch.object(socket.socket, 'connect_ex', deny_external), \
        patch.object(socket.socket, 'sendto', deny_external):
    raise SystemExit(pytest.main(sys.argv[1:]))
