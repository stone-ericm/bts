"""Give HTTPServer a loopback name without issuing a reverse lookup."""

import socket

from sitecustomize import loopback


def pytest_configure(config):
    original = socket.gethostbyaddr

    def gethostbyaddr(host):
        if loopback(host):
            return "localhost", [], [host]
        return original(host)

    config.add_cleanup(lambda: setattr(socket, "gethostbyaddr", original))
    socket.gethostbyaddr = gethostbyaddr
