"""Resolve the HTTPServer's numeric loopback server name without invoking DNS."""

import socket
from unittest.mock import patch

from offline_review import loopback


def pytest_configure(config):
    def numeric_name(host=""):
        if loopback(host):
            return host
        raise AssertionError(f"unexpected getfqdn: {host!r}")

    guard = patch.object(socket, "getfqdn", numeric_name)
    guard.start()
    config.add_cleanup(guard.stop)
