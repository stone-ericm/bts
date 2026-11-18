import datetime as dt
import socket
from unittest.mock import patch

from review_runner import revision_probe


def offline_probe(probe, compare):
    import sitecustomize

    def verify():
        assert sitecustomize.loopback('127.0.0.1') and sitecustomize.loopback('::1')
        with socket.socket() as sock:
            for attempt in (lambda: socket.getaddrinfo('example.invalid', 80),
                            lambda: sock.connect(('192.0.2.1', 80))):
                try:
                    attempt()
                except AssertionError:
                    continue
                raise AssertionError('network guard allowed an outward operation')

    with patch.object(sitecustomize, 'verify', verify):
        return revision_probe(probe, compare)


def freeze(instant):
    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.astimezone().replace(tzinfo=None)

    return Clock
