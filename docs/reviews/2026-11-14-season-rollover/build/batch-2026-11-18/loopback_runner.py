import socket

import offline
import review_runner


def local_reverse_lookup(host):
    if offline._loopback(host):
        return "localhost", [], [host]
    raise AssertionError(f"DNS forbidden: reverse lookup {host!r}")


socket.gethostbyaddr = local_reverse_lookup
review_runner.__file__ = __file__
raise SystemExit(review_runner.main())
