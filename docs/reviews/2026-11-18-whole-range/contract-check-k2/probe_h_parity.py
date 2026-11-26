import json
import runpy
import sys
import tempfile
from pathlib import Path

import bts
from bts.daily_decision import write_decision

from probe_callers_v2 import offline


def main():
    src = Path(bts.__file__).resolve().parents[1]
    probe = runpy.run_path("tests/test_wr_h_longest.py")["parity_fingerprint"]
    with tempfile.TemporaryDirectory() as tmp:
        record = write_decision("2027-07-15", tmp, objective="longest_streak", action="single",
                                source="mdp", delivery_status="delivered", scoreable=True)
        probe(src, json.dumps(record), sys.argv[1])


if __name__ == "__main__":
    import socket
    from unittest.mock import patch

    with patch.object(socket, "getaddrinfo", offline), patch.object(socket.socket, "connect", offline):
        main()
