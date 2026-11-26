"""Check the two warning bodies and two refusal bodies against the contract."""
import re
import socket
import tempfile
from pathlib import Path

from pytest import MonkeyPatch
from probe_text import HERE, archived, ops


def expected(label, **values):
    lines = ops.CONTRACT.read_text().splitlines()
    template, = [line.split("`")[1] for line in lines if line.startswith(f"- **{label}:** `")]
    if label == "In the morning":
        amendment = next(line for line in lines if "**Codex B1 (the morning warning).**" in line)
        old, new = re.findall(r"`([^`]+)`", amendment)[-2:]
        assert template.count(old) == 1
        template = template.replace(old, new)
    return template.format(**values)


ops.exact_body = expected
with tempfile.TemporaryDirectory(dir=HERE) as name, MonkeyPatch.context() as mp:
    mp.setattr(socket.socket, "connect", archived.offline)
    mp.setattr(socket, "create_connection", archived.offline)
    mp.setattr(socket, "getaddrinfo", archived.offline)
    ops.warnings_and_refusals(Path(name), mp)
