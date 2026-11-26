"""Extract composed E2/E3 text, apply Amendment 2 and exercise HEAD."""
import re
import socket
import tempfile
from pathlib import Path

from fixtures import HERE, REVIEWS, fixture
from pytest import MonkeyPatch

ops = fixture("probe_operators")
ops.CONTRACT = REVIEWS / "kickoff-fix-e.md"
archived = fixture("probe_text", "contract-check-e")


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
    mp.setattr(socket.socket, "create_connection", archived.offline)
    mp.setattr(socket.socket, "getaddrinfo", archived.offline)
    ops.warnings_and_refusals(Path(name), mp)
