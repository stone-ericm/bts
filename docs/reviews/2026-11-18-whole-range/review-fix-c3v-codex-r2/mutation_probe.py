"""Prepare scratch revisions and verify which source pytest imports."""

from contextlib import ExitStack
import importlib.util
import io
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
from unittest.mock import patch


REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[1]
SCRATCH = REVIEW / "mutation-archives"
DATE_RENDER = 'replay["blocked_date"].encode("unicode_escape").decode("ascii")'
REASON_RENDER = 'replay["reason"].encode("unicode_escape").decode("ascii")'
PENDING = """        if r not in ("hit", "miss"):
            return None, (f.stem, "no final result on the pick")"""
TODAY_APPLIED = """                return None, (f.stem, "today's result is applied to streak.json but not yet saved on the pick")"""
READ_FAILED = """        except Exception:
            return None, (f.stem, "pick file unreadable or not valid JSON")"""


def replace(body, before, after):
    assert body.count(before) == 1, before
    return body.replace(before, after)


def build_cases():
    yield "no-escaping", "cli", [
        (DATE_RENDER, 'replay["blocked_date"]'),
        (REASON_RENDER, 'replay["reason"]'),
    ]
    # Change only the on-disk receipt. This mutant leaves all stdout/stderr
    # assertions intact, so a kill proves that the raw receipt is checked.
    yield "escape-receipt", "reconcile_receipt", [
        ("            rec = self.record()\n", """            rec = self.record()
            if "blocked_date" in rec["replay"]:
                rec["replay"] = dict(rec["replay"])
                for key in ("blocked_date", "reason"):
                    rec["replay"][key] = rec["replay"][key].encode("unicode_escape").decode("ascii")
"""),
    ]
    yield "repr", "cli", [
        (DATE_RENDER, 'repr(replay["blocked_date"])'),
        (REASON_RENDER, 'repr(replay["reason"])'),
    ]
    start = """        if not _ISO_DATE_RE.match(f.stem) or f.stem > today_iso:
            continue  # skip non-date files and future previews
        try:"""
    yield "M5b", "picks", [
        (start, start[:-len("        try:")] + """        if f.stem == today_iso and today_iso in load_applied_dates(picks_dir):
""" + TODAY_APPLIED[4:] + "\n        try:"),
        ("""            if today_iso in load_applied_dates(picks_dir):
                # A scorer applied today's result to streak.json and has not saved the pick's terminal
                # result yet (it died between the two writes, or is between them now). Replaying without
                # today would save the streak WITHOUT that result while keeping the marker that stops
                # the scorer's retry from re-applying it: the result would be lost (review pp1 B1).
                # Fail closed: keep the live-tracked streak until the scorer finishes.
""" + TODAY_APPLIED + "\n", ""),
    ]
    yield "M5c", "picks", [
        (READ_FAILED, "        except Exception:\n"
         "            if f.stem == today_iso and today_iso in load_applied_dates(picks_dir):\n"
         + TODAY_APPLIED + "\n" + READ_FAILED.split("\n", 1)[1]),
    ]
    yield "M5d", "picks", [
        (PENDING, """        try:
            probe_daily = load_pick(f.stem, picks_dir)
        except Exception:
            probe_daily = "ERR"
        if probe_daily is None:
            return None, (f.stem, "hit pick could not be loaded")
""" + PENDING),
    ]
    yield "M6", "cli", [
        ('        label = "Streak retained" if blocked else "Streak recalculated"',
         '        label = "Streak retained" if replay["state"] != "saved" else "Streak recalculated"'),
    ]


def prepare():
    assert subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"],
                                   cwd=REPO, text=True, timeout=10).strip() == "6ff1e87e"
    for name, revision in (("base", "88b69a16"), ("head", "6ff1e87e")):
        root = SCRATCH / name
        root.mkdir(parents=True, exist_ok=True)
        archive = subprocess.check_output(["git", "archive", revision, "src"], cwd=REPO, timeout=10)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(root, filter="data")
    for name, filename, edits in build_cases():
        shutil.copytree(SCRATCH / "head", SCRATCH / name, dirs_exist_ok=True)
        target = SCRATCH / name / "src" / "bts" / f"{filename}.py"
        body = target.read_text()
        for before, after in edits:
            body = replace(body, before, after)
        compile(body, str(target), "exec")
        target.write_text(body)
        print(f"{name} {target.relative_to(REVIEW)}")


def pytest_configure(config):
    # Permit only the loopback network used by the existing web-saver test.
    allowed = {"localhost", "127.0.0.1", "::1"}
    base_connect = socket.socket.connect
    base_connect_ex = socket.socket.connect_ex
    base_getaddrinfo = socket.getaddrinfo

    def guard_connect(original):
        def guarded(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in allowed:
                raise AssertionError(f"unexpected external connection to {address[0]!r}")
            return original(sock, address)
        return guarded

    def local_getaddrinfo(host, *args, **kwargs):
        if host not in allowed:
            raise AssertionError(f"unexpected external name resolution for {host!r}")
        return base_getaddrinfo(host, *args, **kwargs)

    with ExitStack() as context:
        context.enter_context(patch.object(socket.socket, "connect", guard_connect(base_connect)))
        context.enter_context(patch.object(socket.socket, "connect_ex", guard_connect(base_connect_ex)))
        context.enter_context(patch.object(socket, "getaddrinfo", local_getaddrinfo))
        config.add_cleanup(context.pop_all().close)


def pytest_sessionstart():
    expected = Path(os.environ["WRC3V_REVIEW_SOURCE"]).resolve() / "bts"
    for name in ("picks", "cli", "reconcile_receipt"):
        spec = importlib.util.find_spec(f"bts.{name}")
        assert spec and Path(spec.origin).resolve().parent == expected, (name, spec, expected)
    print(f"Verified bts source: {expected}", flush=True)


if __name__ == "__main__":
    assert sys.argv[1:] == ["prepare"]
    prepare()
