"""Archive the two comparison trees and run offline probe families."""
import importlib.util
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
PYTHON = ROOT / ".venv/bin/python"
BASE = REVIEW / "base-3d49308d"
OLD = REVIEW / "before-153ff8c2"
TREES = {"base": BASE, "before": OLD, "head": ROOT}


def archive(revision, target, *paths):
    payload = subprocess.check_output(["git", "archive", revision, *paths], cwd=ROOT)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as source:
        source.extractall(target, filter="data")


def environment(source):
    return os.environ | {
        "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (REVIEW, source / "src", source, ROOT))),
    }


def run(source, *args, cwd=None):
    return subprocess.run([str(PYTHON), str(REVIEW / "review_runner.py"), *args],
                          cwd=cwd or source, env=environment(source)).returncode


def check_network_guard():
    assert "sitecustomize" in sys.modules
    with socket.socket() as connection:
        for call in (lambda: socket.getaddrinfo("dns-probe.invalid", 443),
                     lambda: connection.connect(("192.0.2.1", 443))):
            try:
                call()
            except AssertionError:
                pass
            else:
                raise AssertionError("Offline guard failed")
    assert socket.gethostbyaddr("127.0.0.1")[0] == "localhost"


def first_review():
    spec = importlib.util.spec_from_file_location("first_review", REVIEW / "first-review-probes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    action, *args = sys.argv[1:]
    if action == "archive":
        archive("3d49308d", BASE, "src", "scripts")
        archive("153ff8c2", OLD, "src", "scripts")
        return 0
    if action == "family":
        from confirmation_probes import COMPARES, SELECTED
        family, = args
        parent = REVIEW / f"probe-{family}"
        parent.mkdir(exist_ok=True)
        for tree in SELECTED[family]:
            code = run(TREES[tree], "child", family, tree)
            if code:
                return code
        COMPARES[family](parent)
        print(f"{family}: revision comparison passed", flush=True)
        return 0
    check_network_guard()
    if action == "child":
        import bts
        from confirmation_probes import PROBES
        family, tree = args
        source = TREES[tree]
        assert Path(bts.__file__).is_relative_to(source / "src"), bts.__file__
        output = REVIEW / f"probe-{family}" / tree
        output.mkdir(exist_ok=True)
        PROBES[family](output, tree)
        print(f"{family} {tree}: assertions passed", flush=True)
        return 0
    if action == "pytest":
        import pytest
        return pytest.main(["-o", "addopts=", "-p", "no:cacheprovider", "-q", "--tb=short", *args])
    raise ValueError(action)


if __name__ == "__main__":
    raise SystemExit(main())
