"""Run revision-specific code in separate offline processes."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
BASE = REVIEW / "base-3d49308d"
OLD = REVIEW / "before-8cac32ea"
TREES = {"base": BASE, "before": OLD, "head": ROOT}
PRIOR = ROOT.parent / "confirm-1118/.codex-review/confirm-1118/first-review-probes.py"


def archive(revision, target, *paths):
    payload = subprocess.check_output(["git", "archive", revision, *paths], cwd=ROOT)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as source:
        source.extractall(target, filter="data")


def environment(source):
    return os.environ | {
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (REVIEW, source / "src", source, ROOT))),
    }


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(source, *args, env=None, cwd=None):
    return subprocess.run([sys.executable, __file__, *args],
                          cwd=cwd or source, env=env or environment(source)).returncode


def first_review():
    return load("first_review", PRIOR)


def main():
    action, *args = sys.argv[1:]
    if action == "archive":
        archive("3d49308d", BASE, "src", "scripts")
        archive("8cac32ea", OLD, "src", "scripts")
        return 0
    if action == "family":
        from revision_probes import COMPARES, SELECTED

        family, = args
        parent = Path(tempfile.mkdtemp(prefix=f"probe-{family}-", dir=REVIEW))
        print(f"{family} evidence: {parent.name}", flush=True)
        for revision in SELECTED[family]:
            env = environment(TREES[revision])
            if family == "p03":
                import datetime as dt
                now = dt.datetime.now(dt.timezone.utc)
                env.update(
                    DYLD_INSERT_LIBRARIES=str(REVIEW / "review_clock.dylib"),
                    BTS_CONFIRM_CLOCK_OFFSET=str(int((now.replace(year=2027) - now).total_seconds())),
                )
            code = run(TREES[revision], "child", family, revision, str(parent), env=env)
            if code:
                return code
        COMPARES[family](parent)
        print(f"{family}: revision comparison passed", flush=True)
        return 0
    if action == "child":
        from sitecustomize import check

        check()
        import bts
        from revision_probes import PROBES

        family, revision, parent = args
        tree = TREES[revision]
        assert Path(bts.__file__).is_relative_to(tree / "src"), bts.__file__
        import scripts.live_forward_capture_once as capture_script
        assert Path(capture_script.__file__).is_relative_to(tree / "scripts")
        out = Path(parent) / revision
        out.mkdir()
        PROBES[family](out, revision)
        print(f"{family} {revision}: assertions passed", flush=True)
        return 0
    raise ValueError(action)


if __name__ == "__main__":
    sys.modules["review_runner"] = sys.modules[__name__]
    raise SystemExit(main())
