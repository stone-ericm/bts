"""Apply one scratch change and run the checks that should observe it."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INSTALLER = Path("scripts/cron-setup-hetzner.sh")
LOCK = "/usr/bin/flock -n /home/bts/.check-pick-entered.lock"
TIMER = "/usr/bin/timeout --verbose -k 30 840"
INSTALLER_CASES = {
    "drop-n": (LOCK, LOCK.replace(" -n", "")),
    "drop-limit": (" " + TIMER, ""),
    "limit-900": (TIMER, TIMER.replace("840", "900")),
    "drop-verbose": (TIMER, TIMER.replace(" --verbose", "")),
    "swap-order": (f"{LOCK} {TIMER}", f"{TIMER} {LOCK}"),
}
TESTS = ["__init__.py", "test_entry_intent.py", "test_wr_f1_wrong_game.py", "test_wr_f1_review.py",
         "test_wr_f1b_lock.py"]


def main(kind):
    with tempfile.TemporaryDirectory(dir=HERE) as scratch:
        root = Path(scratch)
        shutil.copy(ROOT / "pyproject.toml", root)
        (root / "tests").mkdir()
        for name in TESTS:
            shutil.copy(ROOT / "tests" / name, root / "tests" / name)

        if kind in INSTALLER_CASES:
            src = ROOT / "src"
            path = root / INSTALLER
            path.parent.mkdir()
            body = (ROOT / INSTALLER).read_text()
            before, after = INSTALLER_CASES[kind]
            # Some wrapper tokens also occur in comments or the results line.
            lines = [line for line in body.splitlines(keepends=True) if line.startswith("*/15 10-23")]
            assert len(lines) == 1 and lines[0].count(before) == 1
            path.write_text(body.replace(lines[0], lines[0].replace(before, after)))
            tests = ["tests/test_wr_f1b_lock.py::test_generated_wrapper_order_and_interval"]
        else:
            assert kind in ("alerted", "dm_failed", "present_unverified")
            archive = subprocess.check_output(["git", "archive", "717ac97f", "src"], cwd=ROOT, timeout=3)
            subprocess.run(["tar", "xf", "-", "-C", scratch], input=archive, check=True, timeout=4)
            src = root / "src"
            path = src / "bts/cli.py"
            body = path.read_text()
            before = 'if marker["status"] != "confirmed" and (reason == "wrong_game" or prior_wrong_game):'
            after = (f'if marker["status"] not in ("confirmed", "{kind}") '
                     'and (reason == "wrong_game" or prior_wrong_game):')
            assert body.count(before) == 1
            path.write_text(body.replace(before, after))
            tests = ["tests/test_wr_f1_review.py::test_failed_observation"]

        print(f"scratch change {kind}: {path}", flush=True)
        result = subprocess.run(
            [sys.executable, "-c", "import bts, pytest, sys; print('bts.__file__ =', bts.__file__, flush=True); "
             "raise SystemExit(pytest.main(sys.argv[1:]))", "-p", "no:cacheprovider", "-o", "addopts=", "-q",
             "--tb=short", *tests], cwd=root, env={**os.environ, "PYTHONPATH": str(src)}, timeout=15)
        print(f"pytest status: {result.returncode}", flush=True)
        assert result.returncode in (0, 1), result.returncode
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
