"""Process isolation and pinned CLI inputs for the review probes."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo


REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[1]
ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, tzinfo=ET)
BASE = "88b69a16"


def refuse_network(*args, **kwargs):
    raise AssertionError("unexpected network access")


def require_source(revision):
    from bts import cli, picks, reconcile_receipt

    for module in (cli, picks, reconcile_receipt):
        source = REVIEW / revision / "src" / "bts"
        assert Path(module.__file__).resolve().parent == source, module.__file__


def invoke(root, clock, lookback=0):
    from bts import cli, reconcile_receipt
    from click.testing import CliRunner

    real_receipt = reconcile_receipt.ReconcileReceipt

    def pinned_receipt(**kwargs):
        rec = real_receipt(clock=clock, lookback_days=kwargs["lookback_days"])
        rec.run_id = "1" * 32
        return rec

    with patch.object(reconcile_receipt, "ReconcileReceipt", pinned_receipt):
        return CliRunner().invoke(cli.cli, ["reconcile", "--picks-dir", str(root),
                                            "--lookback", str(lookback)])


def streak_bytes(root):
    target = root / "streak.json"
    if target.is_symlink():
        return ("symlink " + os.readlink(target), target.read_bytes().hex() if target.is_file() else None)
    if target.is_dir():
        return "directory"
    return target.read_bytes().hex() if target.exists() else None


def signature(exc, root):
    if exc is None:
        return None
    return type(exc).__name__, [str(arg).replace(str(root.parent.parent), "<scratch>") for arg in exc.args]


def examine(root, clock=lambda: RUN, lookback=0, *, direct=True, fold_day=None):
    from bts import picks, reconcile_receipt

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return RUN.astimezone(tz)

    before = streak_bytes(root)
    fold = None
    with patch.object(picks, "datetime", FixedDatetime), patch.object(picks, "_git_head_sha", return_value=BASE):
        if direct:
            day = fold_day or clock().astimezone(ET).date()
            try:
                fold = ["return", picks._replay_season_streak(root, day.year, day.isoformat())]
            except BaseException as exc:
                fold = ["raises", signature(exc, root)]
        result = invoke(root, clock, lookback)
    receipts = list(reconcile_receipt.receipts_root(root).glob("*/*.json"))
    assert len(receipts) <= 1
    return {"strict": [fold, result.exit_code, signature(result.exception, root), before, streak_bytes(root)],
            "stdout": result.stdout, "stderr": result.stderr,
            "receipt": receipts[0].read_bytes().hex() if receipts else None}


def receipt(evidence):
    raw = evidence["receipt"]
    return json.loads(bytes.fromhex(raw)) if raw else {}


def validate(base, current):
    assert set(base) == set(current)
    counts = Counter()
    for name, old in base.items():
        new = current[name]
        assert old["strict"] == new["strict"], (name, old, new)
        old_receipt, new_receipt = receipt(old), receipt(new)
        state = old_receipt.get("replay", {}).get("state", "unpublished")
        counts[state] += 1
        if state == "unavailable" or "Season replay unavailable" in new["stderr"]:
            warnings = [line for line in new["stderr"].splitlines() if line.startswith("Season replay unavailable")]
            assert len(warnings) == 1 and warnings[0].endswith("; streak retained, not recalculated."), name
            if state == "unavailable":
                detail = new_receipt["replay"]
                assert warnings[0] == (f"Season replay unavailable for {detail['blocked_date']} ({detail['reason']}); "
                                       "streak retained, not recalculated."), name
                assert old_receipt == new_receipt | {"replay": {"state": "unavailable"}}, name
            assert new["stderr"].replace(warnings[0] + "\n", "") == old["stderr"], name
            expected = old["stdout"].replace("Streak: ", "Streak retained: ").replace(
                "Streak recalculated: ", "Streak retained: ")
            assert new["stdout"] == expected, name
        else:
            assert old == new, (name, old, new)
    return dict(counts)


def run_against_archives(script, probes, check=validate, timeout=45):
    if len(sys.argv) == 3:
        revision, raw_output = sys.argv[1:]
        with patch.object(socket.socket, "connect", refuse_network), \
             patch.object(socket.socket, "connect_ex", refuse_network), \
             patch.object(socket, "getaddrinfo", refuse_network), \
             tempfile.TemporaryDirectory(dir=REVIEW) as scratch, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            require_source(revision)
            results = probes(Path(scratch), revision)
            Path(raw_output).write_text(json.dumps(results, indent=2))
        return

    results = []
    for revision in ("base", "current"):
        output = REVIEW / f"{Path(script).stem}-{revision}.json"
        source = REVIEW / revision / "src"
        env = os.environ | {"PYTHONPATH": os.pathsep.join(map(str, (source, REPO))),
                            "PYTHONDONTWRITEBYTECODE": "1", "TZ": "America/New_York"}
        subprocess.run([sys.executable, script, revision, str(output)], env=env, cwd=REPO,
                       timeout=timeout, check=True)
        results.append(json.loads(output.read_text()))
    print(check(*results))
