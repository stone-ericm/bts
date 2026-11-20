"""Print sample console and wire copy for the handover from the real check."""

from contextlib import redirect_stderr
from datetime import timedelta
import io
import logging
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from bts.health import contest_state, runner


ROOT = Path(__file__).resolve().parents[2]
history = runpy.run_path(str(ROOT / "tests/test_wr_c1_history.py"))
probe = SimpleNamespace(**runpy.run_path(str(
    ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-c1/probe.py"
)))
logging.disable(logging.CRITICAL)


def block(text):
    print("\n````text\n" + text + "\n````\n")


with TemporaryDirectory(prefix="wr-c1-copy-") as temp:
    root = Path(temp)
    branch_cases = [
        (label, picks, day, label != "optional_bypass")
        for label, picks, day, reasons in probe.cases(root / "driver")
    ]
    branch_cases.extend(
        (picks.parent.name, picks, probe.D, expected)
        for picks, reasons, expected in history["more_cases"](probe, root / "extra")
    )
    seen = set()
    samples = {}
    with redirect_stderr(io.StringIO()):
        for label, picks, day, expected in branch_cases:
            live = runner._safe_run(
                "contest_state",
                lambda: contest_state.check(picks, expected=expected, now=probe.end(day)),
            )
            historical, bodies = history["run"](
                picks, day, probe.at(day + timedelta(days=1), 2), expected=expected, isolate=True,
            )
            for original, replacement in zip(live, historical, strict=True):
                identity = (replacement.level, replacement.source, replacement.message)
                if identity in seen:
                    continue
                seen.add(identity)
                samples.setdefault(label, []).append((original, replacement))

    print("## Alert copy by branch\n")
    for label, rows in samples.items():
        print(f"### {label}\n")
        for original, replacement in rows:
            print(f"{replacement.level} · {replacement.source}. Same-day sample at the slate endpoint.")
            block(original.message)
            print("Historical message.")
            block(replacement.message)

    print("## Full captured DM samples\n")
    for mixed in (False, True):
        picks = root / f"wire_{mixed}" / "picks"
        probe.auto(picks)
        probe.switch(picks, status="revoked" if mixed else "automatic")
        probe.saver_flag(picks, state="not_earned")
        _, bodies = history["run"](picks, probe.D, probe.at(probe.D + timedelta(days=1), 2), isolate=True)
        body, = bodies
        block(body)
