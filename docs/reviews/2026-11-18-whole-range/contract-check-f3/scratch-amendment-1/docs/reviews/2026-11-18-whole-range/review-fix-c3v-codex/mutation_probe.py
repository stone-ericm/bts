"""Run all four new test files against each single production mutation."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from probe_common import REVIEW, REPO


FILES = ["tests/test_wr_c3v_corpus.py", "tests/test_wr_c3v_delivery.py",
         "tests/test_wr_c3v_reconcile.py", "tests/test_wr_c3_r2_entered_pick_blockers.py"]
WRAPPER = Path("/Users/eric/bts-runs/yield_run.sh")


def replace_once(body, before, after):
    assert body.count(before) == 1, before
    return body.replace(before, after)


def drop_warning(body):
    return replace_once(body, '''    if blocked:
        click.echo(
            f"Season replay unavailable for {replay['blocked_date']} ({replay['reason']}); "
            "streak retained, not recalculated.",
            err=True,
        )
''', "")


def last_blocker(body):
    start = body.index("def _replay_season_streak_detail(")
    end = body.index("\n\ndef is_resume_date_game(", start)
    fold = body[start:end]
    fold = replace_once(fold, "    streak, saver = 0, True\n", "    streak, saver = 0, True\n    blocker = None\n")
    fold, n = re.subn(r'(?m)^( +)return None, (\(f\.stem, .+\))$', r'\1blocker = \2\n\1continue', fold)
    assert n == 4
    fold = replace_once(fold, "    return (streak, saver), None", "    if blocker is not None:\n        return None, blocker\n    return (streak, saver), None")
    return body[:start] + fold + body[end:]


def recalculated(body):
    return replace_once(body, 'label = "Streak retained" if blocked else "Streak recalculated"',
                        'label = "Streak recalculated"')


def skip_past(body):
    return replace_once(body, 'return None, (f.stem, "no final result on the pick")', 'continue')


def order(body):
    today = body.index('        if f.stem == today_iso and r not in ("hit", "miss", "void"):')
    past = body.index('        if r not in ("hit", "miss"):', today)
    hit = body.index('        if r == "hit":', past)
    new_past = body[past:hit].replace('("hit", "miss")', '("hit", "miss", "void")')
    return body[:today] + new_past + body[today:past] + body[hit:]


def await_idle():
    deadline = time.monotonic() + 600
    while subprocess.run(["pgrep", "-f", "yield_run.sh"], stdout=subprocess.DEVNULL,
                         timeout=5, check=False).returncode == 0:
        if time.monotonic() >= deadline:
            raise TimeoutError("shared wrapper never went idle")
        time.sleep(3)


if __name__ == "__main__":
    for mutation, module in [(drop_warning, "cli"), (last_blocker, "picks"),
                             (recalculated, "cli"), (skip_past, "picks"), (order, "picks")]:
        root = REVIEW / ("mutant-" + mutation.__name__)
        source = root / "src"
        shutil.copytree(REVIEW / "current" / "src", source)
        path = source / "bts" / (module + ".py")
        path.write_text(mutation(path.read_text()))
        compile(path.read_text(), str(path), "exec")
        log = REVIEW / ("mutant-" + mutation.__name__ + ".log")
        await_idle()
        command = [str(WRAPPER), str(log), "env", "TZ=America/New_York", "PYTHONDONTWRITEBYTECODE=1",
                   f"PYTHONPATH={source}", sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                   *FILES, "-q", "--tb=short"]
        env = os.environ | {"BTS_YIELD_SECONDS": "4"}
        run = subprocess.run(command, cwd=REPO, env=env, timeout=900, check=False)
        assert run.returncode in (0, 1), (mutation.__name__, run.returncode)
        match = re.search(r"(?:(\d+) failed, )?(\d+) passed in", log.read_text())
        assert match and sum(int(count or 0) for count in match.groups()) == 659, mutation.__name__
        print(mutation.__name__, "failures, passes =", match.groups(), flush=True)
