"""Compare deployed and candidate replay prefixes with read-only input."""

import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import signal
import sys

from compare_probes import HEAD, archive, offline, python, scratch
from probe_support import ET

DEPLOYED = "619abda7"


class ReplayTimeout(BaseException):
    pass


def expired(*_):
    raise ReplayTimeout("replay exceeded five seconds for this date")


def audit(event, args):
    if event == "open" and isinstance(args[2], int):
        import os

        if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise RuntimeError("dry replay attempted a file write")
    if event in ("os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.link", "os.symlink"):
        raise RuntimeError(f"dry replay attempted {event}")


def worker(root, today):
    from bts import picks

    sys.addaudithook(audit)
    signal.signal(signal.SIGALRM, expired)
    out = {}
    day = today.replace(month=1, day=1)
    while day <= today:
        key = day.isoformat()
        signal.alarm(5)
        try:
            value = picks._replay_season_streak(root, day.year, key)
            blocker = None
            if value is None and hasattr(picks, "_fold_season"):
                state = picks._read_reconcile_streak(root / "streak.json") or {}
                _, blocker = picks._fold_season(root, day.year, key, picks._replay_markers(state, day.year))
            out[key] = dict(value=value, blocker=blocker)
        except ReplayTimeout as error:
            out[key] = dict(error=type(error).__name__, detail=str(error))
            break
        except Exception as error:
            out[key] = dict(error=type(error).__name__, detail=str(error))
        finally:
            signal.alarm(0)
        day += timedelta(days=1)
    print(json.dumps(out))


def compare(root, today):
    by_revision = {}
    with scratch("dry") as folder:
        for rev in (DEPLOYED, HEAD):
            source = archive(rev, Path(folder) / rev, "src") / "src"
            result = python(source, __file__, root, "--today", today, "--worker", timeout=90)
            by_revision[rev] = json.loads(result.stdout)
    base, other = (by_revision[rev] for rev in (DEPLOYED, HEAD))
    first_divergence = None
    first_blocks = {rev: None for rev in by_revision}
    for day in sorted(set(base) | set(other)):
        values = []
        for rev, rows in by_revision.items():
            entry = rows.get(day)
            value = entry.get("value") if entry is not None else None
            values.append(value if value is not None else ("error" if entry and entry.get("error") else "blocked"))
            if value is None and first_blocks[rev] is None:
                first_blocks[rev] = {"as_of": day, **(entry or {})}
        if values[0] != values[1] and first_divergence is None:
            first_divergence = dict(as_of=day, **{DEPLOYED: base.get(day), HEAD: other.get(day)})
    key = today.isoformat()
    print(json.dumps(dict(as_of=key, first_difference=first_divergence, first_block=first_blocks,
                          final={DEPLOYED: base.get(key), HEAD: other.get(key)}), indent=2))
    return 2 if first_divergence else 3 if any(first_blocks.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("picks_dir", type=Path)
    parser.add_argument("--today", type=date.fromisoformat, default=datetime.now(ET).date())
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.picks_dir.resolve(strict=True)
    if not root.is_dir():
        parser.error("picks_dir must be a directory")
    if args.worker:
        with offline():
            worker(root, args.today)
        return 0
    return compare(root, args.today)


if __name__ == "__main__":
    raise SystemExit(main())
