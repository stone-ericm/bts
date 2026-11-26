"""Verify a scan-only scratch fix and measure the healthy scan cost."""

from datetime import date, timedelta
import json
from pathlib import Path
import sys
from time import perf_counter

from compare_probes import HEAD, archive, offline, python, scratch
from probe_support import decision, dump, pick_data


OLD = '''    dates = {path.stem for path in picks_dir.glob(f"{season}-*.json")}
    dates.update(path.parent.name for path in picks_dir.glob(f"{season}-*/decision.json"))'''
NEW = '''    dates = set()
    try:
        with os.scandir(picks_dir) as entries:
            for entry in entries:
                if entry.name.startswith(f"{season}-"):
                    day = entry.name.removesuffix(".json")
                    if _ISO_DATE_RE.fullmatch(day):
                        dates.add(day)
    except OSError as error:
        raise NonRegularReplayFile(f"cannot list {picks_dir}: {error}") from error'''

DECISION = '            text = _read_reconcile_file(decision_path(day, picks_dir))'
GUARDED = '''            try:
                day_mode = (picks_dir / day).lstat().st_mode
            except FileNotFoundError:
                text = None
            else:
                if not stat.S_ISDIR(day_mode):
                    raise NonRegularReplayFile(f"{picks_dir / day} is not a directory")
                text = _read_reconcile_file(decision_path(day, picks_dir))'''


def worker(root):
    import probe_blocker_glob_and_parent as old
    from bts import picks

    print(json.dumps(old._run_suite("scan-control")))
    count = 162
    root.mkdir()
    for index in range(count):
        day = (date(2027, 4, 1) + timedelta(days=index)).isoformat()
        dump(root / f"{day}.json", pick_data(day, "hit"))
        decision(root, day, scoreable=True)
    for index in range(25):
        if index == 5:
            start = perf_counter()
        assert picks._fold_season(root, 2027, "2027-12-31", []) == ((count, True), None)
    print(json.dumps({"millis_per_fold": (perf_counter() - start) * 1000 / 20, "dates": count}))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with offline():
            worker(Path(sys.argv[1]))
    else:
        with scratch("scan") as folder:
            root = Path(folder)
            src = archive(HEAD, root / "source", "src") / "src"
            path = src / "bts/picks.py"
            source = path.read_text()
            for label in ("observed", "fixed"):
                if label == "fixed":
                    assert source.count(OLD) == source.count(DECISION) == 1
                    path.write_text(source.replace(OLD, NEW).replace(DECISION, GUARDED))
                result = python(src, __file__, root / label, timeout=25)
                print(label, result.stdout, sep="\n", end="")
