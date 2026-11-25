"""Run the focused suite, fresh season walk and each scratch-source change."""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "scripts/audit/season_walk/results"
SUITE = HERE / "run_suite.py"
SOURCE = ROOT / "src/bts"
ENV = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1",
           MPLCONFIGDIR=str(HERE / "matplotlib"), PYTHONPATH=str(HERE / "offline"))


def run(args, log, env=ENV, limit=90):
    with (HERE / log).open("w") as out:
        command = [sys.executable, *map(str, args)]
        print("Executing", command, flush=True)
        return subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                              timeout=limit).returncode


def compare_walk():
    from scripts.audit.season_walk import walk

    with gzip.open(RESULTS / "walk-bc143131.json.gz", "rt") as f:
        baseline = json.load(f)
    with gzip.open(RESULTS / "walk-e.json.gz", "rt") as f:
        committed = json.load(f)
    updated = json.loads((HERE / "walk.json").read_text())
    assert baseline["runs"] == updated["runs"] == 4599
    assert baseline["by_day"] == updated["by_day"] and baseline["boundary"] == updated["boundary"]
    assert updated == committed
    assert not updated["problems"] and not walk.check(updated["records"])
    fields = (
        "day", "state", "action", "pick_bar", "expected_days_left", "days_left", "days_effective",
        "batter", "double", "saver_available", "chosen_on_kind", "objective", "source", "reason", "no_pick_reason",
    )
    flags = {line.split(": skip message", 1)[0] for line in baseline["problems"] if ": skip message" in line}
    cases = Counter()
    for old, new in zip(baseline["records"], updated["records"], strict=True):
        assert tuple(old[k] for k in fields) == tuple(new[k] for k in fields)
        if f"{old['day']} {old['state']}" not in flags:
            continue
        text = new["skip_dm"]
        if "at or above the " in text:
            cases["K3"] += 1
        elif "No probability comparison is available for this check." in text:
            cases["K4"] += 1
        else:
            raise AssertionError(new)
    assert cases == {"K3": 229, "K4": 612}, cases
    # Plant the wrong text in a real positive-bar skip record.
    point = next(row for row in updated["records"] if "at or above" in (row["skip_dm"] or ""))
    poisoned = point | {"skip_dm": point["skip_dm"].replace("at or above", "below", 1)}
    assert not walk.check([point])
    failed = walk.check([poisoned])
    assert any("skip message" in problem for problem in failed), failed
    print("Walk equality, 4,599 fresh records, 229 K3 and 612 K4; planted failure:", failed, flush=True)


def variations():
    def edits(file, old, new, count=1):
        return file, old, new, count

    return {
        "top-prediction": edits("scheduler", """            summary.update(
                best_batter=name if isinstance(name, str) else "?",
                best_team=team if isinstance(team, str) else "?",
                best_p=probability,
                candidate_verified=True,
            )""", '            summary["candidate_verified"] = True'),
        "below-on-equality": edits("scheduler", "if probability < bar:", "if probability <= bar:"),
        "drop-unverified-label": edits("scheduler", "Top model prediction (eligibility unverified)", "Top model prediction"),
        "swap-K1-K5": edits("scheduler", "if isinstance(reason, str) and reason:",
                            'if isinstance(reason, str) and reason and fields.get("basis") != "no_decision":'),
        "old-K3-journal-sentence": edits(
            "scheduler", '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
            """                message = format_skip_summary(skip_summary)
                if ", at or above the " in message:
                    message += " Will pick if a confirmed lineup clears it."
                print(f"  SKIP — {message}", file=sys.stderr)""",
        ),
        "unescaped-banner-name": edits("web", "    body = html_lib.escape(format_skip_summary(summary))",
                                       """    body = html_lib.escape(format_skip_summary(summary))
    name = summary.get("best_batter")
    if isinstance(name, str):
        name = " ".join(name.splitlines())
        body = body.replace(html_lib.escape(name), name, 1)"""),
        "drop-delivery-saver": edits(
            "scheduler", '''"saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts "
            "saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."''',
            '"saver used, check bts saver-state --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."',
        ),
        "old-late-refusal": edits("scheduler", '"The entry check may report a mismatch if it verifies the entry in time.")',
                                 '"Enter a pick manually if you want one now.")', 2),
        "string-render-raises": edits("scheduler", "    if type(value) not in (int, float):",
                                     '    if isinstance(value, str):\n        raise TypeError("string probability")\n'
                                     '    if type(value) not in (int, float):'),
    }


def scratch_runs():
    with tempfile.TemporaryDirectory(prefix="mutation-", dir=HERE) as name:
        src = Path(name) / "src"
        shutil.copytree(SOURCE, src / "bts", ignore=shutil.ignore_patterns("__pycache__"))
        for label, (file, old, new, count) in variations().items():
            path = src / "bts" / (file + ".py")
            original = path.read_text()
            assert original.count(old) == count, (label, original.count(old))
            path.write_text(original.replace(old, new, 1))
            # Avoid stale bytecode during scratch edits even if two source sizes match.
            result = run([SUITE], f"mutation-{label}.log", ENV | {
                "WR_E_MUTATION_SRC": str(src), "PYTHONPATH": os.pathsep.join((str(src), ENV["PYTHONPATH"])),
            })
            path.write_text(original)
            assert result in (0, 1), (label, result)
            print(label, "survived" if result == 0 else "detected", flush=True)


def main():
    sys.path.insert(0, str(ROOT))
    requested = sys.argv[1:] or ["suite", "walk", "mutations"]
    failures = []
    if "suite" in requested:
        result = run([SUITE], "combined.log")
        print("Combined test exit code:", result, flush=True)
        if result:
            failures.append("suite")
    if "walk" in requested:
        result = run(["-m", "scripts.audit.season_walk.walk", "--out", HERE / "walk.json"], "walk.log")
        if result:
            failures.append("walk")
        else:
            compare_walk()
    if "mutations" in requested:
        scratch_runs()
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
