"""Immutable pytest family: repeated, ordered, isolated, old-source, and mutation runs."""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REAL_DT, REAL_DATE = dt.datetime, dt.date
ADOPT = "tests/test_season_contract_cs4b1_review.py"
AUTHOR = "tests/test_season_contract_cs4b1.py"
EDGE = "tests/test_season_contract_cs4b1_edges.py"
NODES = [
    "test_b1_status_ledger_line_repr_recursion_error_never_emits_double_ledger_header",
    "test_b2_status_saver_boolean_season_classified_invalid_not_another_season[True]",
    "test_b2_status_saver_boolean_season_classified_invalid_not_another_season[False]",
    "test_m1_wrong_enable_season_precedes_lstat_and_open_with_seeded_switch",
    "test_m2_genuine_12000_deep_json_triggers_recursion_error_in_switch_and_status",
    "test_m6_v2_season_error_precedes_activation_id_without_season_in_test_name",
    "test_between_profile_and_rounds_dec31_clock_advance_preserves_entry_year",
]


def audit(event, args):
    if event.startswith("socket.") and event != "socket.__new__":
        raise AssertionError("network forbidden: " + event)


sys.addaudithook(audit)


def child(label, source, year, limit, selections):
    sys.path.insert(0, str(source))
    import pandas
    import pytest
    fixed = REAL_DT(year, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))

    class Clock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

        @classmethod
        def today(cls):
            return cls.now()

        @classmethod
        def utcnow(cls):
            return fixed.astimezone(dt.timezone.utc).replace(tzinfo=None)

    class Day(REAL_DATE):
        @classmethod
        def today(cls):
            return cls(year, 7, 15)

    dt.datetime, dt.date = Clock, Day
    if limit:
        sys.setrecursionlimit(limit)
    import bts.cli as cli
    assert Path(cli.__file__).is_relative_to(source), cli.__file__
    records = {}

    class Recorder:
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed:
                records[report.nodeid] = dict(outcome=report.outcome, when=report.when,
                                             failure=str(report.longrepr) if report.failed else None)

    rc = pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", "--basetemp", str(HERE / ("pytest-temp-" + label)), *selections], plugins=[Recorder()])
    result = dict(rc=int(rc), counts=dict(collections.Counter(r["outcome"] for r in records.values())),
                  records=records, clock_year=year, recursion_limit=sys.getrecursionlimit(), python=sys.version)
    (HERE / ("test-" + label + ".json")).write_text(json.dumps(result, indent=2))


def run(label, source, year, selections, limit=0):
    with (HERE / ("test-" + label + ".log")).open("w") as out:
        subprocess.run([sys.executable, __file__, "child", label, str(source), str(year), str(limit), *selections],
                       cwd=REPO, stdout=out, stderr=subprocess.STDOUT, check=True,
                       env=os.environ | {"TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"})
    result = json.loads((HERE / ("test-" + label + ".json")).read_text())
    print(label, result["counts"], "rc", result["rc"], flush=True)
    return result


def mutant(label, filename, old, new):
    target = HERE / "mutants" / label / "src"
    shutil.copytree(REPO / "src", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / filename
    text = path.read_text()
    assert text.count(old) == 1, (label, text.count(old))
    path.write_text(text.replace(old, new))
    return target


def main():
    if len(sys.argv) > 1:
        child(sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6:])
        return
    summary = {}
    for year in [2027, 2028]:
        for i in range(20):
            label = f"alone-{year}-{i:02}"
            summary[label] = run(label, REPO / "src", year, [ADOPT])["counts"]
        label = f"after-author-edge-{year}"
        summary[label] = run(label, REPO / "src", year, [AUTHOR, EDGE, ADOPT])["counts"]
    for i, node in enumerate(NODES):
        label = "isolated-" + str(i)
        summary[label] = run(label, REPO / "src", 2028, [ADOPT + "::" + node])["counts"]
    old = run("old-seven", HERE / "old/src", 2027, [ADOPT])
    summary["old-seven"] = old["counts"]
    for limit in [500, 2000, 20000]:
        label = "recursion-" + str(limit)
        summary[label] = run(label, REPO / "src", 2027, [ADOPT], limit)["counts"]
    # Each adopted behavioural assertion must reject a concrete production mutation.
    old_ledger = (HERE / "old/src/bts/season_contract.py").read_text().split('    path = directory / "contest_ledger.jsonl"')[1]
    head_ledger = (REPO / "src/bts/season_contract.py").read_text().split('    path = directory / "contest_ledger.jsonl"')[1]
    mutations = {
        "B1": ("bts/season_contract.py", head_ledger, old_ledger, [NODES[0]]),
        "B2": ("bts/season_contract.py", 'and type(raw.get("season")) is int)',
               'and isinstance(raw.get("season"), int))', NODES[1:3]),
        "M1": ("bts/cli.py",
               '    if season != year:\n        raise click.UsageError(f"--season must equal the current ET year {year}")\n    picks = Path(picks_dir)\n    state = switch_state(picks, now)',
               '    picks = Path(picks_dir)\n    state = switch_state(picks, now)\n    if season != year:\n        raise click.UsageError(f"--season must equal the current ET year {year}")', [NODES[3]]),
        "M2": ("bts/contest_state.py", '_UNREADABLE = (OSError, ValueError, RecursionError)',
               '_UNREADABLE = (OSError, ValueError)', [NODES[4]]),
        "M6": ("bts/contest_state.py",
               '        if type(season) is not int:\n            raise ContestStateError(f"automatic contest season invalid in {path}: must be an integer")\n        activation_id = data.get("activation_id")\n        if not valid_activation_id(activation_id):\n            raise ContestStateError(f"automatic contest activation_id invalid in {path}: must be 32 lowercase hexadecimal characters")',
               '        activation_id = data.get("activation_id")\n        if not valid_activation_id(activation_id):\n            raise ContestStateError(f"automatic contest activation_id invalid in {path}: must be 32 lowercase hexadecimal characters")\n        if type(season) is not int:\n            raise ContestStateError(f"automatic contest season invalid in {path}: must be an integer")', [NODES[5]]),
        "clock": ("bts/cli.py", '            now,\n        )\n    except httpx.HTTPError',
                  '            datetime.now(timezone.utc),\n        )\n    except httpx.HTTPError', [NODES[6]]),
    }
    for label, (filename, before, after, nodes) in mutations.items():
        source = mutant(label, filename, before, after)
        result = run("mutant-" + label, source, 2027, [ADOPT + "::" + n for n in nodes])
        summary["mutant-" + label] = result["counts"]
        assert result["counts"] == {"failed": len(nodes)}, label
    # Run only the two newly covered empty-list cells for persistence mutations.
    empty = [AUTHOR, "-k", "test_response_validation_and_alerts and bad5"]
    summary["empty-control"] = run("empty-control", REPO / "src", 2027, empty)["counts"]
    for label, before, after in [
        ("dry-persists", '    if dry_run:\n        if switched:', '    if False and dry_run:\n        if switched:'),
        ("unswitched-skips-success", '    if dry_run:\n        if switched:', '    if dry_run or not switched:\n        if switched:'),
        ("unswitched-writes-auto", '    if switched:\n        _atomic_write_json(out_path, observation)', '    if True:\n        _atomic_write_json(out_path, observation)'),
    ]:
        source = mutant(label, "bts/cli.py", before, after)
        result = run("mutant-" + label, source, 2027, empty)
        summary["mutant-" + label] = result["counts"]
        assert result["counts"].get("failed", 0) >= 1, label
    (HERE / "tests-summary.json").write_text(json.dumps(summary, indent=2))
    assert summary["old-seven"] == {"failed": 3, "passed": 4}
    for label, counts in summary.items():
        if label.startswith(("alone-", "after-", "isolated-")):
            assert not counts.get("failed"), (label, counts)
    assert summary["empty-control"] == {"passed": 2}, summary["empty-control"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
