"""Apply each requested scratch change by itself and run the F3 tests."""
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
TEST = "tests/test_wr_f3_skip_state_review.py"


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


MUTATIONS = {
    "read_on_all_outcomes": [
        ("    outcome = classify_day(",
         "    _read_evidence(Path(picks_dir) / 'account_state' / 'contest_streak.manual.json')\n"
         "    outcome = classify_day("),
    ],
    "drop_archives": [("def _archives(account: Path):\n", "def _archives(account: Path):\n    return\n")],
    "use_requests": [
        ("def _read_evidence(path: Path) -> object:",
         "def _read_evidence(path: Path, request_date=None) -> object:"),
        ('                    return json.loads(raw.decode("utf-8"))',
         '''                    decoded = raw.decode("utf-8")
                    if request_date is None:
                        return json.loads(decoded)
                    for line in reversed(decoded.splitlines()):
                        try:
                            row = json.loads(line)
                        except (ValueError, RecursionError):
                            continue
                        if isinstance(row, dict) and row.get("effective_date") == request_date:
                            return row
                    return None'''),
        ("    data = _read_evidence(path)",
         "    request_date = day.isoformat() if path.name == 'contest_manual_requests.jsonl' else None\n"
         "    data = _read_evidence(path, request_date)"),
        ('    current = _dated_evidence(account / "contest_streak.manual.json", day)',
         '    current = _dated_evidence(account / "contest_manual_requests.jsonl", day)'),
    ],
    "drop_revoke_suppression": [("    if revoked is not None and not matching_dated:", "    if revoked is not None:")],
    "compare_streak_only": [
        ("current[1] == skip_inputs", "current[1][0] == streak"),
        ("matching_dated |= inputs == skip_inputs", "matching_dated |= inputs[0] == streak"),
        ("dated_inputs != skip_inputs", "dated_inputs[0] != streak"),
    ],
    "drop_byte_bound": [
        ("            while len(raw) <= EVIDENCE_MAX_BYTES:\n"
         "                chunk = os.read(fd, EVIDENCE_MAX_BYTES + 1 - len(raw))",
         "            while True:\n                chunk = os.read(fd, 4096)"),
    ],
    "drop_overflow_handling": [("except (InvalidContract, OverflowError):", "except InvalidContract:")],
    "drop_enter_condition": [("_revocation_time(account, day) if intent == \"enter\" else None",
                              "_revocation_time(account, day)")],
    "keep_deliberate_skip_text": [
        ('return DayOutcome(date=date, state="skip_state_review", detail=detail)',
         'return DayOutcome(date=date, state="skip_state_review", detail="The policy skipped the day.")'),
    ],
}


def main(scratch):
    source = (REPO / "src/bts/day_outcome.py").read_text()
    scratch = scratch.resolve()
    assert not scratch.is_relative_to(REPO)
    target = scratch / "src/bts/day_outcome.py"
    assert target.read_text() == source
    environment = os.environ | {"TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1",
                                "PYTHONPATH": f"{scratch / 'src'}:{REPO}"}
    entry = ("import bts, pytest, sys; print('bts.__file__=' + bts.__file__, flush=True); "
             "raise SystemExit(pytest.main(sys.argv[1:]))")
    for name, edits in {"control": (), **MUTATIONS}.items():
        changed = source
        for before, after in edits:
            changed = replace_once(changed, before, after)
        try:
            target.write_text(changed)
            result = subprocess.run([sys.executable, "-c", entry, "-p", "no:cacheprovider", "-q", TEST],
                                    cwd=REPO, env=environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=12, check=False, text=True)
        finally:
            target.write_text(source)
        (Path(__file__).parent / f"mutation-{name}.log").write_text(result.stdout)
        assert result.stdout.startswith(f"bts.__file__={scratch}/src/bts/__init__.py\n"), result.stdout[:500]
        assert result.returncode in (0, 1), (name, result.returncode, result.stdout[-1000:])
        assert "collected 83 items" in result.stdout, result.stdout[:1000]
        assert name != "control" or result.returncode == 0, result.stdout[-1000:]
        summary = [line for line in result.stdout.splitlines() if " in " in line and "===" in line][-1]
        print(f"{name}: exit {result.returncode} {summary}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
