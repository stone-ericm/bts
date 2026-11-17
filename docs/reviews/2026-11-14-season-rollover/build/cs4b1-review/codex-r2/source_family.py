"""Same-fault audit of newly added/touched source, without repeating round-one matrices."""
from __future__ import annotations

import ast
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.dont_write_bytecode = True
ACTIVE = None


def audit(event, args):
    if event.startswith("socket.") and event != "socket.__new__":
        raise AssertionError("network forbidden: " + event)
    if ACTIVE is not None:
        if event == "open":
            path, mode, flags = args
            if isinstance(path, (str, bytes)) and str(ACTIVE) in str(path):
                assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC), args
        if event in {"os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.chmod", "os.symlink"}:
            raise AssertionError("status mutation: " + event)


sys.addaudithook(audit)
sys.path.insert(0, str(REPO / "src"))


def main():
    global ACTIVE
    # Attribute unchanged integer checks separately from new code using actual diff lines.
    diff = subprocess.run(["git", "diff", "--unified=0", "8421e3cd", "HEAD", "--", "src"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    changed = {}
    filename, line = None, None
    for text in diff.splitlines():
        if text.startswith("+++ b/"):
            filename = text[6:]
            changed[filename] = set()
        elif text.startswith("@@"):
            line = int(re.search(r"\+(\d+)", text)[1])
        elif line is not None and text.startswith("+"):
            changed[filename].add(line)
            line += 1
        elif line is not None and not text.startswith(("-", "\\")):
            line += 1
    integer_checks = []
    for name, added in changed.items():
        source = (REPO / name).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance" and len(node.args) == 2:
                types = ast.unparse(node.args[1])
                if re.search(r"\bint\b", types):
                    integer_checks.append(dict(file=name, line=node.lineno, expression=ast.unparse(node), added=node.lineno in added))
    assert not any(x["added"] for x in integer_checks), integer_checks
    import pandas
    import bts.cli as cli
    import bts.season_contract as sc
    import bts.contest_state as cs
    import bts.contest_fetch as cf
    from click.testing import CliRunner
    real = dt.datetime
    now = real(2027, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))

    class Clock(real):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    dt.datetime = Clock
    root = HERE / "source-work"
    acc = root / "picks/account_state"
    acc.mkdir(parents=True)
    runner = CliRunner()
    records = []
    contract = dict(schema_version=sc.SCHEMA, season=2027, status="automatic", activation_id="b" * 32,
                    switched_at="2027-07-15T16:00:00Z", revoked_at=None, note=None)
    auto = dict(schema_version=sc.AUTO_SCHEMA, active_streak=7, best_streak=9, season=2027,
                activation_id="b" * 32, source="mlb_bts_profile", source_date="2027-07-14")

    def status(label):
        global ACTIVE
        ACTIVE = root
        try:
            r = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(root / "picks")])
        finally:
            ACTIVE = None
        record = dict(label=label, exit=r.exit_code, output=r.output.replace(str(root), "<case>"))
        records.append(record)
        for prefix in ["switch:", "manual:", "automatic:", "saver:", "ledger:"]:
            assert len([ln for ln in r.output.splitlines() if ln.startswith(prefix)]) == 1, record
        return record

    values = [None, False, True, 0, 2027, 2028, 2027.0, "2027", [], [2027], {}, {"year": 2027}]
    for value in values:
        (acc / "season_contract.json").write_text(json.dumps(contract | {"season": value}))
        st = sc.switch_state(root / "picks", now)
        r = status(["switch-season", value])
        if type(value) is int:
            assert st.state == ("automatic" if value == 2027 else "other_season"), r
            assert r["exit"] == 0, r
        else:
            assert st.state == "corrupt" and st.check == 3 and r["exit"] == 1, r
        (acc / "season_contract.json").unlink()
        (acc / "contest_streak.json").write_text(json.dumps(auto | {"season": value}))
        r = status(["auto-season", value])
        assert ("automatic: refused" in r["output"]) is (type(value) is not int), r
        (acc / "contest_streak.json").unlink()
    for field in ["activeStreak", "seasonBestStreak"]:
        for value in [False, True]:
            try:
                cf.validate_fetch(dict(activeStreak=7, seasonBestStreak=9) | {field: value})
            except cf.ContestFetchError as e:
                assert "integer" in str(e)
            else:
                raise AssertionError((field, value))
    # Find nested values which decode successfully but cannot be represented at this stack.
    bounds = json.loads((HERE / "head-boundaries.json").read_text())
    depths = sorted({1, 12000, 50000, *(n for b in bounds.values() for n in range(b - 15, b + 5))})
    for filename, fields, base in [
        ("contest_streak.manual.json", ["active_streak", "source", "schema_version"], dict(active_streak=7)),
        ("contest_streak.json", ["active_streak", "source", "schema_version"], auto),
        ("saver_state.json", ["season", "state", "source"], dict(state="active", season=2027)),
        ("season_contract.json", ["season", "status", "note", "schema_version"], contract),
        ("contest_ledger.jsonl", ["predictions", "extra"], dict(recorded_at="retained")),
    ]:
        for field in fields:
            for depth in depths:
                parts = [json.dumps(k) + ":" + json.dumps(v) for k, v in base.items() if k != field]
                parts.append(json.dumps(field) + ":" + "[" * depth + "0" + "]" * depth)
                path = acc / filename
                path.write_text("{" + ",".join(parts) + "}\n")
                r = status(["nested-rendering", filename, field, depth])
                assert r["exit"] == (1 if filename == "season_contract.json" else 0), r
                if filename == "contest_ledger.jsonl" and depth < min(bounds.values()) - 15:
                    assert "ledger: readable; rows: 1; bad lines: 0" in r["output"], r
                path.unlink()
    report = dict(integer_checks=integer_checks, status_probes=len(records),
                  changed_files={name: len(lines) for name, lines in changed.items()}, records=records)
    (HERE / "source-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(integer_checks=integer_checks, status_probes=len(records), changed_files=report["changed_files"]), indent=2))


if __name__ == "__main__":
    main()
