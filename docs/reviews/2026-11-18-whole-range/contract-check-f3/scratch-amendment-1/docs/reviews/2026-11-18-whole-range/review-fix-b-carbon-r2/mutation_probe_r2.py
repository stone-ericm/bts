"""Item 5 probe: mutate scratch copies of HEAD src/ and run focused pytest suites one at a time."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[1]
SRC = REPO / "src"
PYTHON = REPO / ".venv/bin/python"

MUTANTS = {
    "1a_rethread_nonblocking_r1": {
        "desc": "Re-thread non-blocking reader into status decision (Round-1 style)",
        "copy_from": REVIEW / "r1/src",
        "edits": [],
    },
    "1b_rethread_nonblocking_in_verdict": {
        "desc": "Re-thread read_regular_file_text inside season_contract.py verdict() without signature change",
        "copy_from": SRC,
        "edits": [(
            "bts/season_contract.py",
            "    def verdict(*, omit_selected_dated=False):\n        try:\n            return load_decision_streak_state(",
            "    def verdict(*, omit_selected_dated=False):\n"
            "        from unittest.mock import patch as _p\n"
            "        from bts.util import read_regular_file_text as _rr\n"
            "        try:\n"
            "            with _p.object(Path, 'read_text', lambda self, *a, **kw: _rr(self)):\n"
            "                return load_decision_streak_state(",
        )],
    },
    "2a_status_fallback_ignores_saver": {
        "desc": "Status automatic fallback (omit_selected_dated=True) ignores load_saver_state",
        "copy_from": SRC,
        "edits": [(
            "bts/contest_state.py",
            "    saver = (SaverState(state=contest.saver_state, season=contest.effective_date.year)\n"
            "             if dated_in_effect else load_saver_state(picks_dir, season=now_year))",
            "    saver = (SaverState(state=contest.saver_state, season=contest.effective_date.year)\n"
            "             if dated_in_effect else SaverState('uninitialized', None) if omit_selected_dated\n"
            "             else load_saver_state(picks_dir, season=now_year))",
        )],
    },
    "2b_status_verdict_ignores_saver_all": {
        "desc": "Status verdict() stubs load_saver_state to uninitialized for both live and automatic lines",
        "copy_from": SRC,
        "edits": [(
            "bts/season_contract.py",
            "    def verdict(*, omit_selected_dated=False):\n        try:\n            return load_decision_streak_state(",
            "    def verdict(*, omit_selected_dated=False):\n"
            "        import bts.contest_state as _cs\n"
            "        from bts.saver_state import SaverState as _SS\n"
            "        from unittest.mock import patch as _p\n"
            "        try:\n"
            "            with _p.object(_cs, 'load_saver_state', lambda *a, **kw: _SS('uninitialized', None)):\n"
            "                return load_decision_streak_state(",
        )],
    },
    "3_old_stamp_helper": {
        "desc": "Validate cooldown stamps with _reading_time_utc (Round-1 helper)",
        "copy_from": SRC,
        "edits": [(
            "bts/cli.py",
            "        try:\n"
            "            # Match the cooldown comparison. UTC conversion can overflow for an\n"
            "            # aware timestamp near the ends of datetime's representable range.\n"
            "            return datetime.fromisoformat(stamp).utcoffset() is not None\n"
            "        except ValueError:\n"
            "            return False",
            "        from bts.contest_state import _reading_time_utc\n"
            "        return _reading_time_utc(stamp) is not None",
        )],
    },
    "4_drop_offset_requirement": {
        "desc": "Drop timezone offset requirement when validating cooldown stamps",
        "copy_from": SRC,
        "edits": [(
            "bts/cli.py",
            "            return datetime.fromisoformat(stamp).utcoffset() is not None",
            "            datetime.fromisoformat(stamp)\n            return True",
        )],
    },
}

SUITES = [
    "tests/test_wr_b_contract.py",
    "tests/test_wr_b_edges.py",
    "tests/test_season_contract_cs4b1_edges.py",
    "tests/test_season_contract_cs4b2.py",
    "tests/test_contest_fetch.py",
]

summary = {}
mut_root = REVIEW / "mutants"
if mut_root.exists():
    shutil.rmtree(mut_root)
mut_root.mkdir(parents=True)

for name, spec in MUTANTS.items():
    target_src = mut_root / name / "src"
    shutil.copytree(spec["copy_from"], target_src)
    for rel, old, new in spec["edits"]:
        f = target_src / rel
        txt = f.read_text(encoding="utf-8")
        assert txt.count(old) == 1, f"{name}: target snippet count={txt.count(old)} in {rel}"
        f.write_text(txt.replace(old, new, 1), encoding="utf-8")

    env = dict(
        os.environ,
        TZ="America/New_York",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=f"{target_src}:{REPO}",
    )
    # Run WR-B suite first, then extended CS4b/contest_fetch controls if WR-B passes
    cmd_wrb = [
        "/usr/bin/nice", "-n", "19", str(PYTHON), "-m", "pytest",
        "-p", "no:cacheprovider",
        "tests/test_wr_b_contract.py", "tests/test_wr_b_edges.py",
        "-q", "--tb=line",
    ]
    res_wrb = subprocess.run(cmd_wrb, cwd=REPO, env=env, capture_output=True, text=True)
    log_path = REVIEW / f"mutant-{name}.log"
    combined_log = f"=== WR-B SUITE (rc={res_wrb.returncode}) ===\n{res_wrb.stdout}\n{res_wrb.stderr}\n"

    res_ext_rc = None
    if res_wrb.returncode == 0:
        cmd_ext = [
            "/usr/bin/nice", "-n", "19", str(PYTHON), "-m", "pytest",
            "-p", "no:cacheprovider",
            "tests/test_season_contract_cs4b1_edges.py",
            "tests/test_season_contract_cs4b2.py",
            "tests/test_contest_fetch.py",
            "-q", "--tb=line",
        ]
        res_ext = subprocess.run(cmd_ext, cwd=REPO, env=env, capture_output=True, text=True)
        res_ext_rc = res_ext.returncode
        combined_log += f"\n=== EXTENDED CONTROLS (rc={res_ext.returncode}) ===\n{res_ext.stdout}\n{res_ext.stderr}\n"

    log_path.write_text(combined_log, encoding="utf-8")
    failed_lines = [ln for ln in res_wrb.stdout.splitlines() if ln.startswith("FAILED ")]
    summary[name] = {
        "desc": spec["desc"],
        "wrb_rc": res_wrb.returncode,
        "wrb_failed_count": len(failed_lines),
        "wrb_failed": failed_lines,
        "extended_rc_if_wrb_survived": res_ext_rc,
        "survived_wrb": res_wrb.returncode == 0,
        "survived_all": res_wrb.returncode == 0 and res_ext_rc == 0,
    }
    print(
        f"{name:36} | wrb_rc={res_wrb.returncode} (failed={len(failed_lines)}) "
        f"| ext_rc={res_ext_rc} | survived_wrb={res_wrb.returncode == 0}",
        flush=True,
    )

(REVIEW / "mutants-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
shutil.rmtree(mut_root)
