"""Adversarial review Item 2 + Item 3 complete verification (with exact src/ reader names)."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
import sys
import tempfile

import pytest

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
for p in (str(REVIEW_DIR), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from probe_item2_item3_output_and_ops import (  # noqa: E402
    EARLY,
    RUN,
    TODAY,
    YEST,
    _alarm,
    _invoke_reconcile,
    _pick_dict,
    check_item2_outputs,
)


def verify_item3_operator_step_and_readers() -> dict:
    from bts import daily_decision, picks, skip_policy_shadow, web
    from bts.data import backup, lineup_analyze
    from bts.health import (
        analytics_artifacts_missing,
        fallback_defer,
        result_resolution,
        scheduler_state_integrity,
    )
    from bts.state import export as state_export
    from bts.state import verify as state_verify

    checklist_text = (REPO_ROOT / "docs" / "ops" / "2027-season-start.md").read_text()
    b8b_lines = [ln for ln in checklist_text.splitlines() if ln.startswith("| B8b |")]
    assert len(b8b_lines) == 1
    b8b = b8b_lines[0]

    expected_quote = (
        "`Season replay unavailable for <date> (no final result on the pick); streak retained, not recalculated.`"
    )
    assert expected_quote in b8b, b8b
    assert "`mv data/picks/<date>.json data/picks/<date>/set_aside_never_entered.json`" in b8b, b8b
    assert "confirm in the MLB contest that NOTHING was entered on that date" in b8b, b8b
    assert "Never move a file for a day on which a pick was entered" in b8b, b8b
    assert "For the other three reasons, leave the files in place and report the message." in b8b, b8b

    for pre_existing_dir in (False, True):
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
            work = Path(tmp)
            mp.chdir(work)
            picks_dir = work / "data" / "picks"
            picks_dir.mkdir(parents=True)
            (picks_dir / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", bluesky_posted=True)))
            (picks_dir / f"{EARLY}.json").write_text(json.dumps(_pick_dict(EARLY, None)))
            (picks_dir / f"{YEST}.json").write_text(json.dumps(_pick_dict(YEST, "hit", bluesky_posted=True)))
            (picks_dir / "streak.json").write_text(json.dumps({
                "streak": 99, "saver_available": False, "season": 2027, "applied_dates": [],
            }))
            if pre_existing_dir:
                daily_decision.write_decision(
                    EARLY, picks_dir, action="skip", source="mdp",
                    delivery_status="not_applicable", scoreable=False, now=RUN,
                )
                (picks_dir / EARLY / "scheduler_state.json").write_text(json.dumps({
                    "date": EARLY, "pick_locked": False, "games": [], "runs_completed": [],
                }))

            mp.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "hit"})
            r_before = _invoke_reconcile(Path("data/picks"), mp, lookback=8)
            assert r_before.exit_code == 0
            assert r_before.stderr == (
                f"Season replay unavailable for {EARLY} (no final result on the pick); "
                "streak retained, not recalculated.\n"
            )
            assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 99

            # Exact checklist move step:
            dest_dir = Path(f"data/picks/{EARLY}")
            dest_dir.mkdir(parents=True, exist_ok=True)
            Path(f"data/picks/{EARLY}.json").rename(dest_dir / "set_aside_never_entered.json")

            # Poisoned set_aside_never_entered.json on another date to catch any reader!
            poison_dir = picks_dir / "2027-04-12"
            poison_dir.mkdir(parents=True, exist_ok=True)
            (poison_dir / "set_aside_never_entered.json").write_bytes(b"\xff\xfe{poison_invalid_json")

            r_after = _invoke_reconcile(Path("data/picks"), mp, lookback=8)
            assert r_after.exit_code == 0
            assert r_after.stderr == ""
            assert r_after.stdout.endswith("No scoring changes detected. Streak: 2\n")
            assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 2

            # Exercise every src/ reader of picks_dir and picks_dir/<date>/
            assert picks._replay_season_streak(picks_dir, 2027, TODAY) == (2, True)
            assert picks.load_pick(EARLY, picks_dir) is None
            _ = daily_decision.load_decision(EARLY, picks_dir)
            _ = skip_policy_shadow.record_pending_skips(picks_dir, now=RUN)
            _ = skip_policy_shadow.find_aged_contradictions(picks_dir, now=RUN)
            _ = lineup_analyze.backfill_from_scheduler_state(picks_dir)
            assert scheduler_state_integrity.check(picks_dir, today=date(2027, 4, 20)) == []
            assert fallback_defer.check(picks_dir, today=date(2027, 4, 20)) == []
            assert result_resolution.check(picks_dir, today=date(2027, 4, 20)) == []
            _ = analytics_artifacts_missing.check(picks_dir, today=date(2027, 4, 20))

            snap_out = work / "exported.json"
            exported = state_export.export_initial_state(picks_dir, snap_out)
            assert EARLY not in {h["date"] for h in exported["historical_picks"]}
            regen_dir = work / "regen_picks"
            regen_dir.mkdir()
            for fn in ("2027-04-01.json", f"{YEST}.json", "streak.json"):
                (regen_dir / fn).write_bytes((picks_dir / fn).read_bytes())
            drift = state_verify.diff_pick_files(picks_dir, regen_dir)
            assert drift.issues == [], drift.issues

            acct = picks_dir / "account_state"
            acct.mkdir(parents=True, exist_ok=True)
            (acct / "saver_state.json").write_text(json.dumps({"state": "not_earned", "season": 2027}))
            (acct / "contest_ledger.jsonl").write_text(json.dumps({"streak": 2}) + "\n")
            daily_decision.write_decision(
                YEST, picks_dir, action="single", source="mdp",
                delivery_status="delivered", scoreable=True, now=RUN,
            )
            assert backup.verify_restored_ops_tree(work) == []

            with mp.context() as wmp:
                wmp.setattr(web, "PICKS_DIR", picks_dir)
                all_picks = web.load_all_picks()
                assert EARLY not in {r["date"] for r in all_picks}

            for py_file in (REPO_ROOT / "src").rglob("*.py"):
                txt = py_file.read_text()
                assert "set_aside_never_entered" not in txt, py_file
                assert not re.search(r'\.rglob\(\s*["\']\*\.json["\']\s*\)', txt), py_file

    return {
        "checklist_quote_matches_exact_stderr": True,
        "move_rebuilds_streak_both_absent_and_existing_dir": True,
        "all_src_readers_ignore_set_aside_even_when_poisoned": True,
    }


def main() -> int:
    item2 = check_item2_outputs()
    item3 = verify_item3_operator_step_and_readers()
    summary = {"status": "PASS", "item2": item2, "item3": item3}
    out_path = REVIEW_DIR / "item2_item3_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
