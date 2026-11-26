"""Adversarial probe for Item 3: Checklist B8b(2) operator step across all decision/entry states and src/ archive readers."""
from __future__ import annotations

import builtins
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import cli, contest_state, daily_decision, day_outcome, picks, reconcile_receipt, scheduler, skip_policy_shadow, web
from bts.data import backup, lineup_analyze
from bts.health import analytics_artifacts_missing, fallback_defer, late_delivery, result_resolution, scheduler_state_integrity
from bts.state import export as state_export


ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, 0, tzinfo=ET)
TODAY = "2027-04-20"
FIRST = "2027-04-16"
SECOND = "2027-04-17"
RESOLVED = "2027-04-18"
ARCHIVE_NAME = "set_aside_never_entered.json"
STDERR_RE = re.compile(
    r"^Season replay unavailable for (?P<date>.+) \((?P<reason>[^()]+)\); streak retained, not recalculated\.$"
)


def _slot(day: str) -> dict:
    return {
        "batter_name": "Synthetic Batter",
        "batter_id": 100001,
        "team": "PIT",
        "lineup_position": 1,
        "pitcher_name": "Synthetic Pitcher",
        "pitcher_id": 100002,
        "p_game_hit": 0.80,
        "flags": [],
        "projected_lineup": False,
        "game_pk": 101,
        "game_time": f"{day}T23:00:00Z",
        "pitcher_team": "CHC",
    }


def _pick_payload(day: str, result: str | None = None, **extra) -> dict:
    base = {
        "date": day,
        "run_time": f"{day}T19:00:00Z",
        "pick": _slot(day),
        "double_down": None,
        "runner_up": None,
        "result": result,
    }
    if result in ("hit", "miss", "void"):
        base["slot_results"] = {"pick": result}
    base.update(extra)
    return base


def _run_reconcile(picks_dir: Path, lookback: int = 0):
    real_cls = reconcile_receipt.ReconcileReceipt
    with patch.object(
        reconcile_receipt,
        "ReconcileReceipt",
        side_effect=lambda **kw: real_cls(clock=lambda: RUN, lookback_days=kw["lookback_days"]),
    ):
        return CliRunner().invoke(cli.cli, ["reconcile", "--picks-dir", str(picks_dir), "--lookback", str(lookback)])


def _simulate_operator_step(
    working: Path,
    *,
    mlb_entered: set[str],
    mlb_confirmable: bool = True,
    strict_unreadable_decision: bool = False,
) -> dict:
    """Follow B8b(2) literally on one reconcile run and report what the operator step does."""
    picks_dir = working / "data" / "picks"
    res = _run_reconcile(picks_dir, lookback=0)
    lines = res.stderr.splitlines()
    if not lines:
        return {"action": "none_needed", "stdout": res.stdout, "stderr": res.stderr}
    assert len(lines) == 1, lines
    m = STDERR_RE.match(lines[0])
    assert m is not None, lines[0]
    day = m.group("date")
    reason = m.group("reason")

    if reason != "no final result on the pick":
        return {"action": "stop_other_reason", "date": day, "reason": reason}
    if not mlb_confirmable or day in mlb_entered:
        return {"action": "stop_mlb_entered_or_unconfirmed", "date": day}

    dec_path = picks_dir / day / "decision.json"
    if dec_path.exists():
        try:
            raw_txt = dec_path.read_text()
            parsed = json.loads(raw_txt)
        except (OSError, json.JSONDecodeError) as exc:
            if strict_unreadable_decision:
                return {"action": "stop_decision_unreadable", "date": day, "error": type(exc).__name__}
            parsed = None
            raw_txt = ""
        if isinstance(parsed, dict) and parsed.get("scoreable") is True:
            return {"action": "stop_decision_scoreable_true", "date": day}
        if '"scoreable": true' in raw_txt or '"scoreable":true' in raw_txt:
            return {"action": "stop_decision_raw_scoreable_true", "date": day}

    dest = Path("data/picks") / day / ARCHIVE_NAME
    (working / dest.parent).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["mv", str(Path("data/picks") / f"{day}.json"), str(dest)],
        cwd=working,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if proc.returncode != 0:
        return {"action": "mv_failed", "date": day, "stderr": proc.stderr}
    after = _run_reconcile(picks_dir, lookback=0)
    return {
        "action": "moved_and_reran",
        "date": day,
        "after_stderr": after.stderr,
        "after_stdout": after.stdout,
    }


def check_checklist_text() -> dict:
    doc = (Path(__file__).resolve().parents[2] / "docs" / "ops" / "2027-season-start.md").read_text()
    b8b_lines = [ln for ln in doc.splitlines() if ln.startswith("| B8b |")]
    assert len(b8b_lines) == 1
    b8b = b8b_lines[0]
    return {
        "exact_stderr_quote": (
            "`Season replay unavailable for <date> (no final result on the pick); streak retained, not recalculated.`" in b8b
        ),
        "mentions_mlb_confirm_nothing": "confirm in the MLB contest that NOTHING was entered on that date" in b8b,
        "mentions_decision_json_check": "Check `data/picks/<date>/decision.json` before moving the pick file." in b8b,
        "mentions_scoreable_true_stop": 'If the decision file exists and says `"scoreable": true`, leave the pick file in place.' in b8b,
        "mentions_private_lock": "private lock on a box that enters nothing in MLB" in b8b,
        "mentions_exact_mv": "`mv data/picks/<date>.json data/picks/<date>/set_aside_never_entered.json`" in b8b,
        "mentions_rerun_next_blocker": "Rerun `bts reconcile` after the move" in b8b and "may name the next blocking date" in b8b,
        "mentions_confirm_each_separately": "Confirm each date separately in MLB and check its decision before moving its file." in b8b,
        "mentions_other_three_reasons": "For the other three reasons, leave the files in place and report the message." in b8b,
    }


def check_operator_scenarios() -> dict:
    results = {}
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as td:
        base = Path(td)

        def _seed_base_picks(name: str) -> tuple[Path, Path]:
            w = base / name
            pdir = w / "data" / "picks"
            pdir.mkdir(parents=True)
            (pdir / "streak.json").write_text(json.dumps({"streak": 9, "saver_available": False, "season": 2027}))
            (pdir / f"{RESOLVED}.json").write_text(json.dumps(_pick_payload(RESOLVED, "hit")))
            return w, pdir

        # 1. One blocker (never entered, no decision.json)
        w1, p1 = _seed_base_picks("one_blocker")
        (p1 / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        step1 = _simulate_operator_step(w1, mlb_entered=set())
        results["one_blocker"] = {
            "step": step1,
            "rebuilt_streak": json.loads((p1 / "streak.json").read_text())["streak"],
            "archive_exists": (p1 / FIRST / ARCHIVE_NAME).is_file(),
        }

        # 2a. Two blockers (both never entered)
        w2, p2 = _seed_base_picks("two_blockers_both_clear")
        (p2 / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        (p2 / f"{SECOND}.json").write_text(json.dumps(_pick_payload(SECOND, None)))
        step2_first = _simulate_operator_step(w2, mlb_entered=set())
        streak_mid = json.loads((p2 / "streak.json").read_text())["streak"]
        step2_second = _simulate_operator_step(w2, mlb_entered=set())
        results["two_blockers_both_clear"] = {
            "first_move": step2_first,
            "streak_after_first_move": streak_mid,
            "second_move": step2_second,
            "rebuilt_streak_after_second_move": json.loads((p2 / "streak.json").read_text())["streak"],
        }

        # 2b. Two blockers where second has scoreable: true (private lock)
        w2b, p2b = _seed_base_picks("two_blockers_second_committed")
        (p2b / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        (p2b / f"{SECOND}.json").write_text(json.dumps(_pick_payload(SECOND, None)))
        daily_decision.write_decision(
            SECOND, p2b, action="single", source="mdp", delivery_status="private_locked", scoreable=True, now=RUN
        )
        s2b_1 = _simulate_operator_step(w2b, mlb_entered=set())
        s2b_2 = _simulate_operator_step(w2b, mlb_entered=set())
        results["two_blockers_second_committed"] = {
            "first_move": s2b_1,
            "second_check": s2b_2,
            "second_pick_preserved": (p2b / f"{SECOND}.json").is_file(),
            "streak_retained": json.loads((p2b / "streak.json").read_text())["streak"],
        }

        # 3. decision.json says "scoreable": true (private lock, nothing in MLB)
        w3, p3 = _seed_base_picks("decision_scoreable_true")
        (p3 / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        daily_decision.write_decision(
            FIRST, p3, action="single", source="mdp", delivery_status="private_locked", scoreable=True, now=RUN
        )
        results["decision_scoreable_true"] = {
            "step": _simulate_operator_step(w3, mlb_entered=set()),
            "pick_preserved": (p3 / f"{FIRST}.json").is_file(),
        }

        # 4. decision.json says "scoreable": false (both not-entered and MLB-entered N1 case)
        w4a, p4a = _seed_base_picks("decision_scoreable_false_not_entered")
        (p4a / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        daily_decision.write_decision(
            FIRST, p4a, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN
        )
        w4b, p4b = _seed_base_picks("decision_scoreable_false_mlb_entered")
        (p4b / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        daily_decision.write_decision(
            FIRST, p4b, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN
        )
        results["decision_scoreable_false"] = {
            "not_entered": _simulate_operator_step(w4a, mlb_entered=set()),
            "mlb_entered": _simulate_operator_step(w4b, mlb_entered={FIRST}),
            "mlb_entered_pick_preserved": (p4b / f"{FIRST}.json").is_file(),
        }

        # 5. Missing decision.json:
        #    (a) plain preview not entered -> moved;
        #    (b) MLB-entered without decision.json -> stopped by MLB check;
        #    (c) delivered model pick (notification_sent=True, e.g. suspended game) NOT entered in MLB & missing decision.json;
        #    (d) private lock crashed before decision.json (scheduler_state.json pick_locked=True), nothing in MLB.
        w5b, p5b = _seed_base_picks("decision_missing_mlb_entered")
        (p5b / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))

        w5c, p5c = _seed_base_picks("decision_missing_delivered_unentered")
        (p5c / f"{FIRST}.json").write_text(
            json.dumps(_pick_payload(FIRST, None, notification_sent=True, notification_id="dm-123"))
        )
        dp_5c = picks.load_pick(FIRST, p5c)
        scoreable_5c = daily_decision.is_scoreable_commit(FIRST, p5c, dp_5c)

        w5d, p5d = _seed_base_picks("decision_missing_private_crash_locked")
        (p5d / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        st_5d = scheduler.SchedulerState(FIRST, RUN.isoformat(), [], [], [], True, RUN.isoformat(), None, None)
        scheduler.save_state(st_5d, p5d)

        results["decision_missing"] = {
            "mlb_entered_step": _simulate_operator_step(w5b, mlb_entered={FIRST}),
            "delivered_unentered_is_scoreable_commit": scoreable_5c,
            "delivered_unentered_literal_b8b": _simulate_operator_step(w5c, mlb_entered=set()),
            "private_crash_locked_literal_b8b": _simulate_operator_step(w5d, mlb_entered=set()),
        }

        # 6. Unreadable decision.json (corrupt JSON '{' and chmod 000 on a private_locked commit, nothing in MLB)
        w6a, p6a = _seed_base_picks("decision_corrupt_truncated")
        (p6a / f"{FIRST}.json").write_text(
            json.dumps(_pick_payload(FIRST, None, notification_sent=True, notification_id="dm-123"))
        )
        (p6a / FIRST).mkdir(parents=True)
        (p6a / FIRST / "decision.json").write_text("{")
        scoreable_6a = daily_decision.is_scoreable_commit(FIRST, p6a, picks.load_pick(FIRST, p6a))

        w6b, p6b = _seed_base_picks("decision_chmod_000")
        (p6b / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        daily_decision.write_decision(
            FIRST, p6b, action="single", source="mdp", delivery_status="private_locked", scoreable=True, now=RUN
        )
        dec_6b = p6b / FIRST / "decision.json"
        dec_6b.chmod(0)
        try:
            step_6b_literal = _simulate_operator_step(w6b, mlb_entered=set(), strict_unreadable_decision=False)
        finally:
            dec_6b.chmod(0o600)

        w6c, p6c = _seed_base_picks("decision_chmod_000_mlb_entered")
        (p6c / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, None)))
        daily_decision.write_decision(
            FIRST, p6c, action="single", source="mdp", delivery_status="delivered", scoreable=True, now=RUN
        )
        dec_6c = p6c / FIRST / "decision.json"
        dec_6c.chmod(0)
        try:
            step_6c_mlb = _simulate_operator_step(w6c, mlb_entered={FIRST}, strict_unreadable_decision=False)
        finally:
            dec_6c.chmod(0o600)

        results["decision_unreadable"] = {
            "corrupt_truncated_with_delivered_is_scoreable_commit": scoreable_6a,
            "corrupt_truncated_literal_b8b": _simulate_operator_step(w6a, mlb_entered=set(), strict_unreadable_decision=False),
            "chmod_000_private_literal_b8b": step_6b_literal,
            "chmod_000_mlb_entered_step": step_6c_mlb,
        }

        # 7. Other three reasons -> never move
        other_reasons = {}
        for kind, day, body, markers in [
            ("corrupt", FIRST, "{", []),
            ("today_applied", TODAY, "{}", [TODAY]),
        ]:
            wr, pr = _seed_base_picks(f"reason_{kind}")
            (pr / "streak.json").write_text(json.dumps({"streak": 9, "season": 2027, "applied_dates": markers}))
            (pr / f"{day}.json").write_text(body)
            other_reasons[kind] = _simulate_operator_step(wr, mlb_entered=set())
        results["other_reasons_stopped"] = other_reasons

        # 8. Noncanonical stem (2027-04-16\n.json) -> rendered date has literal \n, mv of rendered path fails safely
        wn, pn = _seed_base_picks("lf_stem_mv")
        (pn / f"{FIRST}\n.json").write_text("{}")
        (pn / f"{FIRST}.json").write_text(json.dumps(_pick_payload(FIRST, "hit")))
        results["lf_stem_mv"] = {
            "step": _simulate_operator_step(wn, mlb_entered=set()),
            "canonical_pick_untouched": (pn / f"{FIRST}.json").is_file(),
            "lf_pick_still_present": (pn / f"{FIRST}\n.json").is_file(),
        }

    return results


def check_src_readers_never_touch_archive() -> dict:
    src_root = Path(__file__).resolve().parents[2] / "src" / "bts"
    # Static scan of all .py files under src/bts for Archive name or wildcard scans of date subdirectories
    literal_mentions = []
    nested_globs = []
    for py in sorted(src_root.rglob("*.py")):
        txt = py.read_text()
        rel = str(py.relative_to(src_root.parents[1]))
        if ARCHIVE_NAME in txt:
            literal_mentions.append(rel)
        for m in re.finditer(r"\.(?:glob|rglob)\(([^)]+)\)", txt):
            nested_globs.append((rel, m.group(0)))

    # Dynamic open/os.open trace across every picks_dir / <date>/ reader in src/
    opened_archive: list[str] = []
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as td:
        working = Path(td)
        picks_dir = working / "data" / "picks"
        picks_dir.mkdir(parents=True)
        (picks_dir / "streak.json").write_text(
            json.dumps({"streak": 4, "saver_available": True, "season": 2027, "applied_dates": []})
        )
        (picks_dir / f"{RESOLVED}.json").write_text(json.dumps(_pick_payload(RESOLVED, "hit")))
        daily_decision.write_decision(
            FIRST, picks_dir, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN
        )
        st = scheduler.SchedulerState(FIRST, RUN.isoformat(), [], [], [], False, None, None, None)
        scheduler.save_state(st, picks_dir)

        archive_path = picks_dir / FIRST / ARCHIVE_NAME
        # Poison bytes: any reader that opens and parses set_aside_never_entered.json will fail or record an open!
        archive_path.write_bytes(b"\xff\xfe{poison_invalid_utf8_and_json")

        orig_path_open = Path.open
        orig_builtin_open = builtins.open
        orig_os_open = os.open

        def _record(target):
            if isinstance(target, (str, bytes, os.PathLike)):
                name = os.fsdecode(os.fspath(target))
                if ARCHIVE_NAME in name:
                    opened_archive.append(name)

        def _traced_path_open(self_p, *a, **k):
            _record(self_p)
            return orig_path_open(self_p, *a, **k)

        def _traced_builtin_open(file, *a, **k):
            _record(file)
            return orig_builtin_open(file, *a, **k)

        def _traced_os_open(path, flags, *a, **k):
            _record(path)
            return orig_os_open(path, flags, *a, **k)

        cfg = {
            "orchestrator": {"picks_dir": str(picks_dir)},
            "scheduler": {"pick_delivery": "private", "entry_intent": "research"},
        }
        with patch.object(Path, "open", _traced_path_open), \
             patch.object(builtins, "open", _traced_builtin_open), \
             patch.object(os, "open", _traced_os_open), \
             patch.object(web, "PICKS_DIR", picks_dir):
            rep = picks._replay_season_streak_detail(picks_dir, 2027, TODAY)
            rec_res = _run_reconcile(picks_dir, lookback=8)
            lp = picks.load_pick(FIRST, picks_dir)
            lpb = picks.load_pick_bytes(FIRST, picks_dir)
            ld = daily_decision.load_decision(FIRST, picks_dir)
            ldb = daily_decision.load_decision_bytes(FIRST, picks_dir)
            isc = daily_decision.is_scoreable_commit(FIRST, picks_dir, lp)
            dout = day_outcome.gather(FIRST, picks_dir, cfg, expected_username=None, fetch_schedule=lambda _d: [{"gamePk": 101}])
            sps = skip_policy_shadow.record_pending_skips(picks_dir, now=RUN)
            sls = scheduler.load_state(FIRST, picks_dir)
            wap = web.load_all_picks()
            wss = web.load_scheduler_state(FIRST)
            lds = contest_state.load_decision_streak_state(picks_dir, require_contest_state=False)
            la = lineup_analyze.backfill_from_scheduler_state(picks_dir)
            backup.verify_restored_ops_tree(working)
            snap = state_export.export_initial_state(picks_dir, working / "export.json")
            h_ssi = scheduler_state_integrity.check(picks_dir, today=date.fromisoformat(TODAY))
            h_fd = fallback_defer.check(picks_dir, today=date.fromisoformat(TODAY), now=RUN)
            h_ld = late_delivery.check(picks_dir, today=date.fromisoformat(FIRST), now=RUN)
            h_aam = analytics_artifacts_missing._load_scheduler_state(picks_dir, FIRST)
            h_rr = result_resolution.check(picks_dir, today=date.fromisoformat(TODAY))

        return {
            "literal_mentions_in_src": literal_mentions,
            "globs_in_src_count": len(nested_globs),
            "opened_archive_paths": opened_archive,
            "replay_detail": rep,
            "reconcile_exit": rec_res.exit_code,
            "load_pick_first": lp,
            "load_pick_bytes_first": lpb,
            "decision_loaded_scoreable": ld["scoreable"] if ld else None,
            "is_scoreable_commit": isc,
            "day_outcome_state": dout.state,
            "web_picks_dates": [p["date"] for p in wap],
            "export_dates": [p["date"] for p in snap["historical_picks"]],
            "contest_state_streak": lds.streak,
            "health_alerts_count": len(h_ssi) + len(h_fd) + len(h_ld) + len(h_rr),
            "lineup_backfill_count": len(la),
            "skip_shadow_count": len(sps),
            "scheduler_state_loaded": sls is not None,
            "web_scheduler_state_keys": sorted(wss.keys()),
            "aam_scheduler_state_keys": sorted(h_aam.keys()),
        }


def main() -> None:
    signal.alarm(30)
    out = {
        "checklist_text": check_checklist_text(),
        "operator_scenarios": check_operator_scenarios(),
        "src_readers": check_src_readers_never_touch_archive(),
    }
    out_path = Path(__file__).with_name("item3_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print("checklist_text all true:", all(out["checklist_text"].values()), out["checklist_text"])
    print("opened_archive_paths:", out["src_readers"]["opened_archive_paths"])
    print("literal_mentions_in_src:", out["src_readers"]["literal_mentions_in_src"])
    for k, v in out["operator_scenarios"].items():
        print(f"  scenario[{k}]: {json.dumps(v)}")


if __name__ == "__main__":
    main()
