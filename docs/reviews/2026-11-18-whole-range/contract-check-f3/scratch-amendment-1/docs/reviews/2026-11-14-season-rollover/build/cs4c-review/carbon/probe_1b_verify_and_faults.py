"""Verification of Probe 1 base/branch captures, 320-cell loader parity, fault injection, and stamp precision."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_ROOT = REVIEW_DIR / "scratch" / "base_1c6cbc77"
ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
SCHEMA = "bts_pick_automatic_reading_v1"


def _expected_stale_message(warning_day_iso: str, detail: str) -> str:
    return (
        f"The pick was delivered using an automatic contest reading flagged stale ({detail}). "
        "Check MLB. If the streak or saver differs, use bts set-contest-streak "
        f"--date {warning_day_iso} --streak N --best-streak B --saver S with the "
        "scheduler's picks directory and the values shown in MLB. "
        "Use the current ET date when you run it. Check the latest fetch status."
    )


def _compare_scenarios(base_sc: dict, branch_sc: dict) -> None:
    assert set(base_sc) == set(branch_sc)
    stale_warn_prefix = "  STALE CONTEST READING — "
    stale_cells = 0
    fresh_stamped_cells = 0
    dated_stamped_cells = 0
    control_cells = 0

    for sid in sorted(base_sc):
        b0 = base_sc[sid]
        b1 = branch_sc[sid]

        assert b0["ret"] == b1["ret"], f"{sid}: return value mismatch {b0['ret']} vs {b1['ret']}"
        assert b0["reentry_ret"] == b1["reentry_ret"], f"{sid}: reentry mismatch"
        assert b0["dm_calls"] == b1["dm_calls"], f"{sid}: dm_calls mismatch"
        assert b0["post_calls"] == b1["post_calls"], f"{sid}: post_calls mismatch"
        assert b0["capture_calls"] == b1["capture_calls"], f"{sid}: capture mismatch"
        assert b0["in_mem_locked"] == b1["in_mem_locked"], f"{sid}: in_mem_locked mismatch"
        assert b0["evo_lines"] == b1["evo_lines"], f"{sid}: lineup_evolution mismatch"
        assert b0["crit_bytes_exact"] == b1["crit_bytes_exact"], (
            f"{sid}: health_dm_delivery_status.json bytes changed!"
        )

        # Pick JSON identical apart from trailing automatic_reading
        if b0["pick_json"] is None:
            assert b1["pick_json"] is None, f"{sid}: pick_json existence mismatch"
        else:
            p1 = dict(b1["pick_json"])
            assert list(p1)[-2:] == ["chosen_on", "automatic_reading"], (
                f"{sid}: wrong trailing keys {list(p1)[-2:]}"
            )
            p1.pop("automatic_reading")
            assert p1 == b0["pick_json"], f"{sid}: non-automatic_reading pick fields differ"

        # Files in <picks_dir>/<DAY>/: decision.json and scheduler_state.json must be
        # byte-identical with no automatic_reading; pick archives must have
        # ['chosen_on', 'automatic_reading', <archive_key>] as last three keys!
        assert set(b0["archives"]) == set(b1["archives"]), f"{sid}: <DAY>/*.json filenames differ"
        for aname in b0["archives"]:
            if aname in {"decision.json", "scheduler_state.json"}:
                assert "automatic_reading" not in b1["archives"][aname], (
                    f"{sid}: {aname} unexpectedly gained automatic_reading!"
                )
                assert b0["archives"][aname] == b1["archives"][aname], (
                    f"{sid}: {aname} differs between base and branch!"
                )
            else:
                a1 = dict(b1["archives"][aname])
                keys = list(a1)
                assert keys[-3:-1] == ["chosen_on", "automatic_reading"], (
                    f"{sid}: {aname} archive key order {keys[-3:]}"
                )
                a1.pop("automatic_reading")
                assert a1 == b0["archives"][aname], f"{sid}: {aname} archive payload differs"

        is_qualifying_attempt = (
            len(b1["attempt_snap"]) == 1
            and not sid.startswith("not_required:")
            and not sid.startswith("private:")
        )
        if not is_qualifying_attempt:
            control_cells += 1
            assert b1["auto_fn_calls"] == 0, f"{sid}: unexpected automatic_reading_record call!"
            assert b0["alert_dm_calls"] == b1["alert_dm_calls"], f"{sid}: unexpected alert DM"
            assert b0["stderr_lines"] == b1["stderr_lines"], f"{sid}: unexpected stderr diff"
            assert b1["warn_status"] is None, f"{sid}: unexpected warn status file"
        else:
            assert b1["auto_fn_calls"] == 1, f"{sid}: expected 1 automatic_reading_record call"
            att_auto = b1["attempt_snap"][0]["automatic_reading"]
            fin_auto = b1["pick_json"]["automatic_reading"]
            assert att_auto == fin_auto, f"{sid}: attempt vs final automatic_reading mismatch"

            confirmed = b1["ret"] is True
            if ":dated" in sid:
                dated_stamped_cells += 1
                assert fin_auto is None, f"{sid}: dated should stamp None"
                assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                assert b0["stderr_lines"] == b1["stderr_lines"]
                assert b1["warn_status"] is None
            elif ":fresh_auto" in sid:
                fresh_stamped_cells += 1
                assert fin_auto is not None and fin_auto["stale"] is False and fin_auto["reason"] is None
                assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                assert b0["stderr_lines"] == b1["stderr_lines"]
                assert b1["warn_status"] is None
            else:
                assert fin_auto is not None and fin_auto["stale"] is True
                if not confirmed:
                    assert b1["pick_json"]["delivery_attempted"] is False
                    assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                    assert b0["stderr_lines"] == b1["stderr_lines"]
                    assert b1["warn_status"] is None
                else:
                    stale_cells += 1
                    reason = fin_auto["reason"]
                    if reason == "old":
                        detail = (
                            f"last accepted fetch {fin_auto['recorded_at']}, "
                            f"about {fin_auto['age_hours']:.2f} hours ago"
                        )
                    elif reason == "future":
                        detail = f"its recorded_at {fin_auto['recorded_at']} is in the future"
                    else:
                        detail = "it has no usable recorded_at"
                    msg = _expected_stale_message(DAY, detail)
                    expected_line = stale_warn_prefix + msg
                    b1_non_warn = [ln for ln in b1["stderr_lines"] if ln != expected_line]
                    assert b1["stderr_lines"].count(expected_line) == 1, f"{sid}: missing/dup stderr warning"
                    assert b1_non_warn == b0["stderr_lines"], f"{sid}: other stderr lines changed"
                    if sid.startswith("no_recipient:public:"):
                        assert b1["alert_dm_calls"] == b0["alert_dm_calls"] == []
                        assert b1["warn_status"] is None, f"{sid}: Amendment 1 violated"
                    else:
                        expected_env = ("operator.test", f"BTS health WARN attention:\n- [contest_state] {msg}")
                        extra = [c for c in b1["alert_dm_calls"] if c not in b0["alert_dm_calls"]]
                        assert extra == [expected_env], f"{sid}: unexpected warning DM {extra}"
                        assert b1["warn_status"] is not None
                        assert b1["warn_status"]["status"] == "sent"
                        assert b1["warn_status"]["sent_day"] == DAY
                        assert b1["warn_status"]["sent_sources"] == ["WARN_ATTENTION:contest_state:stale_reading"]

    print(
        f"[PASS] Base vs Branch delivery parity: {len(base_sc)} scenarios "
        f"(controls={control_cells}, dated_stamped={dated_stamped_cells}, "
        f"fresh_stamped={fresh_stamped_cells}, confirmed_stale={stale_cells}) matched 100%."
    )


def _loader_worker() -> dict:
    sys.path.insert(0, str(REVIEW_DIR))
    from netguard import forbid_network

    with forbid_network(allow_loopback=False):
        from bts import contest_state

        out = {}
        for req in (False, True):
            for switch_mode in ("none", "active", "revoked", "wrong_id"):
                for manual_mode in ("none", "dated_today", "dated_yesterday", "legacy_unexpired", "legacy_expired"):
                    for auto_mode in ("none", "v2_valid", "v2_naive_time", "v2_nonstr_time", "v1_legacy"):
                        for gap_mode in ("none", "hit_1", "miss_1", "hit_2"):
                            with tempfile.TemporaryDirectory() as td:
                                pd_dir = Path(td) / "picks"
                                acct = pd_dir / "account_state"
                                acct.mkdir(parents=True, exist_ok=True)
                                (pd_dir / "streak.json").write_text(json.dumps({
                                    "season": 2027, "streak": 4, "saver_available": True,
                                }))
                                if switch_mode != "none":
                                    (acct / "season_contract.json").write_text(json.dumps({
                                        "schema_version": "bts_season_contract_v1",
                                        "season": 2027,
                                        "status": "revoked" if switch_mode == "revoked" else "automatic",
                                        "activation_id": ("b" * 32 if switch_mode == "wrong_id" else "a" * 32),
                                        "switched_at": NOW.isoformat(),
                                        "revoked_at": NOW.isoformat() if switch_mode == "revoked" else None,
                                        "note": "revoked" if switch_mode == "revoked" else None,
                                    }))
                                if manual_mode == "dated_today":
                                    (acct / "contest_streak.manual.json").write_text(json.dumps({
                                        "schema_version": "bts_contest_streak_manual_v3",
                                        "season": 2027, "effective_date": DAY,
                                        "active_streak": 11, "best_streak": 15, "saver_state": "active",
                                        "recorded_at": "2027-07-15T10:00:00Z",
                                        "expires_at": f"{DAY}T23:59:59.999999-04:00",
                                        "source": "manual_cli",
                                    }))
                                elif manual_mode == "dated_yesterday":
                                    (acct / "contest_streak.manual.json").write_text(json.dumps({
                                        "schema_version": "bts_contest_streak_manual_v3",
                                        "season": 2027, "effective_date": "2027-07-14",
                                        "active_streak": 11, "best_streak": 15, "saver_state": "active",
                                        "recorded_at": "2027-07-14T10:00:00Z",
                                        "expires_at": "2027-07-14T23:59:59.999999-04:00",
                                        "source": "manual_cli",
                                    }))
                                elif manual_mode in {"legacy_unexpired", "legacy_expired"}:
                                    exp = (NOW + timedelta(hours=2) if manual_mode == "legacy_unexpired"
                                           else NOW - timedelta(hours=2)).isoformat()
                                    (acct / "contest_streak.manual.json").write_text(json.dumps({
                                        "schema_version": "bts_contest_streak_manual_v1",
                                        "active_streak": 9, "best_streak": 14,
                                        "source_date": "2027-07-13",
                                        "recorded_at": "2027-07-13T08:00:00",
                                        "override_expires_at": exp,
                                    }))
                                if auto_mode != "none":
                                    rec_val: object = "2027-07-13T13:30:00-04:00"
                                    if auto_mode == "v2_naive_time":
                                        rec_val = "2027-07-13T13:30:00"
                                    elif auto_mode == "v2_nonstr_time":
                                        rec_val = 12345
                                    auto_body = {
                                        "schema_version": ("bts_contest_streak_auto_v1"
                                                           if auto_mode == "v1_legacy"
                                                           else "bts_contest_streak_auto_v2"),
                                        "active_streak": 8, "best_streak": 12,
                                        "source": "mlb_bts_profile",
                                        "source_date": "2027-07-13",
                                        "recorded_at": rec_val,
                                    }
                                    if auto_mode != "v1_legacy":
                                        auto_body["season"] = 2027
                                        auto_body["activation_id"] = "a" * 32
                                    (acct / "contest_streak.json").write_text(json.dumps(auto_body))
                                if gap_mode in {"hit_1", "hit_2"}:
                                    (pd_dir / "2027-07-14.json").write_text(json.dumps({"result": "hit"}))
                                if gap_mode == "hit_2":
                                    (pd_dir / "2027-07-15.json").write_text(json.dumps({"result": "hit"}))
                                if gap_mode == "miss_1":
                                    (pd_dir / "2027-07-14.json").write_text(json.dumps({"result": "miss"}))

                                key = f"{req}:{switch_mode}:{manual_mode}:{auto_mode}:{gap_mode}"
                                try:
                                    st = contest_state.load_decision_streak_state(
                                        pd_dir, require_contest_state=req, now=NOW,
                                    )
                                    d = asdict(st)
                                    if d.get("contest_source_date") is not None:
                                        d["contest_source_date"] = d["contest_source_date"].isoformat()
                                    selected = contest_state.load_contest_streak_state(
                                        pd_dir, now=NOW, require_dated_auto=req,
                                    )
                                    out[key] = {
                                        "ok": True,
                                        "state": d,
                                        "selected_recorded_at": selected.recorded_at if selected else None,
                                    }
                                except contest_state.ContestStateError as exc:
                                    out[key] = {
                                        "ok": False,
                                        "error": str(exc).replace(str(pd_dir), "<PICKS_DIR>"),
                                    }
        return out


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--loader-worker":
        sys.stdout.write(json.dumps(_loader_worker()))
        return

    base_sc = json.loads((REVIEW_DIR / "probe_1_base.json").read_text())
    branch_sc = json.loads((REVIEW_DIR / "probe_1_branch.json").read_text())
    _compare_scenarios(base_sc, branch_sc)

    def run_loader(src_path: Path) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{src_path}:{REPO_ROOT}"
        env["TZ"] = "America/New_York"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--loader-worker"],
            env=env, capture_output=True, text=True, check=True,
        )
        return json.loads(proc.stdout)

    b0_ld = run_loader(BASE_ROOT / "src")
    b1_ld = run_loader(REPO_ROOT / "src")
    assert set(b0_ld) == set(b1_ld)
    for k, v0 in b0_ld.items():
        v1 = b1_ld[k]
        assert v0["ok"] == v1["ok"], f"loader {k} ok mismatch"
        if not v0["ok"]:
            assert v0["error"] == v1["error"], f"loader {k} error mismatch"
        else:
            s1 = dict(v1["state"])
            assert list(s1)[-1] == "contest_recorded_at"
            rec_at = s1.pop("contest_recorded_at")
            assert s1 == v0["state"], f"loader {k} fields changed"
            if s1["observation_kind"] == "model":
                assert rec_at is None
            else:
                assert rec_at == v1["selected_recorded_at"], f"loader {k} contest_recorded_at mismatch"
    print(f"[PASS] Row 1 & Row 8 load_decision_streak_state parity: {len(b0_ld)} cells matched 100%.")

    sys.path.insert(0, str(REVIEW_DIR))
    import probe_1_2_7_delivery_stamp_row8 as p1
    p1._check_row8_source_invariance()
    p1._run_branch_adversarial_faults_and_stamp_precision()
    print("PROBE_1B_ALL_PASSED")


if __name__ == "__main__":
    main()
