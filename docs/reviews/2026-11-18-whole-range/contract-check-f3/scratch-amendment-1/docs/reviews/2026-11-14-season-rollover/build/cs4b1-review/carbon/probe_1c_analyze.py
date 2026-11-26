"""Analyze probe1b_base.json and probe1b_branch.json (5,400 cases each) + verify {} vs non-empty non-list on base/branch."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ID_A = "a" * 32


def main() -> None:
    base = json.loads((HERE / "probe1b_base.json").read_text(encoding="utf-8"))
    branch = json.loads((HERE / "probe1b_branch.json").read_text(encoding="utf-8"))
    assert base["count"] == branch["count"] == 5400
    assert branch["io_violations"] == []

    diff_buckets: dict[str, int] = {}
    unexpected: list[str] = []

    for key, b_rec in base["records"].items():
        sw_name, auto_name, resp_kind, dry_part = key.split("|")
        dry = dry_part == "dry=1"
        n_rec = branch["records"][key]

        if auto_name in {"not_utf8", "not_object"}:
            if not (b_rec["exit"] == n_rec["exit"] == 1 and b_rec["dms"] == n_rec["dms"] == []):
                unexpected.append(f"prior_crash mismatch {key}: base={b_rec['exit']} branch={n_rec['exit']}")
            diff_buckets["unchanged_prior_crash"] = diff_buckets.get("unchanged_prior_crash", 0) + 1
            continue

        if auto_name in {"other_account_user", "other_account_uid"}:
            if not (b_rec["exit"] == n_rec["exit"] == 2 and len(b_rec["dms"]) == len(n_rec["dms"]) == 1):
                unexpected.append(f"identity_guard mismatch {key}: base={b_rec} branch={n_rec}")
            diff_buckets["unchanged_identity_guard"] = diff_buckets.get("unchanged_identity_guard", 0) + 1
            continue

        # Note: pred_nonlist in probe1b used {"a": 1}, which raised AttributeError on base ('str' has no .get) -> exit 2 on base too!
        # Empty dict {} on base exits 0 (tested separately below).
        already_exit2_on_base = {
            "pred_null", "pred_nonlist", "pred_nonobj", "rounds_503", "rounds_429",
            "profile_null", "active_gt_best", "bool_streak",
        }
        newly_rejected_row3 = {"pred_missing", "rounds_empty", "best_58"}

        if resp_kind in already_exit2_on_base:
            if not (b_rec["exit"] == n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"existing_reject mismatch {key}: base={b_rec['exit']} branch={n_rec}")
            bucket = "row3_explicit_list_msg" if resp_kind in {"pred_null", "pred_nonlist"} else "unchanged_early_reject"
            diff_buckets[bucket] = diff_buckets.get(bucket, 0) + 1
            continue

        if resp_kind in newly_rejected_row3:
            if not (b_rec["exit"] == 0 and n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"row3_new_reject mismatch {key}: base_exit={b_rec['exit']} branch={n_rec}")
            if n_rec["before"].get("picks/account_state/contest_streak.json") != n_rec["after"].get("picks/account_state/contest_streak.json"):
                unexpected.append(f"row3_modified_auto {key}")
            diff_buckets["row3_newly_rejected"] = diff_buckets.get("row3_newly_rejected", 0) + 1
            continue

        assert resp_kind in {"valid_settled", "valid_unsettled", "pred_empty", "best_57"}

        is_corrupt_switch = sw_name.startswith("c") or sw_name.startswith("pair_")
        if is_corrupt_switch:
            if not (n_rec["exit"] == 2 and len(n_rec["dms"]) == 1 and n_rec["ledger_text"] is None):
                unexpected.append(f"corrupt_switch mismatch {key}: {n_rec}")
            expected_check = {
                "c1_nonobj": "check 1:", "c2_schema": "check 2:", "c3_bool_season": "check 3:",
                "c4_status": "check 4:", "c5_id": "check 5:", "c6_switched": "check 6:",
                "c7_revoked": "check 7:", "c8_note": "check 8:",
                "pair_c2_c8_other_yr": "check 2:", "pair_c3_c5": "check 3:", "pair_c6_c7": "check 6:",
            }[sw_name]
            if expected_check not in n_rec["out"]:
                unexpected.append(f"corrupt_switch wrong check {key}: out={n_rec['out']!r}")
            diff_buckets["row4_5_corrupt_switch"] = diff_buckets.get("row4_5_corrupt_switch", 0) + 1
            continue

        if dry:
            if not (n_rec["exit"] == 0 and n_rec["dms"] == [] and n_rec["before"] == n_rec["after"]):
                unexpected.append(f"dry_run mismatch {key}: {n_rec}")
            expected_phrase = (
                "season 2027 automatic; would write"
                if sw_name == "automatic"
                else "season 2027 not switched to automatic; would record reading in ledger and fetch status"
            )
            if expected_phrase not in n_rec["out"]:
                unexpected.append(f"dry_run phrase mismatch {key}: {n_rec['out']!r}")
            diff_buckets["row5_dry_run_valid"] = diff_buckets.get("row5_dry_run_valid", 0) + 1
            continue

        if sw_name == "automatic":
            if n_rec["exit"] != 0 or n_rec["dms"] != []:
                unexpected.append(f"auto_success mismatch {key}: {n_rec}")
            auto_doc = json.loads(n_rec["auto_text"])
            led_doc = json.loads(n_rec["ledger_text"].strip())
            if not (
                auto_doc["schema_version"] == "bts_contest_streak_auto_v2"
                and auto_doc["season"] == 2027
                and auto_doc["activation_id"] == ID_A
                and auto_doc["recorded_at"] == "2027-07-15T16:00:00Z"
            ):
                unexpected.append(f"auto_doc stamp wrong {key}: {auto_doc}")
            if not (
                led_doc["season"] == 2027
                and led_doc["switched"] is True
                and led_doc["activation_id"] == ID_A
                and led_doc["recorded_at"] == "2027-07-15T16:00:00Z"
            ):
                unexpected.append(f"ledger_doc stamp wrong {key}: {led_doc}")
            wo = n_rec["write_order"]
            has_settled_2027 = resp_kind in {"valid_settled", "best_57"}
            if not (
                len(wo) >= 3
                and wo[0].endswith("->picks/account_state/contest_streak.json")
                and wo[1].endswith("->health_state/contest_streak_fetch_status.json")
                and wo[2] == "picks/account_state/contest_ledger.jsonl"
            ):
                unexpected.append(f"auto write_order wrong {key}: {wo}")
            if has_settled_2027:
                if resp_kind == "valid_settled" and n_rec["saver_text"] is None:
                    unexpected.append(f"saver not initialized on valid_settled {key}")
                if resp_kind == "best_57" and n_rec["saver_text"] is not None:
                    unexpected.append(f"cold best_57 initialized saver unexpectedly {key}")
            else:
                if len(wo) != 3 or n_rec["saver_text"] is not None:
                    unexpected.append(f"unsettled touched saver {key}: {wo}")
            diff_buckets["row4_6_automatic_write"] = diff_buckets.get("row4_6_automatic_write", 0) + 1
        else:
            assert sw_name in {"absent", "other_season", "revoked"}
            if n_rec["exit"] != 0 or n_rec["dms"] != []:
                unexpected.append(f"unswitched mismatch {key}: {n_rec}")
            led_doc = json.loads(n_rec["ledger_text"].strip())
            if not (
                led_doc["season"] == 2027
                and led_doc["switched"] is False
                and led_doc["activation_id"] is None
                and led_doc["recorded_at"] == "2027-07-15T16:00:00Z"
            ):
                unexpected.append(f"unswitched ledger wrong {key}: {led_doc}")
            wo = n_rec["write_order"]
            if not (
                len(wo) == 2
                and wo[0].endswith("->health_state/contest_streak_fetch_status.json")
                and wo[1] == "picks/account_state/contest_ledger.jsonl"
            ):
                unexpected.append(f"unswitched write_order wrong {key}: {wo}")
            if "season 2027 is not switched to automatic; reading recorded in the ledger and the fetch status" not in n_rec["out"]:
                unexpected.append(f"unswitched output wrong {key}: {n_rec['out']!r}")
            diff_buckets["row4_6_unswitched_ledger_status_only"] = diff_buckets.get("row4_6_unswitched_ledger_status_only", 0) + 1

    # Verify pairwise failure precedence between base and branch
    bp, np = base["pairwise"], branch["pairwise"]
    assert bp["profile_null_vs_rounds503"] == np["profile_null_vs_rounds503"]
    assert not np["profile_null_vs_rounds503"]["rounds_called"] and np["profile_null_vs_rounds503"]["category"] == "actionable"
    assert bp["profile_null_vs_rounds429"] == np["profile_null_vs_rounds429"]
    assert not np["profile_null_vs_rounds429"]["rounds_called"] and np["profile_null_vs_rounds429"]["category"] == "actionable"
    assert bp["pred_null_vs_rounds503"] == np["pred_null_vs_rounds503"]
    assert np["pred_null_vs_rounds503"]["rounds_called"] and np["pred_null_vs_rounds503"]["category"] == "transient"
    assert bp["pred_null_vs_rounds429"] == np["pred_null_vs_rounds429"]
    assert np["pred_null_vs_rounds429"]["rounds_called"] and np["pred_null_vs_rounds429"]["category"] == "rate_limited"
    assert bp["pred_missing_vs_empty_rounds"]["exit"] == 0 and "predictions key is missing" in np["pred_missing_vs_empty_rounds"]["last_error"]
    assert bp["pred_null_vs_empty_rounds"]["exit"] == 2 and "predictions must be a list" in np["pred_null_vs_empty_rounds"]["last_error"]
    assert bp["pred_nonlist_vs_empty_rounds"]["exit"] == 2 and "predictions must be a list" in np["pred_nonlist_vs_empty_rounds"]["last_error"]
    assert bp["pred_nonobj_vs_empty_rounds"]["exit"] == 2 and "predictions must contain only objects" in np["pred_nonobj_vs_empty_rounds"]["last_error"]
    assert "invalid literal for int()" in np["bad_roundid_vs_best58"]["last_error"]
    assert ">= activeStreak" in np["active60_gt_best59_vs_cap57"]["last_error"]
    assert "<= 57" in np["active58_eq_best58_vs_cap57"]["last_error"]
    assert "seasonBestStreak must be an integer" in np["bool_best_streak"]["last_error"]

    # Also explicitly check predictions={} (empty dict): accepted (exit 0) on base, rejected (exit 2) on branch!
    empty_dict_snip = (
        "import sys, datetime as dt; from pathlib import Path; from types import SimpleNamespace; "
        "sys.path.insert(0, sys.argv[1]); "
        "import bts.cli as cli, bts.contest_fetch as cf, bts.leaderboard.auth as auth; "
        "from click.testing import CliRunner; "
        "auth.load_session_cookies = lambda: {'oktaid': 'u'}; auth.extract_uid = lambda c: 'u'; "
        "auth.fetch_login_session = lambda **kw: SimpleNamespace(user_id=1, username='u', xsid='x'); "
        "cf.fetch_profile = lambda *a: {'activeStreak': 7, 'seasonBestStreak': 9, 'predictions': {}}; "
        "cli._fetch_rounds = lambda: {1: dt.date(2027, 7, 14)}; "
        "r = CliRunner().invoke(cli.cli, ['fetch-contest-streak', '--picks-dir', sys.argv[2]]); "
        "print(r.exit_code)"
    )
    ed_base = subprocess.check_output([sys.executable, "-c", empty_dict_snip, str(HERE / "base_tree/src"), str(HERE / "_ed_base/picks")], text=True).strip()
    ed_branch = subprocess.check_output([sys.executable, "-c", empty_dict_snip, str(ROOT / "src"), str(HERE / "_ed_branch/picks")], text=True).strip()
    import shutil as _sh
    _sh.rmtree(HERE / "_ed_base", ignore_errors=True)
    _sh.rmtree(HERE / "_ed_branch", ignore_errors=True)
    assert ed_base == "0" and ed_branch == "2", (ed_base, ed_branch)

    # Verify write-failure prefixes on branch!
    pf = branch["prefixes"]
    assert pf["automatic|obs"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": False, "ledger_len": None, "saver_exists": False}
    assert pf["automatic|status"] == {"exit": 1, "dms": 0, "auto_changed": True, "status_exists": False, "ledger_len": None, "saver_exists": False}
    assert pf["automatic|ledger_open"] == {"exit": 1, "dms": 0, "auto_changed": True, "status_exists": True, "ledger_len": None, "saver_exists": False}
    assert pf["automatic|ledger_partial"] == {"exit": 1, "dms": 0, "auto_changed": True, "status_exists": True, "ledger_len": 12, "saver_exists": False}
    assert pf["automatic|saver_lock"]["exit"] == 1 and pf["automatic|saver_lock"]["dms"] == 0 and pf["automatic|saver_lock"]["auto_changed"] and pf["automatic|saver_lock"]["status_exists"] and pf["automatic|saver_lock"]["ledger_len"] > 50 and not pf["automatic|saver_lock"]["saver_exists"]
    assert pf["automatic|saver_state"]["exit"] == 1 and pf["automatic|saver_state"]["dms"] == 0 and pf["automatic|saver_state"]["auto_changed"] and not pf["automatic|saver_state"]["saver_exists"]
    assert pf["automatic|saver_audit"]["exit"] == 0 and pf["automatic|saver_audit"]["dms"] == 0 and pf["automatic|saver_audit"]["auto_changed"] and pf["automatic|saver_audit"]["saver_exists"]
    for sw in ["absent", "other_season", "revoked"]:
        assert pf[f"{sw}|obs"]["exit"] == 0 and not pf[f"{sw}|obs"]["auto_changed"] and pf[f"{sw}|obs"]["status_exists"] and pf[f"{sw}|obs"]["ledger_len"] > 50
        assert pf[f"{sw}|status"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": False, "ledger_len": None, "saver_exists": False}
        assert pf[f"{sw}|ledger_open"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": True, "ledger_len": None, "saver_exists": False}
        assert pf[f"{sw}|ledger_partial"] == {"exit": 1, "dms": 0, "auto_changed": False, "status_exists": True, "ledger_len": 12, "saver_exists": False}
        for s_stage in ["saver_lock", "saver_state", "saver_audit"]:
            assert pf[f"{sw}|{s_stage}"]["exit"] == 0 and not pf[f"{sw}|{s_stage}"]["auto_changed"] and not pf[f"{sw}|{s_stage}"]["saver_exists"]
    assert pf["c8_note|status"] == {"exit": 1, "dms": 1, "auto_changed": False, "status_exists": False, "ledger_len": None, "saver_exists": False}
    assert pf["c8_note|obs"] == {"exit": 2, "dms": 1, "auto_changed": False, "status_exists": True, "ledger_len": None, "saver_exists": False}
    assert pf["c8_note|ledger_open"] == {"exit": 2, "dms": 1, "auto_changed": False, "status_exists": True, "ledger_len": None, "saver_exists": False}

    # Verify clock boundary cases on branch vs base
    for ckey, crec in branch["clocks"].items():
        adv, src_part = ckey.split("|")
        assert crec["exit"] == 0, f"clock exit failed {ckey}: {crec}"
        assert crec["auto"]["season"] == 2027
        assert crec["ledger"]["season"] == 2027
        assert crec["auto"]["recorded_at"] == "2028-01-01T04:59:59.500000Z"
        assert crec["ledger"]["recorded_at"] == "2028-01-01T04:59:59.500000Z"
        if adv in {"auth", "between_profile_rounds", "obs_write"}:
            assert crec["status"]["last_success_at"] == "2028-01-01T05:00:00.500000Z"
        if src_part == "src=2027":
            assert crec["earn_seasons"] == [2027], f"expected earn(2027) on {ckey}: {crec}"
        else:
            assert crec["earn_seasons"] == [], f"expected no earn on {ckey}: {crec}"

    assert base["clocks"]["auth|src=2028"]["earn_seasons"] == [2028]
    assert base["clocks"]["auth|src=2027"]["earn_seasons"] == []

    summary = {
        "total_matrix_cases_per_tree": branch["count"],
        "total_runs": branch["count"] * 2,
        "io_violations": branch["io_violations"],
        "unexpected_diffs": unexpected,
        "buckets": diff_buckets,
        "empty_dict_predictions": {"base_exit": int(ed_base), "branch_exit": int(ed_branch)},
        "prefix_cases_verified": len(pf),
        "pairwise_cases_verified": len(np),
        "clock_cases_verified": len(branch["clocks"]),
    }
    (HERE / "probe1b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert not unexpected, f"UNEXPECTED DIFFS ({len(unexpected)}): {unexpected[:10]}"
    print("PROBE 1C SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
