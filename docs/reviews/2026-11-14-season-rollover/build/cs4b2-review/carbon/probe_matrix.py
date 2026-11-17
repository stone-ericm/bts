"""Adversarial base-vs-branch matrix probe for CS4b-2 (Items 1, 2, 3, 4).

Installs a sys.addaudithook forbidding all DNS/network connect events and
recording every open/stat on season_contract.json and observation files.
Compares 0bbfc8d4 (archived base src) against HEAD (branch src) across the
full cross-product of switch x manual x automatic x streak.json x saver_state.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

AUDIT_STATE = {
    "tracking": False,
    "denied_network": [],
    "switch_opens": 0,
    "switch_stats": 0,
    "obs_opens": [],
}

_NET_EVENTS = frozenset({
    "socket.getaddrinfo",
    "socket.getnameinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.connect",
    "socket.connect_ex",
    "socket.sendto",
})


def _audit_hook(event: str, args: tuple) -> None:
    if event in _NET_EVENTS:
        AUDIT_STATE["denied_network"].append((event, repr(args)))
        raise RuntimeError(f"audit hook blocked network event: {event}")
    if not AUDIT_STATE["tracking"]:
        return
    if event == "open" and args:
        p = str(args[0])
        if p.endswith("season_contract.json"):
            AUDIT_STATE["switch_opens"] += 1
        elif p.endswith("contest_streak.json") or p.endswith("contest_streak.manual.json"):
            AUDIT_STATE["obs_opens"].append(p)
    elif event == "os.stat" and args:
        p = str(args[0])
        if p.endswith("season_contract.json"):
            AUDIT_STATE["switch_stats"] += 1


sys.addaudithook(_audit_hook)

import argparse
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)
ID = "a" * 32


def _reset_audit() -> None:
    AUDIT_STATE["switch_opens"] = 0
    AUDIT_STATE["switch_stats"] = 0
    AUDIT_STATE["obs_opens"].clear()


def _serialize_decision(fn):
    try:
        res = fn()
        d = asdict(res)
        if d.get("contest_source_date") is not None:
            d["contest_source_date"] = d["contest_source_date"].isoformat()
        return {"ok": True, "val": d}
    except Exception as exc:
        return {"ok": False, "err": type(exc).__name__, "msg": str(exc)}


def _serialize_contest(fn):
    try:
        res = fn()
        if res is None:
            return {"ok": True, "val": None}
        d = asdict(res)
        d["path"] = res.path.name
        for k in ("source_date", "effective_date"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        if d.get("override_expires_at") is not None:
            d["override_expires_at"] = d["override_expires_at"].isoformat()
        return {"ok": True, "val": d}
    except Exception as exc:
        return {"ok": False, "err": type(exc).__name__, "msg": str(exc)}


def _serialize_health(fn):
    try:
        alerts = fn()
        return {
            "ok": True,
            "val": [{"level": a.level, "source": a.source, "message": a.message} for a in alerts],
        }
    except Exception as exc:
        return {"ok": False, "err": type(exc).__name__, "msg": str(exc)}


def _write_entry(path: Path, spec) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.unlink()
    if spec is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec == "__DIR__":
        path.mkdir()
    elif isinstance(spec, bytes):
        path.write_bytes(spec)
    else:
        path.write_text(json.dumps(spec), encoding="utf-8")


def build_fixtures():
    def sw_base(**kw):
        return {
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "automatic",
            "activation_id": ID,
            "switched_at": "2027-07-15T16:00:00Z",
            "revoked_at": None,
            "note": None,
        } | kw

    switches = {
        "absent": None,
        "corrupt1_non_obj": [],
        "corrupt2_schema": sw_base(schema_version="wrong"),
        "corrupt3_bool_season": sw_base(season=True),
        "corrupt4_status": sw_base(status="paused"),
        "corrupt5_bad_id": sw_base(activation_id="A" * 32),
        "corrupt6_null_switched": sw_base(switched_at=None),
        "corrupt7_bad_revoked": sw_base(revoked_at="2027-07-15T12:00:00"),
        "corrupt8_bad_note": sw_base(note=["bad"]),
        "other_season": sw_base(season=2026),
        "revoked": sw_base(status="revoked", revoked_at="2027-07-15T16:00:00Z"),
        "automatic": sw_base(),
    }

    def dated_v3(day: str, **kw):
        d = dt.date.fromisoformat(day)
        end = dt.datetime.combine(d, dt.time.max, ET).isoformat()
        return {
            "schema_version": "bts_contest_streak_manual_v3",
            "effective_date": day,
            "season": d.year,
            "expires_at": end,
            "active_streak": 11,
            "best_streak": 14,
            "saver_state": "not_earned",
            "recorded_at": "2027-07-15T15:00:00Z",
            "source": "manual_cli",
        } | kw

    manuals = {
        "absent": None,
        "legacy_unexpired": {
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 5,
            "best_streak": 9,
            "source": "manual_cli",
            "source_date": "2027-07-15",
            "override_expires_at": "2027-07-16T16:00:00Z",
        },
        "legacy_expired": {
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 5,
            "best_streak": 9,
            "source": "manual_cli",
            "source_date": "2027-07-13",
            "override_expires_at": "2027-07-14T16:00:00Z",
        },
        "legacy_no_expiry": {
            "active_streak": 5,
            "best_streak": 9,
            "source": "manual_cli",
            "source_date": "2027-07-14",
        },
        "manual_disguised_auto_v2": {
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": ID,
            "active_streak": 6,
            "best_streak": 9,
            "source": "mlb_bts_profile",
            "source_date": "2027-07-15",
            "recorded_at": "2027-07-15T15:00:00Z",
            "override_expires_at": "2027-07-16T16:00:00Z",
        },
        "manual_v2_with_effective_date": {
            "schema_version": "bts_contest_streak_manual_v2",
            "effective_date": "2027-07-15",
            "season": 2027,
            "active_streak": 6,
            "best_streak": 9,
            "saver_state": "active",
            "source": "manual_cli",
            "override_expires_at": "2027-07-16T16:00:00Z",
        },
        "dated_today": dated_v3("2027-07-15"),
        "dated_yesterday": dated_v3("2027-07-14", override_expires_at="2099-01-01T00:00:00Z"),
        "dated_tomorrow": dated_v3("2027-07-16", override_expires_at="2099-01-01T00:00:00Z"),
        "invalid_dated": dated_v3("2027-07-15", saver_state="bogus"),
        "invalid_legacy": {"active_streak": True},
        "unreadable": b"\xff\xfe",
    }

    def auto_base(**kw):
        return {
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": ID,
            "active_streak": 12,
            "best_streak": 15,
            "source": "mlb_bts_profile",
            "source_date": "2027-07-14",
            "recorded_at": "2027-07-15T14:00:00Z",
        } | kw

    missing_schema = auto_base()
    del missing_schema["schema_version"]

    autos = {
        "absent": None,
        "auto_v1": auto_base(schema_version="bts_contest_streak_auto_v1"),
        "schema_missing": missing_schema,
        "schema_null": auto_base(schema_version=None),
        "schema_int": auto_base(schema_version=17),
        "schema_unknown": auto_base(schema_version="bts_contest_streak_auto_v9"),
        "auto_disguised_dated_v3": dated_v3("2027-07-15"),
        "auto_v2_covered": auto_base(),
        "auto_v2_other_id": auto_base(activation_id="b" * 32),
        "auto_v2_other_season": auto_base(season=2026),
        "auto_v2_other_season_and_id": auto_base(season=2026, activation_id="b" * 32),
        "auto_v2_bad_season_type": auto_base(season=True),
        "auto_v2_bad_id_type": auto_base(activation_id="Z" * 32),
        "refused": auto_base(active_streak=58, best_streak=58),
        "corrupt": b"\xff\xfe",
        "undated": auto_base(source_date=None, recorded_at=None),
        "undated_bad_v2_season": auto_base(season=True, source_date=None, recorded_at=None),
        "recorded_at_only": auto_base(source_date=None),
    }

    models = {
        "present": {"season": 2027, "streak": 4, "saver_available": True},
        "absent": None,
        "corrupt": b"\xff\xfe",
    }

    savers = {
        "uninitialized": None,
        "not_earned": {"season": 2027, "state": "not_earned"},
        "active": {"season": 2027, "state": "active"},
        "used": {"season": 2027, "state": "used"},
    }

    return switches, manuals, autos, models, savers


def run_worker(src_dir: Path, out_path: Path, is_branch: bool) -> None:
    sys.path.insert(0, str(src_dir.resolve()))
    import bts.contest_state as cs
    import bts.health.contest_state as hc

    switches, manuals, autos, models, savers = build_fixtures()
    results = {}
    audit_violations = []

    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        state_dir = picks / "account_state"
        state_dir.mkdir()
        # One settled pick at 2027-07-15 so source_date=2027-07-14 exercises lagged/INFO coverage
        # while dated_today (2027-07-15) exercises fresh!
        (picks / "2027-07-15.json").write_text('{"result": "hit"}', encoding="utf-8")

        sw_path = state_dir / "season_contract.json"
        man_path = state_dir / "contest_streak.manual.json"
        auto_path = state_dir / "contest_streak.json"
        mod_path = picks / "streak.json"
        sav_path = state_dir / "saver_state.json"

        for mod_k, mod_v in models.items():
            _write_entry(mod_path, mod_v)
            for sav_k, sav_v in savers.items():
                _write_entry(sav_path, sav_v)
                for sw_k, sw_v in switches.items():
                    _write_entry(sw_path, sw_v)
                    for man_k, man_v in manuals.items():
                        _write_entry(man_path, man_v)
                        for auto_k, auto_v in autos.items():
                            _write_entry(auto_path, auto_v)
                            key = f"{sw_k}|{man_k}|{auto_k}|{mod_k}|{sav_k}"

                            # 1. Optional decision (require_contest_state=False)
                            _reset_audit()
                            AUDIT_STATE["tracking"] = True
                            opt_dec = _serialize_decision(
                                lambda: cs.load_decision_streak_state(
                                    picks, require_contest_state=False, now=NOW
                                )
                            )
                            AUDIT_STATE["tracking"] = False
                            if AUDIT_STATE["switch_opens"] or AUDIT_STATE["switch_stats"]:
                                audit_violations.append(f"opt_dec_touched_switch:{key}")

                            # 2. Health expected=False
                            _reset_audit()
                            AUDIT_STATE["tracking"] = True
                            opt_hlth = _serialize_health(
                                lambda: hc.check(picks, expected=False, now=NOW)
                            )
                            AUDIT_STATE["tracking"] = False
                            if AUDIT_STATE["switch_opens"] or AUDIT_STATE["switch_stats"]:
                                audit_violations.append(f"opt_hlth_touched_switch:{key}")

                            # 3. Direct load_contest_streak_state (both require_dated_auto)
                            _reset_audit()
                            AUDIT_STATE["tracking"] = True
                            sel_opt = _serialize_contest(
                                lambda: cs.load_contest_streak_state(
                                    picks, now=NOW, require_dated_auto=False
                                )
                            )
                            sel_req = _serialize_contest(
                                lambda: cs.load_contest_streak_state(
                                    picks, now=NOW, require_dated_auto=True
                                )
                            )
                            AUDIT_STATE["tracking"] = False
                            if AUDIT_STATE["switch_opens"] or AUDIT_STATE["switch_stats"]:
                                audit_violations.append(f"loader_touched_switch:{key}")

                            # 4. Required decision (require_contest_state=True)
                            _reset_audit()
                            AUDIT_STATE["tracking"] = True
                            req_dec = _serialize_decision(
                                lambda: cs.load_decision_streak_state(
                                    picks, require_contest_state=True, now=NOW
                                )
                            )
                            AUDIT_STATE["tracking"] = False
                            req_dec_sw_opens = AUDIT_STATE["switch_opens"]
                            req_dec_sw_stats = AUDIT_STATE["switch_stats"]
                            if man_k == "dated_today" and (req_dec_sw_opens or req_dec_sw_stats):
                                audit_violations.append(f"dated_dec_touched_switch:{key}")
                            if is_branch and req_dec_sw_opens > 1:
                                audit_violations.append(f"req_dec_multi_switch_open:{key}:{req_dec_sw_opens}")

                            # 5. Health expected=True
                            _reset_audit()
                            AUDIT_STATE["tracking"] = True
                            req_hlth = _serialize_health(
                                lambda: hc.check(picks, expected=True, now=NOW)
                            )
                            AUDIT_STATE["tracking"] = False
                            req_hlth_sw_opens = AUDIT_STATE["switch_opens"]
                            req_hlth_sw_stats = AUDIT_STATE["switch_stats"]
                            if man_k == "dated_today" and (req_hlth_sw_opens or req_hlth_sw_stats):
                                audit_violations.append(f"dated_hlth_touched_switch:{key}")
                            if is_branch and req_hlth_sw_opens > 1:
                                audit_violations.append(f"req_hlth_multi_switch_open:{key}:{req_hlth_sw_opens}")

                            results[key] = (
                                opt_dec,
                                opt_hlth,
                                sel_opt,
                                sel_req,
                                req_dec,
                                req_hlth,
                            )

        # Secondary slice: freshness/best_status & health disagreements on branch/base
        extra = {}
        if is_branch:
            import bts.season_contract as sc

            # Check live_automatic_refusal never opens observation files directly
            _write_entry(sw_path, switches["automatic"])
            _write_entry(man_path, None)
            _write_entry(auto_path, autos["auto_v1"])
            parsed = cs._parse_state_file(auto_path)
            _reset_audit()
            AUDIT_STATE["tracking"] = True
            ref_msg = sc.live_automatic_refusal(picks, NOW, parsed)
            AUDIT_STATE["tracking"] = False
            extra["direct_refusal_obs_opens"] = list(AUDIT_STATE["obs_opens"])
            extra["direct_refusal_switch_opens"] = AUDIT_STATE["switch_opens"]
            extra["direct_refusal_msg_present"] = ref_msg is not None

            # Post-loader exception (overlong saver int) on covered vs uncovered auto
            big_saver = '{"n":' + ("9" * 5000) + "}"
            sav_path.write_text(big_saver, encoding="utf-8")
            for label, auto_spec in [("covered", autos["auto_v2_covered"]), ("uncovered", autos["auto_v1"])]:
                _write_entry(auto_path, auto_spec)
                extra[f"overlong_saver_{label}_dec"] = _serialize_decision(
                    lambda: cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                )
                extra[f"overlong_saver_{label}_hlth"] = _serialize_health(
                    lambda: hc.check(picks, expected=True, now=NOW)
                )

            # Future source_date on covered auto: live accepts (best untrusted), health CRITICAL
            _write_entry(sav_path, savers["active"])
            _write_entry(auto_path, autos["auto_v2_covered"] | {"source_date": "2027-07-20"})
            extra["future_source_covered_dec"] = _serialize_decision(
                lambda: cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
            )
            extra["future_source_covered_hlth"] = _serialize_health(
                lambda: hc.check(picks, expected=True, now=NOW)
            )

        # Fresh / stale (>=2 gap and unconfirmed miss) parity slice on accepted cells
        _write_entry(sav_path, savers["active"])
        _write_entry(sw_path, switches["automatic"])
        _write_entry(man_path, None)
        _write_entry(auto_path, autos["auto_v2_covered"])
        (picks / "2027-07-15.json").unlink()
        extra["fresh_no_picks_dec"] = _serialize_decision(
            lambda: cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
        )
        (picks / "2027-07-15.json").write_text('{"result": "hit"}', encoding="utf-8")
        (picks / "2027-07-16.json").write_text('{"result": "hit"}', encoding="utf-8")
        extra["stale_gap2_dec"] = _serialize_decision(
            lambda: cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
        )
        extra["stale_gap2_hlth"] = _serialize_health(
            lambda: hc.check(picks, expected=True, now=NOW)
        )
        (picks / "2027-07-16.json").unlink()
        (picks / "2027-07-15.json").write_text('{"result": "miss"}', encoding="utf-8")
        extra["stale_miss_dec"] = _serialize_decision(
            lambda: cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
        )

    payload = {
        "cell_count": len(results),
        "denied_network": AUDIT_STATE["denied_network"],
        "audit_violations": audit_violations,
        "extra": extra,
        "results": results,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def expected_refusal_tag(sw_k: str, man_k: str, auto_k: str, sel_req: dict) -> str | None:
    """Independent contract oracle for Row 1 / Row 3 when loader succeeded and not dated_today."""
    assert sel_req["ok"]
    sel = sel_req["val"]
    if man_k == "dated_today":
        return None
    if sw_k.startswith("corrupt"):
        check_num = sw_k.split("_", 1)[0].replace("corrupt", "")
        return f"corrupt_check_{check_num}"
    if sel is not None and sel["path"] == "contest_streak.manual.json":
        return "legacy_manual"
    if sw_k == "absent":
        return "switch_absent"
    if sw_k == "other_season":
        return "switch_other_season"
    if sw_k == "revoked":
        return "switch_revoked"
    if sel is None:
        return "no_auto"
    # Automatic file selected under automatic switch for 2027:
    if auto_k in ("auto_v2_covered", "recorded_at_only"):
        return None
    if auto_k in ("auto_v1", "schema_missing", "schema_null", "schema_int", "schema_unknown", "auto_disguised_dated_v3"):
        return f"bad_schema_{auto_k}"
    if auto_k in ("auto_v2_other_season", "auto_v2_other_season_and_id"):
        return "bad_season"
    if auto_k == "auto_v2_other_id":
        return "bad_id"
    raise AssertionError(f"unexpected cell reached oracle: {sw_k} {man_k} {auto_k}")


def check_tag_matches_message(tag: str, msg: str) -> bool:
    first = msg.splitlines()[0]
    if tag.startswith("corrupt_check_"):
        n = tag.split("_")[-1]
        return f"corrupt season switch, check {n}:" in first
    if tag == "legacy_manual":
        return "selected legacy manual override is no longer accepted" in first
    if tag == "switch_absent":
        return "season 2027 switch is absent." in first
    if tag == "switch_other_season":
        return "season 2027 switch is other_season (file season 2026)." in first
    if tag == "switch_revoked":
        return "season 2027 switch is revoked." in first
    if tag == "no_auto":
        return "there is no automatic observation for this activation." in first
    if tag == "bad_schema_auto_v1":
        return "schema_version is 'bts_contest_streak_auto_v1', not 'bts_contest_streak_auto_v2'" in first
    if tag == "bad_schema_schema_unknown":
        return "schema_version is 'bts_contest_streak_auto_v9', not 'bts_contest_streak_auto_v2'" in first
    if tag == "bad_schema_auto_disguised_dated_v3":
        return "schema_version is 'bts_contest_streak_manual_v3', not 'bts_contest_streak_auto_v2'" in first
    if tag in ("bad_schema_schema_missing", "bad_schema_schema_null", "bad_schema_schema_int"):
        return "schema_version is missing, null or not a string; look at status for the exact value" in first
    if tag == "bad_season":
        return "automatic observation season is 2026, not 2027" in first
    if tag == "bad_id":
        return f"automatic observation activation_id is {'b' * 32!r}, not the switch's {ID!r}" in first
    return False


def run_compare(repo_root: Path, work_dir: Path) -> None:
    base_json = work_dir / "matrix_base.json"
    branch_json = work_dir / "matrix_branch.json"
    script = Path(__file__).resolve()

    for src, out, flag in [
        (work_dir / "base/src", base_json, []),
        (repo_root / "src", branch_json, ["--branch"]),
    ]:
        proc = subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out), *flag],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"worker failed ({src}):\n{proc.stdout}\n{proc.stderr}")

    base = json.loads(base_json.read_text(encoding="utf-8"))
    branch = json.loads(branch_json.read_text(encoding="utf-8"))

    assert base["denied_network"] == []
    assert branch["denied_network"] == []
    assert base["audit_violations"] == [], base["audit_violations"][:10]
    assert branch["audit_violations"] == [], branch["audit_violations"][:10]
    assert branch["extra"]["direct_refusal_obs_opens"] == []
    assert branch["extra"]["direct_refusal_switch_opens"] == 1
    assert branch["extra"]["direct_refusal_msg_present"] is True

    counts = {
        "total_cells": branch["cell_count"],
        "opt_dec_diff": 0,
        "opt_hlth_diff": 0,
        "sel_opt_diff": 0,
        "sel_req_diff": 0,
        "req_dec_identical": 0,
        "req_dec_changed": 0,
        "req_dec_accepted_branch": 0,
        "req_dec_loader_err_branch": 0,
        "req_dec_refused_branch": 0,
        "req_hlth_identical": 0,
        "req_hlth_changed": 0,
        "req_hlth_undated_auto_changed": 0,
        "req_hlth_refusal_appended": 0,
        "uncovered_or_legacy_accepted_live": 0,
        "unexplained_diffs": [],
    }

    for key, br_tuple in branch["results"].items():
        ba_tuple = base["results"][key]
        sw_k, man_k, auto_k, mod_k, sav_k = key.split("|")
        ba_opt_d, ba_opt_h, ba_sel_o, ba_sel_r, ba_req_d, ba_req_h = ba_tuple
        br_opt_d, br_opt_h, br_sel_o, br_sel_r, br_req_d, br_req_h = br_tuple

        if br_opt_d != ba_opt_d:
            counts["opt_dec_diff"] += 1
            counts["unexplained_diffs"].append(("opt_dec", key, ba_opt_d, br_opt_d))
        if br_opt_h != ba_opt_h:
            counts["opt_hlth_diff"] += 1
            counts["unexplained_diffs"].append(("opt_hlth", key, ba_opt_h, br_opt_h))
        if br_sel_o != ba_sel_o:
            counts["sel_opt_diff"] += 1
            counts["unexplained_diffs"].append(("sel_opt", key, ba_sel_o, br_sel_o))
        if br_sel_r != ba_sel_r:
            counts["sel_req_diff"] += 1
            counts["unexplained_diffs"].append(("sel_req", key, ba_sel_r, br_sel_r))

        if br_req_d == ba_req_d:
            counts["req_dec_identical"] += 1
        else:
            counts["req_dec_changed"] += 1

        if br_req_h == ba_req_h:
            counts["req_hlth_identical"] += 1
        else:
            counts["req_hlth_changed"] += 1

        # Loader error case for require_dated_auto=True:
        if not br_sel_r["ok"]:
            counts["req_dec_loader_err_branch"] += 1
            if br_req_d != ba_req_d:
                counts["unexplained_diffs"].append(("loader_err_dec_changed", key, ba_req_d, br_req_d))
            expected_hlth = [{"level": "CRITICAL", "source": "contest_state", "message": br_sel_r["msg"]}]
            if br_req_h != {"ok": True, "val": expected_hlth}:
                counts["unexplained_diffs"].append(("loader_err_hlth_wrong", key, br_req_h))
            if br_req_h != ba_req_h:
                # Must ONLY be when auto_k is undated or undated_bad_v2_season!
                if auto_k not in ("undated", "undated_bad_v2_season"):
                    counts["unexplained_diffs"].append(("unexpected_loader_err_hlth_diff", key, ba_req_h, br_req_h))
                else:
                    counts["req_hlth_undated_auto_changed"] += 1
            continue

        tag = expected_refusal_tag(sw_k, man_k, auto_k, br_sel_r)
        if tag is None:
            # Must be accepted and byte-for-byte identical to base!
            counts["req_dec_accepted_branch"] += 1
            if not br_req_d["ok"] or br_req_d != ba_req_d:
                counts["unexplained_diffs"].append(("accepted_dec_diff", key, ba_req_d, br_req_d))
            if br_req_h != ba_req_h:
                counts["unexplained_diffs"].append(("accepted_hlth_diff", key, ba_req_h, br_req_h))
            # Verify this is genuinely dated_today or covered auto_v2!
            if man_k != "dated_today" and not (
                sw_k == "automatic" and auto_k in ("auto_v2_covered", "recorded_at_only")
            ):
                counts["uncovered_or_legacy_accepted_live"] += 1
        else:
            counts["req_dec_refused_branch"] += 1
            if br_req_d["ok"]:
                counts["uncovered_or_legacy_accepted_live"] += 1
                counts["unexplained_diffs"].append(("should_have_refused", key, tag, br_req_d))
            else:
                if br_req_d["err"] != "ContestStateError" or not check_tag_matches_message(tag, br_req_d["msg"]):
                    counts["unexplained_diffs"].append(("wrong_refusal_msg", key, tag, br_req_d))
                # Health check must equal ba_req_h + [CRITICAL contest_state refusal]
                refusal_alert = {"level": "CRITICAL", "source": "contest_state", "message": br_req_d["msg"]}
                expected_alerts = ba_req_h["val"] + [refusal_alert]
                if br_req_h != {"ok": True, "val": expected_alerts}:
                    counts["unexplained_diffs"].append(("wrong_hlth_refusal", key, expected_alerts, br_req_h))
                else:
                    counts["req_hlth_refusal_appended"] += 1
                # Also verify refusal appears exactly once and last:
                occurrences = sum(1 for a in br_req_h["val"] if a["message"] == br_req_d["msg"])
                if occurrences != 1 or br_req_h["val"][-1] != refusal_alert:
                    counts["unexplained_diffs"].append(("refusal_not_once_and_last", key, br_req_h))

    # Secondary parity assertions on fresh / stale>=2 / miss
    for k in ("fresh_no_picks_dec", "stale_gap2_dec", "stale_gap2_hlth", "stale_miss_dec"):
        if branch["extra"][k] != base["extra"][k]:
            counts["unexplained_diffs"].append(("extra_parity", k, base["extra"][k], branch["extra"][k]))

    summary = {
        "counts": counts,
        "branch_extra": branch["extra"],
    }
    report_path = work_dir / "matrix_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Clean up large raw JSONs only? No, keep them as evidence!
    print(json.dumps(counts, indent=2))
    assert counts["opt_dec_diff"] == 0
    assert counts["opt_hlth_diff"] == 0
    assert counts["sel_opt_diff"] == 0
    assert counts["sel_req_diff"] == 0
    assert counts["uncovered_or_legacy_accepted_live"] == 0
    assert counts["unexplained_diffs"] == [], counts["unexplained_diffs"][:5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--branch", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    work = Path(__file__).resolve().parent
    if args.compare:
        run_compare(repo, work)
    else:
        run_worker(args.src, args.out, args.branch)
