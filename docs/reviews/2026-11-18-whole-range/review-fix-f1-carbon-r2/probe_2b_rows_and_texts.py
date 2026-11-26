"""Items 2, 5, and 12: Contract rows N1-N13, M1-M13, exact R64 texts, and R2 edits/limits."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from shlex import quote
import subprocess
import sys
import tempfile

from probe_common import (
    DAY,
    HERE,
    PITCH,
    SandboxWorld,
    UNIT_RIGHT_1,
    UNIT_RIGHT_2,
    UNIT_WRONG_1,
    UNIT_WRONG_2,
    bts_source_path,
    make_row,
)

BASE_SRC = HERE / "base_archive" / "src"

TPL_FIRST = (
    "⚠️ BTS entry game needs verification. {legs}. "
    "Check the delivered pick and the game shown in MLB for each listed leg. "
    "Change only a leg whose correct game you can verify and that MLB still lets you edit. "
    "Keep any correct leg as entered."
)
TPL_REMINDER = (
    "⏰ BTS entry game verification is still needed. {legs}. "
    "Check the delivered pick against the game shown in MLB."
)
TPL_LOCKED = (
    "⚠️ Delivery of this pick was NOT confirmed. The scheduler tried to send {names}. "
    "MLB game evidence needs verification. {legs}. "
    "Check your DMs and the MLB entry before making any change. "
    "Run `bts season-contract status --picks-dir {qdir}`."
)
ALL_CLEAR_MATCH = f"✅ BTS pick entry confirmed for {DAY} (match)."
ALL_CLEAR_UNCHECKED = f"✅ BTS pick entry: the batter is entered for {DAY}; the game could not be checked."


def run_rows_and_texts():
    checks_passed = []
    with tempfile.TemporaryDirectory(prefix="f1_rows_") as tmp:
        root = Path(tmp)

        # N1: no_pick ladder + cutoff
        w = SandboxWorld(root / "n1")
        w.commit()
        w.capture([UNIT_WRONG_1])
        r0 = w.check("18:30", [])
        assert r0["exit"] == 1 and r0["marker"]["reason"] == "no_pick" and r0["marker"]["escalations"] == ["initial"]
        assert "game_unresolved" not in r0["marker"]
        assert r0["dm"] == ["⚠️ BTS pick NOT entered in MLB app: Batter 1010 — first pitch 7:40 PM ET (65 min to submit). Fix it now!"]
        r_mid = w.check("18:45", [])
        assert r_mid["dm"] == [] and r_mid["stdout"] == "check-pick-entered: still not entered (no_pick); re-verifying each run until cutoff\n"
        r_t30 = w.check("19:05", [])
        assert r_t30["dm"] == ["⏰ STILL NOT entered (no_pick): Batter 1010 — 30 min left to submit!"]
        r_cut = w.check("19:34", [], elapsed=61)
        assert r_cut["dm"] == [] and r_cut["stdout"] == "check-pick-entered: submission cutoff passed during fetch; skipping the nag (entry can no longer change)\n"
        checks_passed.append("N1")

        # N2: mismatch (wrong batter and missing DD leg)
        w = SandboxWorld(root / "n2")
        w.commit(dd=(1020, 201, PITCH))
        w.capture([UNIT_RIGHT_1, UNIT_RIGHT_2])
        r_wb = w.check("18:30", [make_row(77, 3, 1), make_row(78, 2, 2)])
        assert r_wb["exit"] == 1 and r_wb["marker"]["reason"] == "mismatch" and "game_unresolved" not in r_wb["marker"]
        w2 = SandboxWorld(root / "n2b")
        w2.commit(dd=(1020, 201, PITCH))
        r_miss_dd = w2.check("18:30", [make_row(77, 1, 1)])
        assert r_miss_dd["exit"] == 1 and r_miss_dd["marker"]["reason"] == "mismatch"
        checks_passed.append("N2")

        # N3: right batter in right game -> confirmed, all-clear after alerted, later already confirmed
        w = SandboxWorld(root / "n3")
        w.commit()
        w.capture([UNIT_RIGHT_1])
        w.check("18:30", [])
        r_ok = w.check("18:45", [make_row(77, 1, 1)])
        assert r_ok["exit"] == 0 and r_ok["marker"]["status"] == "confirmed" and r_ok["dm"] == [ALL_CLEAR_MATCH]
        assert "escalations" not in r_ok["marker"] and "game_unresolved" not in r_ok["marker"]
        r_later = w.check("19:00", [make_row(77, 1, 1)])
        assert r_later["exit"] == 0 and r_later["fetches"] == [] and r_later["stdout"] == f"check-pick-entered: already confirmed for {DAY}\n"
        checks_passed.append("N3")

        # N4: no capture, absent unit, null feedId, wrong_round, ambiguous -> confirmed on clean day
        for tag, cap_fn, rows_in, exp_morn in (
            ("no_cap", lambda ww: None, [make_row(77, 1, 1)], "entered_game_unverified"),
            ("absent_unit", lambda ww: ww.capture([{"id": 99, "roundId": 1, "feedId": 101}]), [make_row(77, 1, 1)], "entered_game_unverified"),
            ("null_feed", lambda ww: ww.capture([{"id": 77, "roundId": 1, "feedId": None}]), [make_row(77, 1, 1)], "entered_game_unverified"),
            ("wrong_round", lambda ww: ww.capture([{"id": 77, "roundId": 2, "feedId": 102}]), [make_row(77, 1, 1)], "entry_unverifiable"),
            ("ambiguous", lambda ww: ww.capture([UNIT_WRONG_1, {"id": 78, "roundId": 1, "feedId": 103}]), [make_row(77, 1, 1), make_row(78, 1, 2)], "entry_unverifiable"),
        ):
            w = SandboxWorld(root / f"n4_{tag}")
            w.commit()
            cap_fn(w)
            res = w.check("18:45", rows_in)
            assert res["exit"] == 0 and res["marker"]["status"] == "confirmed" and res["marker"]["reason"] == "match"
            assert w.morning()["stdout"]["state"] == exp_morn
        checks_passed.append("N4")

        # N5: present_unverified
        w = SandboxWorld(root / "n5")
        w.commit()
        w.capture([UNIT_WRONG_1])
        r_pu = w.check("18:45", [make_row(77, 1, 1)], crosswalk={})
        assert r_pu["exit"] == 0 and r_pu["marker"]["status"] == "present_unverified" and "game_unresolved" not in r_pu["marker"]
        checks_passed.append("N5")

        # N6: fetch failures + identity mismatch
        for fail_kind in ("auth", "profile", "pending", "rounds", "crosswalk", "identity"):
            w = SandboxWorld(root / f"n6_{fail_kind}")
            w.commit()
            w.capture([UNIT_WRONG_1])
            kw = {"username": "impostor"} if fail_kind == "identity" else {"fetch_failure": fail_kind}
            r_f = w.check("18:45", [make_row(77, 1, 1)], **kw)
            assert r_f["exit"] == 0 and r_f["marker"] is None and r_f["dm"] == []
        checks_passed.append("N6")

        # N7: no pick file, not committed, outside window (76m > default 75m window, and exact 5m cutoff), already confirmed
        w = SandboxWorld(root / "n7_nofile")
        assert w.check("18:45")["fetches"] == []
        w = SandboxWorld(root / "n7_uncommitted")
        w.commit(delivery=None)
        assert w.check("18:45")["fetches"] == []
        w = SandboxWorld(root / "n7_win")
        w.commit()
        assert w.check("18:24")["stdout"] == "check-pick-entered: outside window (76 min to pitch)\n"
        assert w.check("18:25")["fetches"] != []  # exact 75m upper window boundary is inside
        assert w.check("19:35")["stdout"] == "check-pick-entered: outside window (5 min to pitch)\n"
        checks_passed.append("N7")

        # N8: locked_unconfirmed with no_pick or mismatch -> exact base text
        for rk, rows_in, clause in (
            ("no_pick", [], "No entry found in MLB"),
            ("mismatch", [make_row(77, 2, 1)], "The MLB entry differs"),
        ):
            w = SandboxWorld(root / f"n8_{rk}")
            w.commit(delivery="locked_unconfirmed")
            r_n8 = w.check("18:45", rows_in)
            expected_n8 = (
                f"⚠️ Delivery of this pick was NOT confirmed. Batter 1010 is the pick the scheduler "
                f"tried to send. {clause}. First pitch 7:40 PM ET (50 min to submit). Check your DMs and "
                f"MLB entry before entering anything, and run `bts season-contract status --picks-dir {quote(str(w.picks))}`."
            )
            assert r_n8["dm"] == [expected_n8]
        checks_passed.append("N8")

        # N9: double with both legs in right games
        w = SandboxWorld(root / "n9")
        w.commit(dd=(1020, 201, PITCH))
        w.capture([UNIT_RIGHT_1, UNIT_RIGHT_2])
        r_n9 = w.check("18:45", [make_row(77, 1, 1), make_row(78, 2, 2)])
        assert r_n9["exit"] == 0 and r_n9["marker"]["status"] == "confirmed"
        assert r_n9["receipts"][-1]["qualification"]["all_confirmed"] is True
        checks_passed.append("N9")

        # N12: non-integer game_pk & shared batter on clean day -> base confirmation
        for lbl, g_val in (("null", None), ("str", "101"), ("float", 101.0), ("bool_t", True), ("bool_f", False)):
            w = SandboxWorld(root / f"n12_{lbl}")
            w.commit(game=g_val)
            w.capture([UNIT_WRONG_1])
            r_12 = w.check("18:45", [make_row(77, 1, 1)])
            assert r_12["exit"] == 0 and r_12["marker"]["status"] == "confirmed"
        w = SandboxWorld(root / "n12_shared")
        w.commit(dd=(1010, 201, PITCH))
        w.capture([UNIT_WRONG_1, UNIT_WRONG_2])
        r_sh = w.check("18:45", [make_row(77, 1, 1), make_row(78, 1, 2)])
        assert r_sh["exit"] == 0 and r_sh["marker"]["status"] == "confirmed"
        checks_passed.append("N12")

        # M1: 18:30 missing -> 18:45 wrong game -> 19:00 re-verifies
        w = SandboxWorld(root / "m1")
        w.commit()
        w.capture([UNIT_WRONG_1])
        w.check("18:30", [])
        r_m1 = w.check("18:45", [make_row(77, 1, 1)])
        leg1 = "pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET), the entry is recorded in game 102"
        assert r_m1["exit"] == 1 and r_m1["dm"] == [TPL_FIRST.format(legs=leg1)]
        assert r_m1["marker"]["escalations"] == ["initial", "wrong_game"] and r_m1["marker"]["game_unresolved"] is True
        r_m1_re = w.check("19:00", [make_row(77, 1, 1)])
        assert r_m1_re["exit"] == 1 and len(r_m1_re["fetches"]) == 5 and r_m1_re["dm"] == []
        assert r_m1_re["stdout"] == f"check-pick-entered: {DAY} game verification unresolved (pick); re-verifying each run until cutoff\n"
        checks_passed.append("M1")

        # M2 + N10: first run wrong_game (escalations == ['wrong_game']), then fixed -> confirmed + all-clear
        w = SandboxWorld(root / "m2_n10")
        w.commit()
        w.capture([UNIT_WRONG_1])
        r_m2 = w.check("18:45", [make_row(77, 1, 1)])
        assert r_m2["marker"]["escalations"] == ["wrong_game"] and r_m2["marker"]["game_unresolved"] is True
        w.capture([UNIT_RIGHT_1])
        r_n10 = w.check("19:00", [make_row(77, 1, 1)])
        assert r_n10["exit"] == 0 and r_n10["marker"]["status"] == "confirmed"
        assert "game_unresolved" not in r_n10["marker"] and r_n10["dm"] == [ALL_CLEAR_MATCH]
        checks_passed.append("M2+N10")

        # M3: double — each single wrong leg in turn, plus both wrong legs
        for which in ("pick_wrong", "dd_wrong", "both_wrong"):
            w = SandboxWorld(root / f"m3_{which}")
            w.commit(dd=(1020, 201, f"{DAY}T20:10:00-04:00"))
            u1 = UNIT_WRONG_1 if which in ("pick_wrong", "both_wrong") else UNIT_RIGHT_1
            u2 = UNIT_WRONG_2 if which in ("dd_wrong", "both_wrong") else UNIT_RIGHT_2
            w.capture([u1, u2])
            res = w.check("18:45", [make_row(77, 1, 1), make_row(78, 2, 2)])
            l_p = "pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET), the entry is recorded in game 102"
            l_d = "double-down Batter 1020: recommended game 201 (first pitch 8:10 PM ET), the entry is recorded in game 202"
            exp_legs = l_p if which == "pick_wrong" else l_d if which == "dd_wrong" else f"{l_p}; {l_d}"
            exp_roles = "pick" if which == "pick_wrong" else "double-down" if which == "dd_wrong" else "pick + double-down"
            assert res["dm"] == [TPL_FIRST.format(legs=exp_legs)]
            res2 = w.check("18:50", [make_row(77, 1, 1), make_row(78, 2, 2)])
            assert res2["stdout"] == f"check-pick-entered: {DAY} game verification unresolved ({exp_roles}); re-verifying each run until cutoff\n"
        checks_passed.append("M3")

        # M4: t30 and t15 reminders fire once each
        w = SandboxWorld(root / "m4")
        w.commit()
        w.capture([UNIT_WRONG_1])
        w.check("18:45", [make_row(77, 1, 1)])
        assert w.check("19:05", [make_row(77, 1, 1)])["dm"] == [TPL_REMINDER.format(legs=leg1)]
        assert w.check("19:06", [make_row(77, 1, 1)])["dm"] == []
        assert w.check("19:20", [make_row(77, 1, 1)])["dm"] == [TPL_REMINDER.format(legs=leg1)]
        assert w.check("19:21", [make_row(77, 1, 1)])["dm"] == []
        checks_passed.append("M4")

        # M5: failed DM writes dm_failed (fresh) or alerted (already alerted), retries on next run
        for had_alert in (False, True):
            w = SandboxWorld(root / f"m5_{had_alert}")
            w.commit()
            w.capture([UNIT_WRONG_1])
            if had_alert:
                w.check("18:30", [])
            rf = w.check("18:45", [make_row(77, 1, 1)], dm="failure")
            assert rf["dm"] == [] and rf["attempted_dm"] == [TPL_FIRST.format(legs=leg1)]
            assert rf["marker"]["status"] == ("alerted" if had_alert else "dm_failed")
            assert rf["marker"]["game_unresolved"] is True
            rr = w.check("18:46", [make_row(77, 1, 1)])
            assert rr["dm"] == [TPL_FIRST.format(legs=leg1)]
        checks_passed.append("M5")

        # M6 + R2 edit 1 + N13: locked_unconfirmed across all tiers + missing/non-string batter names
        for bad_name in (None, "", 0, False):
            w = SandboxWorld(root / f"n13_{bad_name!r}")
            w.commit(name=bad_name)
            w.capture([UNIT_WRONG_1])
            rn = w.check("18:45", [make_row(77, 1, 1)])
            assert rn["dm"] == [TPL_FIRST.format(legs=leg1.replace("Batter 1010", "batter 1010"))]
        for p_name, d_name, exp_sent in (
            (None, "Batter 1020", "batter 1010 + DD Batter 1020"),
            ("Batter 1010", "", "Batter 1010 + DD batter 1020"),
            (None, 42, "batter 1010 + DD batter 1020"),
        ):
            w = SandboxWorld(root / f"m6_{exp_sent.replace(' ', '_')}")
            w.commit(name=p_name, dd=(1020, 201, PITCH, d_name), delivery="locked_unconfirmed")
            w.capture([UNIT_WRONG_1, UNIT_RIGHT_2])
            for at in ("18:45", "19:05", "19:20"):
                rm6 = w.check(at, [make_row(77, 1, 1), make_row(78, 2, 2)])
                exp_leg = leg1.replace("Batter 1010", p_name if isinstance(p_name, str) and p_name else "batter 1010")
                assert rm6["dm"] == [TPL_LOCKED.format(names=exp_sent, legs=exp_leg, qdir=quote(str(w.picks)))]
        checks_passed.append("M6+N13+R2.1")

        # M7 + R2 edit 3: cutoff passed during check
        w = SandboxWorld(root / "m7")
        w.commit()
        w.capture([UNIT_WRONG_1])
        rm7 = w.check("19:34", [make_row(77, 1, 1)], elapsed=61)
        assert rm7["exit"] == 1 and rm7["dm"] == [] and rm7["marker"]["game_unresolved"] is True
        assert rm7["stdout"] == "check-pick-entered: submission cutoff passed during this check; skipping the nag (entry can no longer change)\n"
        checks_passed.append("M7+R2.3")

        # M8: wrong game alerted -> entry removed before T-30 -> immediate initial DM
        w = SandboxWorld(root / "m8")
        w.commit()
        w.capture([UNIT_WRONG_1])
        w.check("18:30", [make_row(77, 1, 1)])
        rm8 = w.check("18:45", [])
        assert rm8["exit"] == 1 and rm8["marker"]["escalations"] == ["initial", "wrong_game"]
        assert rm8["marker"]["game_unresolved"] is True
        assert rm8["dm"] == ["⚠️ BTS pick NOT entered in MLB app: Batter 1010 — first pitch 7:40 PM ET (50 min to submit). Fix it now!"]
        checks_passed.append("M8")

        # M9 + R2 edit 2: lost binding in case (b), plus non-integer game_pk in case (b)
        w = SandboxWorld(root / "m9")
        w.commit()
        w.capture([UNIT_WRONG_1], f"{DAY}T200000Z")
        w.check("18:30", [make_row(77, 1, 1)])
        w.capture([UNIT_RIGHT_1], f"{DAY}T210000Z")
        rm9 = w.check("19:05", [make_row(77, 1, 1)])
        assert rm9["exit"] == 1 and rm9["marker"]["game_unresolved"] is True
        assert rm9["dm"] == [TPL_REMINDER.format(legs="pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET)")]

        for lbl, bad_gpk in (("null", None), ("str", "101"), ("float", 101.0), ("bool", True)):
            w = SandboxWorld(root / f"r2_2_{lbl}")
            w.commit(game=bad_gpk, dd=(1020, 201, PITCH))
            w.capture([{"id": 77, "roundId": 1, "feedId": None}, UNIT_WRONG_2])
            w.check("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)])
            w.capture([{"id": 77, "roundId": 1, "feedId": None}, UNIT_RIGHT_2])
            r_id = w.check("19:05", [make_row(77, 1, 1), make_row(78, 2, 2)])
            assert r_id["dm"] == [TPL_REMINDER.format(legs="pick Batter 1010: recommended game ID unavailable (first pitch 7:40 PM ET)")]
        checks_passed.append("M9+R2.2")

        # M10: raising read confirms batter only (Amendment 3)
        for prior_mode in ("none", "no_pick", "wrong_game"):
            w = SandboxWorld(root / f"m10_{prior_mode}")
            w.commit()
            w.capture([UNIT_WRONG_1])
            if prior_mode == "no_pick":
                w.check("18:30", [])
            elif prior_mode == "wrong_game":
                w.check("18:30", [make_row(77, 1, 1)])
            (w.units_dir / f"{DAY}T220000Z.json.gz").write_bytes(b"corrupt-not-gzip")
            rm10 = w.check("18:45", [make_row(77, 1, 1)])
            assert rm10["exit"] == 0
            assert rm10["marker"]["status"] == "confirmed" and rm10["marker"]["reason"] == "match_game_unchecked"
            assert "game_unresolved" not in rm10["marker"] and "escalations" not in rm10["marker"]
            assert rm10["stderr"] == (
                "check-pick-entered: game check unavailable (BadGzipFile); batter match only\n"
                "check-pick-entered: entry receipt unavailable (BadGzipFile)\n"
            )
            assert rm10["stdout"].endswith(f"check-pick-entered: {DAY} batter entered; the game could not be checked (match_game_unchecked)\n")
            assert rm10["dm"] == ([ALL_CLEAR_UNCHECKED] if prior_mode != "none" else [])
            assert w.check("19:00", [make_row(77, 1, 1)])["fetches"] == []
        checks_passed.append("M10")

        # M11, M12, M13: failed wrong_game DM + bridge + conflicting capture; and BOTH leg orders of partial double proof
        for has_init in (True, False):
            for bridge in ("crosswalk", "removal", "mismatch"):
                w = SandboxWorld(root / f"m11_12_{has_init}_{bridge}")
                w.commit()
                w.capture([UNIT_WRONG_1], f"{DAY}T200000Z")
                if has_init:
                    w.check("18:25", [])
                w.check("18:35", [make_row(77, 1, 1)], dm="failure")
                if bridge == "crosswalk":
                    w.check("18:45", [make_row(77, 1, 1)], crosswalk={})
                elif bridge == "removal":
                    w.check("18:45", [])
                else:
                    w.check("18:45", [make_row(77, 3, 1)])
                w.capture([UNIT_RIGHT_1], f"{DAY}T210000Z")
                r_end = w.check("19:00", [make_row(77, 1, 1)])
                assert r_end["exit"] == 1 and r_end["marker"]["reason"] == "wrong_game"
                assert r_end["marker"]["game_unresolved"] is True
                assert r_end["dm"] == [TPL_FIRST.format(legs="pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET)")]

        for proved_leg in (1, 2):
            w = SandboxWorld(root / f"m13_leg{proved_leg}")
            w.commit(dd=(1020, 201, PITCH))
            w.capture([UNIT_WRONG_1, UNIT_WRONG_2])
            w.check("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)], dm="failure")
            w.check("18:40", [make_row(77, 1, 1), make_row(78, 2, 2)], crosswalk={})
            if proved_leg == 1:
                w.capture([UNIT_RIGHT_1, {"id": 78, "roundId": 1, "feedId": None}])
            else:
                w.capture([{"id": 77, "roundId": 1, "feedId": None}, UNIT_RIGHT_2])
            r_part = w.check("18:45", [make_row(77, 1, 1), make_row(78, 2, 2)])
            assert r_part["exit"] == 1 and r_part["marker"]["game_unresolved"] is True
            assert ALL_CLEAR_MATCH not in r_part["attempted_dm"]
        checks_passed.append("M11+M12+M13")

        # Unreadable saved game_time raises ValueError before fetch
        w = SandboxWorld(root / "bad_time")
        w.commit(pitch="not-a-timestamp")
        r_bt = w.check("18:45", [make_row(77, 1, 1)])
        assert r_bt["exit"] != 0 and r_bt["exception"] == "ValueError" and r_bt["fetches"] == []

        # A6.2 + N11: EOD health check across all 4 marker statuses + pre-cutoff quietness
        for st_val, exp_lvl in (("alerted", ["WARN"]), ("dm_failed", ["CRITICAL"]), ("present_unverified", []), ("confirmed", [])):
            w = SandboxWorld(root / f"a6_2_{st_val}")
            w.commit()
            w.set_marker({"date": DAY, "status": st_val, "reason": "wrong_game", "game_unresolved": st_val != "confirmed"})
            assert [a["level"] for a in w.health(datetime(2027, 3, 24, 23, 34, tzinfo=timezone.utc))] == []
            assert [a["level"] for a in w.health(datetime(2027, 3, 24, 23, 35, tzinfo=timezone.utc))] == exp_lvl
        checks_passed.append("N11+A6.2")

        # A6.4: Real rollback check running archived base 1e69d0f0 against candidate's alerted/wrong_game marker
        w_rb = SandboxWorld(root / "a6_4_rollback")
        w_rb.commit()
        w_rb.capture([UNIT_WRONG_1])
        r_cand = w_rb.check("18:30", [make_row(77, 1, 1)])
        assert r_cand["marker"]["status"] == "alerted" and r_cand["marker"]["game_unresolved"] is True
        rb_code = (
            "import json, sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); "
            "from probe_common import SandboxWorld, bts_source_path, make_row; "
            "w = SandboxWorld(Path(sys.argv[2])); "
            "r1 = w.check('18:45', [make_row(77, 1, 1)]); "
            "r2 = w.check('19:00', [make_row(77, 1, 1)]); "
            "print(json.dumps({'bts': bts_source_path(), 'r1_exit': r1['exit'], 'r1_marker': r1['marker'], "
            "'r1_dm': r1['dm'], 'r2_fetches': r2['fetches'], 'r2_stdout': r2['stdout']}))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", rb_code, str(HERE), str(w_rb.root)],
            env={**os.environ, "PYTHONPATH": str(BASE_SRC), "TZ": "America/New_York"},
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        rb_out = json.loads(proc.stdout)
        assert str(BASE_SRC) in rb_out["bts"]
        assert rb_out["r1_exit"] == 0 and rb_out["r1_marker"]["status"] == "confirmed"
        assert "game_unresolved" not in rb_out["r1_marker"]
        assert rb_out["r1_dm"] == [ALL_CLEAR_MATCH]
        assert rb_out["r2_fetches"] == [] and rb_out["r2_stdout"] == f"check-pick-entered: already confirmed for {DAY}\n"
        checks_passed.append("A6.4")

        # A6.5: Morning day-outcome priority & separate-read disagreements
        w = SandboxWorld(root / "a6_5_unverified_clean")
        w.commit()
        w.capture([UNIT_WRONG_1], f"{DAY}T200000Z")
        w.check("18:30", [make_row(77, 1, 1)])
        w.capture([UNIT_RIGHT_1], f"{DAY}T210000Z")
        w.check("18:45", [make_row(77, 1, 1)])
        assert [a["level"] for a in w.health()] == ["WARN"]
        m_res = w.morning()
        assert m_res["exit"] == 0 and m_res["stdout"]["state"] == "entered_game_unverified" and m_res["dm"] == []

        w = SandboxWorld(root / "a6_5_entered_clean")
        w.commit()
        w.capture([UNIT_WRONG_1], f"{DAY}T200000Z")
        from bts import entry_receipt as er_mod
        orig_load = er_mod.load_units
        load_count = []

        def swap_after_first_read(*a):
            load_count.append(1)
            res_u = orig_load(*a)
            if len(load_count) == 1:
                w.capture([UNIT_RIGHT_1], f"{DAY}T200000Z")
            return res_u

        r_swap = w.check("18:30", [make_row(77, 1, 1)], extra_patches=[("bts.entry_receipt.load_units", swap_after_first_read)])
        assert r_swap["exit"] == 1 and [a["level"] for a in w.health()] == ["WARN"]
        m_swap = w.morning()
        assert m_swap["exit"] == 0 and m_swap["stdout"]["state"] == "entered"

        w_abs = SandboxWorld(root / "a6_5_not_entered")
        w_abs.commit()
        w_abs.capture([UNIT_WRONG_1])
        w_abs.check("18:30", [])
        w_abs.check("18:45", [make_row(77, 1, 1)])
        assert w_abs.morning()["stdout"]["state"] == "not_entered"

        w_only = SandboxWorld(root / "a6_5_unverifiable")
        w_only.commit()
        w_only.capture([UNIT_WRONG_1])
        w_only.check("18:45", [make_row(77, 1, 1)])
        assert w_only.morning()["stdout"]["state"] == "entry_unverifiable"
        checks_passed.append("A6.5")

    summary = {"bts_file": bts_source_path(), "checks_passed": checks_passed}
    (HERE / "log_2_rows_and_texts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_rows_and_texts()
