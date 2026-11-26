"""Probe 02 (Brief Items 1 cleanup, 2, and 3):
- Isolated re-check of D|* early-exit cells between base (1e69d0f0) and head (1d30de86)
- Full execution of contract rows N1-N13 and M1-M10 + closing-test snapshots with real capture files
- Systematic search for false alarms and missed wrong-game cases over real captures and response shapes
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from shlex import quote
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_SRC = HERE / "base_pkg" / "src"
HEAD_SRC = REPO_ROOT / "src"
CLOSING_DIR = REPO_ROOT / "docs" / "reviews" / "2026-11-18-whole-range" / "closing-test" / "codex-blind" / "evidence"


def isolated_early_exits_worker(out_file: Path):
    import httpx
    from bts.contest_fetch import ContestFetchError
    from bts.leaderboard.auth import AuthError
    from review_harness import DEFAULT_DAY, UNIT_WRONG_1, ReviewWorld, make_row, strip_volatile

    out = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, setup_fn, run_kw in [
            ("no_pick_file", lambda w: None, {}),
            ("not_committed_preview", lambda w: w.write_pick_and_decision(delivery=None, notification_sent=False), {}),
            ("not_committed_unscoreable", lambda w: w.write_pick_and_decision(delivery="delivered", scoreable=False, notification_sent=False), {}),
            ("identity_mismatch", lambda w: (w.write_pick_and_decision(), w.write_capture([UNIT_WRONG_1])), {"pending": [make_row()], "session_username": "intruder"}),
            ("fetch_login", lambda w: w.write_pick_and_decision(), {"fetch_error": ("login", AuthError("expired"))}),
            ("fetch_profile", lambda w: w.write_pick_and_decision(), {"fetch_error": ("profile", httpx.ConnectError("down"))}),
            ("fetch_pending", lambda w: w.write_pick_and_decision(), {"fetch_error": ("pending", ContestFetchError("bad"))}),
            ("fetch_rounds", lambda w: w.write_pick_and_decision(), {"fetch_error": ("rounds", ValueError("bad"))}),
            ("fetch_crosswalk", lambda w: w.write_pick_and_decision(), {"fetch_error": ("crosswalk", KeyError("bad"))}),
        ]:
            w = ReviewWorld(root / name)
            setup_fn(w)
            out[name] = strip_volatile(w.run_watchdog("18:30", **run_kw), w.picks_dir)
    out_file.write_text(json.dumps(out))


def verify_isolated_early_exits():
    with TemporaryDirectory() as tmp:
        b_out = Path(tmp) / "b.json"
        h_out = Path(tmp) / "h.json"
        for src_dir, out_f in ((BASE_SRC, b_out), (HEAD_SRC, h_out)):
            env = dict(os.environ, PYTHONPATH=f"{src_dir}:{HERE}", TZ="America/New_York")
            subprocess.run([sys.executable, __file__, "--early-worker", str(out_f)], env=env, timeout=30, check=True)
        b_data = json.loads(b_out.read_text())
        h_data = json.loads(h_out.read_text())
        assert b_data == h_data, f"Diff in isolated early exits: {b_data} vs {h_data}"
    return len(b_data)


def run_rows_and_counterexamples():
    sys.path.insert(0, str(HEAD_SRC))
    sys.path.insert(0, str(HERE))
    from review_harness import (
        DEFAULT_DAY,
        DEFAULT_PITCH,
        UNIT_RIGHT_1,
        UNIT_RIGHT_2,
        UNIT_WRONG_1,
        UNIT_WRONG_2,
        ReviewWorld,
        make_row,
    )

    FIRST_TPL = (
        "⚠️ BTS entry game needs verification. {legs}. "
        "Check the delivered pick and the game shown in MLB for each listed leg. "
        "Change only a leg whose correct game you can verify and that MLB still lets you edit. "
        "Keep any correct leg as entered."
    )
    REM_TPL = (
        "⏰ BTS entry game verification is still needed. {legs}. "
        "Check the delivered pick against the game shown in MLB."
    )
    LEG1_WRONG = "pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET), the entry is recorded in game 102"
    LEG2_WRONG = "double-down Batter 1020: recommended game 201 (first pitch 7:40 PM ET), the entry is recorded in game 202"

    row_results = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Replay the 3 real closing-test snapshots
        closing_checks = {}
        for label, feed_id in (("wrong_game", 102), ("right_game", 101), ("unbound", None)):
            gz = CLOSING_DIR / f"interactions-entry-{label}.json.gz"
            with gzip.open(gz, "rt") as fh:
                events = json.load(fh)
            snap = next(e[3] for e in reversed(events) if e[1] == "snapshot")
            w = ReviewWorld(root / f"closing_{label}")
            (w.picks_dir / f"{DEFAULT_DAY}.json").write_text(json.dumps(snap[f"data/picks/{DEFAULT_DAY}.json"]))
            dec_file = w.picks_dir / DEFAULT_DAY / "decision.json"
            dec_file.parent.mkdir(parents=True, exist_ok=True)
            dec_file.write_text(json.dumps(snap[f"data/picks/{DEFAULT_DAY}/decision.json"]))
            bid = snap[f"data/picks/{DEFAULT_DAY}.json"]["pick"]["batter_id"]
            w.write_capture([{"id": 77, "roundId": 1, "feedId": feed_id}], "20270324T210000Z")
            r_1830 = w.run_watchdog("18:30", [], crosswalk={1: bid}, expected_username=None)
            r_1845 = w.run_watchdog("18:45", [make_row(77, 1, 1)], crosswalk={1: bid}, expected_username=None)
            r_1900 = w.run_watchdog("19:00", [make_row(77, 1, 1)], crosswalk={1: bid}, expected_username=None)
            closing_checks[label] = [r_1830["exit_code"], r_1845["exit_code"], r_1900["exit_code"], w.read_marker()["status"]]
        assert closing_checks["wrong_game"] == [1, 1, 1, "alerted"], closing_checks
        assert closing_checks["right_game"] == [1, 0, 0, "confirmed"], closing_checks
        assert closing_checks["unbound"] == [1, 0, 0, "confirmed"], closing_checks
        row_results["closing_snapshots"] = closing_checks

        # N1: no_pick ladder
        w = ReviewWorld(root / "N1")
        w.write_pick_and_decision()
        r1 = w.run_watchdog("18:30", [])
        r2 = w.run_watchdog("18:45", [])
        r3 = w.run_watchdog("19:05", [])
        r4 = w.run_watchdog("19:20", [])
        assert [r1["exit_code"], r2["exit_code"], r3["exit_code"], r4["exit_code"]] == [1, 1, 1, 1]
        assert [len(r1["dm_sent"]), len(r2["dm_sent"]), len(r3["dm_sent"]), len(r4["dm_sent"])] == [1, 0, 1, 1]
        assert r4["marker"]["escalations"] == ["initial", "t15", "t30"]
        row_results["N1"] = "pass"

        # N2: mismatch (wrong batter and missing DD leg)
        w = ReviewWorld(root / "N2a")
        w.write_pick_and_decision()
        r = w.run_watchdog("18:30", [make_row(player=2)])
        assert r["exit_code"] == 1 and r["marker"]["reason"] == "mismatch" and "does NOT match" in r["dm_sent"][0]
        w = ReviewWorld(root / "N2b")
        w.write_pick_and_decision(dd=(1020, 201))
        r = w.run_watchdog("18:30", [make_row(player=1)])
        assert r["exit_code"] == 1 and r["marker"]["reason"] == "mismatch"
        row_results["N2"] = "pass"

        # N3: right batter in right game (bound unit), after alert -> all-clear + already confirmed
        w = ReviewWorld(root / "N3")
        w.write_pick_and_decision()
        w.write_capture([UNIT_RIGHT_1])
        w.run_watchdog("18:30", [])
        r_ok = w.run_watchdog("18:45", [make_row()])
        r_skip = w.run_watchdog("19:00", [make_row()])
        assert r_ok["exit_code"] == 0 and r_ok["marker"]["status"] == "confirmed"
        assert r_ok["dm_sent"] == [f"✅ BTS pick entry confirmed for {DEFAULT_DAY} (match)."]
        assert r_skip["exit_code"] == 0 and r_skip["fetches"] == [] and "already confirmed" in r_skip["stdout"]
        row_results["N3"] = "pass"

        # N4: no capture, absent unit, null feedId, wrong_round, ambiguous (on day with no prior wrong_game)
        for sub, units_cfg, pend in [
            ("no_capture", None, [make_row()]),
            ("absent_unit", [{"id": 88, "roundId": 1, "feedId": 101}], [make_row()]),
            ("null_feed", [{"id": 77, "roundId": 1, "feedId": None}], [make_row()]),
            ("wrong_round", [{"id": 77, "roundId": 2, "feedId": 102}], [make_row()]),
            ("ambiguous", [UNIT_WRONG_1, {"id": 78, "roundId": 1, "feedId": 102}], [make_row(77, 1, 1), make_row(78, 1, 2)]),
        ]:
            w = ReviewWorld(root / f"N4_{sub}")
            w.write_pick_and_decision()
            if units_cfg is not None:
                w.write_capture(units_cfg)
            r = w.run_watchdog("18:30", pend)
            assert r["exit_code"] == 0 and r["marker"]["status"] == "confirmed" and r["dm_sent"] == [], (sub, r)
            if sub in ("no_capture", "absent_unit", "null_feed"):
                assert w.run_day_outcome()["outcome"]["state"] == "entered_game_unverified"
        row_results["N4"] = "pass"

        # N5: present_unverified
        w = ReviewWorld(root / "N5")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        r1 = w.run_watchdog("18:30", [make_row(player=99)])
        r2 = w.run_watchdog("18:45", [make_row(player=99)])
        assert r1["exit_code"] == 0 and r1["marker"]["status"] == "present_unverified"
        assert r2["fetches"] != []  # non-terminal
        row_results["N5"] = "pass"

        # N6 & N7: covered in verify_isolated_early_exits + outside window + already confirmed
        w = ReviewWorld(root / "N7_win")
        w.write_pick_and_decision()
        assert w.run_watchdog("12:00", [make_row()])["fetches"] == []
        w.set_marker({"date": DEFAULT_DAY, "status": "confirmed", "reason": "match"})
        assert w.run_watchdog("18:30", [make_row()])["fetches"] == []
        row_results["N6_N7"] = "pass"

        # N8: locked_unconfirmed with no_pick and mismatch
        for sub, pend, clause in [("no_pick", [], "No entry found in MLB"), ("mismatch", [make_row(player=2)], "The MLB entry differs")]:
            w = ReviewWorld(root / f"N8_{sub}")
            w.write_pick_and_decision(delivery="locked_unconfirmed")
            r = w.run_watchdog("18:30", pend)
            assert clause in r["dm_sent"][0] and r["exit_code"] == 1
        row_results["N8"] = "pass"

        # N9: double with both legs in right games
        w = ReviewWorld(root / "N9")
        w.write_pick_and_decision(dd=(1020, 201))
        w.write_capture([UNIT_RIGHT_1, UNIT_RIGHT_2])
        r = w.run_watchdog("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)])
        assert r["exit_code"] == 0 and r["marker"]["status"] == "confirmed"
        row_results["N9"] = "pass"

        # N10: wrong game seen -> moved to right game (all legs confirmed needed on a double!)
        w = ReviewWorld(root / "N10_double")
        w.write_pick_and_decision(dd=(1020, 201))
        w.write_capture([UNIT_RIGHT_1, UNIT_WRONG_1 | {"id": 87}, UNIT_RIGHT_2, {"id": 88, "roundId": 1, "feedId": None}])
        # 18:30: primary in wrong unit 87 (102), DD in right unit 78 (201) -> wrong_game
        r1 = w.run_watchdog("18:30", [make_row(87, 1, 1), make_row(78, 2, 2)])
        assert r1["exit_code"] == 1 and r1["dm_sent"] == [FIRST_TPL.format(legs=LEG1_WRONG)]
        # 18:45: primary fixed to unit 77 (101), DD moved to unbound unit 88 -> NOT all_confirmed -> stays unresolved!
        r2 = w.run_watchdog("18:45", [make_row(77, 1, 1), make_row(88, 2, 2)])
        assert r2["exit_code"] == 1 and r2["marker"]["status"] == "alerted" and r2["dm_sent"] == []
        # 19:00: both in confirmed units 77 & 78 -> confirmed + all-clear DM!
        r3 = w.run_watchdog("19:00", [make_row(77, 1, 1), make_row(78, 2, 2)])
        assert r3["exit_code"] == 0 and r3["marker"]["status"] == "confirmed"
        assert r3["dm_sent"] == [f"✅ BTS pick entry confirmed for {DEFAULT_DAY} (match)."]
        row_results["N10"] = "pass"

        # N11: day-outcome and health check on unresolved wrong_game day
        w = ReviewWorld(root / "N11")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        w.run_watchdog("18:30", [make_row()])
        hlth = w.run_health_pick_entry()
        morn = w.run_day_outcome()
        assert len(hlth) == 1 and hlth[0]["level"] == "WARN" and "last status=alerted, reason=wrong_game" in hlth[0]["message"]
        assert morn["exit_code"] == 1 and morn["outcome"]["state"] == "entry_unverifiable"
        row_results["N11"] = "pass"

        # N12: null game_pk and shared batter on day with no prior wrong-game observation
        w = ReviewWorld(root / "N12_null")
        w.write_pick_and_decision(game_pk=None)
        w.write_capture([UNIT_WRONG_1])
        assert w.run_watchdog("18:30", [make_row()])["marker"]["status"] == "confirmed"

        w = ReviewWorld(root / "N12_shared")
        w.write_pick_and_decision(batter_id=1010, game_pk=101, dd=(1010, 201))
        w.write_capture([UNIT_WRONG_1, UNIT_WRONG_2])
        assert w.run_watchdog("18:30", [make_row(77, 1, 1)])["marker"]["status"] == "confirmed"
        row_results["N12"] = "pass"

        # N13: null or empty batter_name in wrong-game leg (single and double, normal and locked_unconfirmed)
        for nm in (None, ""):
            for deliv in ("delivered", "locked_unconfirmed"):
                w = ReviewWorld(root / f"N13_{nm}_{deliv}")
                w.write_pick_and_decision(batter_name=nm, dd=(1020, 201, DEFAULT_PITCH, nm), delivery=deliv)
                w.write_capture([UNIT_WRONG_1, UNIT_RIGHT_2])
                r = w.run_watchdog("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)])
                assert r["exit_code"] == 1 and r["exception"] is None
                assert "pick batter 1010: recommended game 101" in r["dm_sent"][0]
                assert "double-down" not in r["dm_sent"][0].split("MLB game evidence")[-1] if deliv == "locked_unconfirmed" else "double-down" not in r["dm_sent"][0]
        row_results["N13"] = "pass"

        # M1 to M10 explicit verification
        # M1
        w = ReviewWorld(root / "M1")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        assert w.run_watchdog("18:30", [])["marker"]["escalations"] == ["initial"]
        r_m1 = w.run_watchdog("18:45", [make_row()])
        assert r_m1["exit_code"] == 1 and r_m1["marker"]["escalations"] == ["initial", "wrong_game"]
        assert r_m1["dm_sent"] == [FIRST_TPL.format(legs=LEG1_WRONG)]
        r_m1_re = w.run_watchdog("19:00", [make_row()])
        assert r_m1_re["exit_code"] == 1 and r_m1_re["fetches"] != [] and r_m1_re["dm_sent"] == []
        assert r_m1_re["stdout"] == f"check-pick-entered: {DEFAULT_DAY} game verification unresolved (pick); re-verifying each run until cutoff\n"

        # M2
        w = ReviewWorld(root / "M2")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        r_m2 = w.run_watchdog("18:30", [make_row()])
        assert r_m2["marker"]["escalations"] == ["wrong_game"] and r_m2["dm_sent"] == [FIRST_TPL.format(legs=LEG1_WRONG)]

        # M3 (both legs tested: primary right + DD wrong, primary wrong + DD right, both wrong)
        w = ReviewWorld(root / "M3")
        w.write_pick_and_decision(dd=(1020, 201))
        w.write_capture([UNIT_RIGHT_1, UNIT_WRONG_2, {"id": 87, "roundId": 1, "feedId": 102}])
        r_dd = w.run_watchdog("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)])
        assert r_dd["dm_sent"] == [FIRST_TPL.format(legs=LEG2_WRONG)]
        w.set_marker(None)
        r_pr = w.run_watchdog("18:30", [make_row(87, 1, 1), make_row(78, 2, 2)])
        assert r_pr["dm_sent"] == [FIRST_TPL.format(legs=f"{LEG1_WRONG}; {LEG2_WRONG}")]

        # M4
        w = ReviewWorld(root / "M4")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        w.run_watchdog("18:30", [make_row()])
        r30 = w.run_watchdog("19:05", [make_row()])
        r30b = w.run_watchdog("19:06", [make_row()])
        r15 = w.run_watchdog("19:20", [make_row()])
        r15b = w.run_watchdog("19:21", [make_row()])
        assert r30["dm_sent"] == [REM_TPL.format(legs=LEG1_WRONG)] and r30b["dm_sent"] == []
        assert r15["dm_sent"] == [REM_TPL.format(legs=LEG1_WRONG)] and r15b["dm_sent"] == []

        # M5
        w = ReviewWorld(root / "M5")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        rf = w.run_watchdog("18:30", [make_row()], dm_mode="failure")
        assert rf["marker"]["status"] == "dm_failed" and rf["marker"]["escalations"] == []
        rr = w.run_watchdog("18:31", [make_row()])
        assert rr["marker"]["status"] == "alerted" and rr["marker"]["escalations"] == ["wrong_game"] and len(rr["dm_sent"]) == 1

        # M6 (locked_unconfirmed in wrong_game, t30, and t15)
        w = ReviewWorld(root / "M6")
        w.write_pick_and_decision(delivery="locked_unconfirmed")
        w.write_capture([UNIT_WRONG_1])
        expected_lu = (
            f"⚠️ Delivery of this pick was NOT confirmed. The scheduler tried to send Batter 1010. "
            f"MLB game evidence needs verification. {LEG1_WRONG}. "
            f"Check your DMs and the MLB entry before making any change. "
            f"Run `bts season-contract status --picks-dir {quote(str(w.picks_dir))}`."
        )
        for hhmm in ("18:30", "19:05", "19:20"):
            r_lu = w.run_watchdog(hhmm, [make_row()])
            assert r_lu["dm_sent"] == [expected_lu], (hhmm, r_lu["dm_sent"])

        # M7
        w = ReviewWorld(root / "M7")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        r_m7 = w.run_watchdog("19:34", [make_row()], elapsed_sec=61.0)
        assert r_m7["exit_code"] == 1 and r_m7["marker"]["status"] == "alerted" and r_m7["marker"]["reason"] == "wrong_game"
        assert r_m7["dm_sent"] == [] and "submission cutoff passed during fetch" in r_m7["stdout"]

        # M8 (wrong game alerted -> removed before T-30, and also wrong batter -> immediate initial DM!)
        for sub, pend, expect_sub in [("removed", [], "NOT entered"), ("wrong_batter", [make_row(player=2)], "does NOT match")]:
            w = ReviewWorld(root / f"M8_{sub}")
            w.write_pick_and_decision()
            w.write_capture([UNIT_WRONG_1])
            w.run_watchdog("18:30", [make_row()])
            r_m8 = w.run_watchdog("18:45", pend)
            assert r_m8["marker"]["escalations"] == ["initial", "wrong_game"]
            assert len(r_m8["dm_sent"]) == 1 and expect_sub in r_m8["dm_sent"][0]

        # M9
        w = ReviewWorld(root / "M9")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1], "20270324T200000Z")
        w.run_watchdog("18:30", [make_row()])
        w.write_capture([UNIT_RIGHT_1], "20270324T210000Z")  # conflicting capture -> unit_unverified
        r_m9 = w.run_watchdog("19:05", [make_row()])
        assert r_m9["exit_code"] == 1 and r_m9["marker"]["status"] == "alerted" and r_m9["marker"]["reason"] == "wrong_game"
        assert r_m9["dm_sent"] == [REM_TPL.format(legs="pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET)")]

        # M10 (Amendment 3)
        for prior_kind in ("none", "no_pick", "wrong_game"):
            w = ReviewWorld(root / f"M10_{prior_kind}")
            w.write_pick_and_decision()
            cap = w.write_capture([UNIT_WRONG_1])
            if prior_kind == "no_pick":
                w.run_watchdog("18:30", [])
            elif prior_kind == "wrong_game":
                w.run_watchdog("18:30", [make_row()])
            cap.write_bytes(b"corrupted gzip")
            r_m10 = w.run_watchdog("18:45", [make_row()])
            assert r_m10["exit_code"] == 0
            assert r_m10["marker"]["status"] == "confirmed" and r_m10["marker"]["reason"] == "match_game_unchecked"
            assert "escalations" not in r_m10["marker"]
            expected_dm = [] if prior_kind == "none" else [f"✅ BTS pick entry: the batter is entered for {DEFAULT_DAY}; the game could not be checked."]
            assert r_m10["dm_sent"] == expected_dm
            assert "game check unavailable (BadGzipFile); batter match only" in r_m10["stderr"]
            r_next = w.run_watchdog("19:00", [make_row()])
            assert r_next["fetches"] == [] and "already confirmed" in r_next["stdout"]
        row_results["M1_to_M10"] = "pass"

        # Item 3: Adversarial real-capture false-alarm & missed-case sweep
        item3_cases = []

        def check_item3(name, *, gpk=101, dd=None, captures=(), pending=(), profile=None, expect_wrong_game: bool):
            w3 = ReviewWorld(root / f"i3_{name}")
            w3.write_pick_and_decision(game_pk=gpk, dd=dd)
            for idx, c_units in enumerate(captures):
                w3.write_capture(c_units, f"20270324T20{idx:02d}00Z")
            res = w3.run_watchdog("18:30", pending, profile=profile)
            is_wg = (res["exit_code"] == 1 and res["marker"]["reason"] == "wrong_game")
            if not expect_wrong_game:
                assert res["exit_code"] == 0 and res["marker"]["status"] == "confirmed", (name, res)
            else:
                assert is_wg and len(res["dm_sent"]) == 1, (name, res)
            item3_cases.append({"case": name, "expect_wrong_game": expect_wrong_game, "exit": res["exit_code"], "reason": res["marker"]["reason"]})

        # Must-detect wrong-game cases (real captures + tricky valid shapes):
        # 1. Pre-binding null feedId in round 1 followed by bound wrong feedId in round 1 -> MUST detect wrong_game!
        check_item3(
            "prebinding_null_then_wrong_bound",
            captures=[[{"id": 77, "roundId": 1, "feedId": None}], [UNIT_WRONG_1]],
            pending=[make_row(77, 1, 1)],
            expect_wrong_game=True,
        )
        # 2. Same row in BOTH profile (settled hit) and pending (result None) -> deduplicated to 1 hit -> MUST detect wrong_game!
        check_item3(
            "settled_profile_plus_pending_same_row",
            captures=[[UNIT_WRONG_1]],
            pending=[make_row(77, 1, 1, result=None)],
            profile={"predictions": [{"roundId": 1, "roundPredictions": [make_row(77, 1, 1, result="hit")]}]},
            expect_wrong_game=True,
        )
        # 3. Wrong game row in profile only (pending already cleared after settlement) -> MUST detect wrong_game!
        check_item3(
            "profile_only_wrong_game",
            captures=[[UNIT_WRONG_1]],
            pending=[],
            profile={"predictions": [{"roundId": 1, "roundPredictions": [make_row(77, 1, 1, result="not_hit")]}]},
            expect_wrong_game=True,
        )
        # 4. Single pick, operator also entered an extra unmapped batter (playerId=99) alongside matched batter in wrong game -> MUST detect!
        check_item3(
            "single_with_extra_unmapped_row",
            captures=[[UNIT_WRONG_1]],
            pending=[make_row(77, 1, 1), make_row(88, 99, 2)],
            expect_wrong_game=True,
        )
        # 5. Double-down where legs swapped games (1010 in 201, 1020 in 101) -> MUST detect both legs!
        check_item3(
            "double_swapped_games",
            gpk=101,
            dd=(1020, 201),
            captures=[[UNIT_RIGHT_1, UNIT_RIGHT_2]],
            pending=[make_row(78, 1, 1), make_row(77, 2, 2)],
            expect_wrong_game=True,
        )
        # 6. Double-down where Leg 1 has null game_pk (ignored by A1a) and Leg 2 has int game_pk in wrong game -> MUST detect Leg 2!
        check_item3(
            "double_null_leg1_wrong_leg2",
            gpk=None,
            dd=(1020, 201),
            captures=[[UNIT_WRONG_1, UNIT_WRONG_2]],
            pending=[make_row(77, 1, 1), make_row(78, 2, 2)],
            expect_wrong_game=True,
        )

        # Must-NOT-false-alarm cases (must confirm on day with no prior wrong_game):
        # 7. Pre-binding capture in round 2 + bound capture in round 1 -> round conflict -> unit_unverified -> confirms!
        check_item3(
            "prebinding_conflicting_round_confirms",
            captures=[[{"id": 77, "roundId": 2, "feedId": None}], [UNIT_WRONG_1]],
            pending=[make_row(77, 1, 1)],
            expect_wrong_game=False,
        )
        # 8. Two captures binding unit 77 to 102 then 101 (rebind) -> unit_unverified -> confirms!
        check_item3(
            "rebind_two_feeds_confirms",
            captures=[[UNIT_WRONG_1], [UNIT_RIGHT_1]],
            pending=[make_row(77, 1, 1)],
            expect_wrong_game=False,
        )
        # 9. Unit has roundId=None even though feedId=102 -> wrong_round (None != 1) -> confirms!
        check_item3(
            "unit_null_round_confirms",
            captures=[[{"id": 77, "roundId": None, "feedId": 102}]],
            pending=[make_row(77, 1, 1)],
            expect_wrong_game=False,
        )
        # 10. Capture from another season (2026*.json.gz) has wrong game -> ignored for 2027 -> confirms!
        w_season = ReviewWorld(root / "other_season")
        w_season.write_pick_and_decision()
        w_season.write_capture([UNIT_WRONG_1], "20260925T210000Z")
        r_season = w_season.run_watchdog("18:30", [make_row()])
        assert r_season["exit_code"] == 0 and r_season["marker"]["status"] == "confirmed"
        item3_cases.append({"case": "other_season_capture_ignored", "expect_wrong_game": False, "exit": 0, "reason": "match"})

    return {"rows": row_results, "item3_count": len(item3_cases), "item3_cases": item3_cases}


def main():
    early_checked = verify_isolated_early_exits()
    res = run_rows_and_counterexamples()
    summary = {"isolated_early_exits_identical": early_checked, **res}
    (HERE / "probe_02_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--early-worker":
        isolated_early_exits_worker(Path(sys.argv[2]))
    else:
        main()
