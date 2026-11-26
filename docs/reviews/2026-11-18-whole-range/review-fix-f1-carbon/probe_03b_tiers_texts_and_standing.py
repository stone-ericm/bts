"""Probe 03b (Brief Items 4, 5, and 6 — all timestamps strictly inside the 75-min pre-pitch window):
- Item 4: Exhaustive event orderings through the marker and reminder tiers (A2)
- Item 5: Character-for-character operator text checks across worst ordinary states (A3, Amendment 3, R64)
- Item 6: Standing rows (malformed captures, FIFO timeout, interrupts, midnight/year boundary, failed sends, bad markers)
"""
from __future__ import annotations

from datetime import date, datetime
import gzip
import json
import os
from pathlib import Path
from shlex import quote
import subprocess
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_SRC = HERE / "base_pkg" / "src"
HEAD_SRC = REPO_ROOT / "src"

sys.path.insert(0, str(HEAD_SRC))
sys.path.insert(0, str(HERE))

from bts import entry_receipt  # noqa: E402
from review_harness import (  # noqa: E402
    DEFAULT_DAY,
    DEFAULT_PITCH,
    UNIT_RIGHT_1,
    UNIT_RIGHT_2,
    UNIT_WRONG_1,
    UNIT_WRONG_2,
    ReviewWorld,
    make_row,
)

FIRST_ALERT = (
    "\u26a0\ufe0f BTS entry game needs verification. {legs}. "
    "Check the delivered pick and the game shown in MLB for each listed leg. "
    "Change only a leg whose correct game you can verify and that MLB still lets you edit. "
    "Keep any correct leg as entered."
)
REMINDER_ALERT = (
    "\u23f0 BTS entry game verification is still needed. {legs}. "
    "Check the delivered pick against the game shown in MLB."
)
LOCKED_UNCONFIRMED_ALERT = (
    "\u26a0\ufe0f Delivery of this pick was NOT confirmed. The scheduler tried to send {names}. "
    "MLB game evidence needs verification. {legs}. "
    "Check your DMs and the MLB entry before making any change. "
    "Run `bts season-contract status --picks-dir {qdir}`."
)
UNCHECKED_ALL_CLEAR = "\u2705 BTS pick entry: the batter is entered for {day}; the game could not be checked."
NORMAL_ALL_CLEAR = "\u2705 BTS pick entry confirmed for {day} (match)."


def check_item4_tiers(root: Path) -> dict:
    checked = []

    def new_world(name: str) -> ReviewWorld:
        w = ReviewWorld(root / f"t4_{name}")
        w.write_pick_and_decision()  # DEFAULT_PITCH = 19:40 ET -> window is (18:25, 19:35), cutoff = 19:35 ET
        w.write_capture([UNIT_RIGHT_1, {"id": 78, "roundId": 1, "feedId": 102}, {"id": 79, "roundId": 1, "feedId": None}])
        return w

    R_NONE = []
    R_WRONG = [make_row(78, 1, 1)]
    R_RIGHT = [make_row(77, 1, 1)]
    R_UNBOUND = [make_row(79, 1, 1)]
    R_XW_GAP = [make_row(78, 99, 1)]

    # 1. All base tiers consumed (initial, t30, t15) -> wrong_game STILL fires immediately!
    w = new_world("all_base_spent_then_wg")
    assert len(w.run_watchdog("18:30", R_NONE)["dm_sent"]) == 1  # initial
    assert len(w.run_watchdog("19:05", R_NONE)["dm_sent"]) == 1  # t30
    assert len(w.run_watchdog("19:20", R_NONE)["dm_sent"]) == 1  # t15
    r_wg = w.run_watchdog("19:25", R_WRONG)
    assert len(r_wg["dm_sent"]) == 1 and r_wg["dm_sent"][0].startswith("\u26a0\ufe0f BTS entry game needs verification.")
    assert r_wg["marker"]["escalations"] == ["initial", "t15", "t30", "wrong_game"]
    assert w.run_watchdog("19:30", R_WRONG)["dm_sent"] == []
    checked.append("all_base_spent_then_wg")

    # 2. First wrong_game at exact t30 (19:05) and exact t15 (19:20) consumes t30 / t30+t15, NOT initial!
    w = new_world("wg_first_at_t30")
    r1 = w.run_watchdog("19:05", R_WRONG)
    assert r1["dm_sent"][0].startswith("\u26a0\ufe0f BTS entry game needs verification.")
    assert r1["marker"]["escalations"] == ["t30", "wrong_game"]
    assert w.run_watchdog("19:10", R_WRONG)["dm_sent"] == []
    r2 = w.run_watchdog("19:20", R_WRONG)
    assert r2["dm_sent"][0].startswith("\u23f0 BTS entry game verification is still needed.")
    assert r2["marker"]["escalations"] == ["t15", "t30", "wrong_game"]
    # Removed at 19:25 -> initial STILL fires immediately!
    r3 = w.run_watchdog("19:25", R_NONE)
    assert len(r3["dm_sent"]) == 1 and "BTS pick NOT entered" in r3["dm_sent"][0]
    assert r3["marker"]["escalations"] == ["initial", "t15", "t30", "wrong_game"]
    checked.append("wg_first_at_t30_then_t15_then_removed")

    w = new_world("wg_first_at_t15")
    r = w.run_watchdog("19:20", R_WRONG)
    assert r["marker"]["escalations"] == ["t15", "t30", "wrong_game"]
    assert w.run_watchdog("19:25", R_WRONG)["dm_sent"] == []
    checked.append("wg_first_at_t15")

    # 3. Wrong game -> removed -> base t30 -> wrong game again at t15 -> fixed -> capture change ignored
    w = new_world("wg_removed_t30_wg_t15_fixed_then_change")
    assert len(w.run_watchdog("18:26", R_WRONG)["dm_sent"]) == 1  # wrong_game
    r_rem = w.run_watchdog("18:35", R_NONE)  # initial!
    assert len(r_rem["dm_sent"]) == 1 and "BTS pick NOT entered" in r_rem["dm_sent"][0]
    assert w.run_watchdog("18:45", R_NONE)["dm_sent"] == []
    r_t30_np = w.run_watchdog("19:05", R_NONE)  # base t30!
    assert len(r_t30_np["dm_sent"]) == 1 and "STILL NOT entered" in r_t30_np["dm_sent"][0]
    r_t15_wg = w.run_watchdog("19:20", R_WRONG)  # wrong_game t15 reminder!
    assert len(r_t15_wg["dm_sent"]) == 1 and "game verification is still needed" in r_t15_wg["dm_sent"][0]
    r_fix = w.run_watchdog("19:25", R_RIGHT)
    assert r_fix["exit_code"] == 0 and r_fix["dm_sent"] == [NORMAL_ALL_CLEAR.format(day=DEFAULT_DAY)]
    assert r_fix["marker"]["escalations"] == ["initial", "t15", "t30", "wrong_game"]
    w.write_capture([{"id": 77, "roundId": 1, "feedId": 999}], "20270324T220000Z")
    r_after = w.run_watchdog("19:30", R_WRONG)
    assert r_after["exit_code"] == 0 and r_after["fetches"] == [] and r_after["dm_sent"] == []
    checked.append("wg_removed_t30_wg_t15_fixed_then_change")

    # 4. DM failures at every step + retry + failed all-clear
    w = new_world("dm_failures_every_step")
    assert w.run_watchdog("18:26", R_NONE, dm_mode="failure")["marker"]["status"] == "dm_failed"
    assert w.run_watchdog("18:27", R_NONE)["marker"]["escalations"] == ["initial"]
    rf_wg = w.run_watchdog("18:30", R_WRONG, dm_mode="failure")
    assert rf_wg["marker"]["status"] == "alerted" and rf_wg["marker"]["escalations"] == ["initial"]
    assert w.run_watchdog("18:31", R_WRONG)["marker"]["escalations"] == ["initial", "wrong_game"]
    rf_30 = w.run_watchdog("19:05", R_WRONG, dm_mode="failure")
    assert rf_30["marker"]["escalations"] == ["initial", "wrong_game"]
    assert len(w.run_watchdog("19:06", R_WRONG)["dm_sent"]) == 1
    rf_15 = w.run_watchdog("19:20", R_WRONG, dm_mode="failure")
    assert rf_15["marker"]["escalations"] == ["initial", "t30", "wrong_game"]
    assert len(w.run_watchdog("19:21", R_WRONG)["dm_sent"]) == 1
    rf_ac = w.run_watchdog("19:25", R_RIGHT, dm_mode="failure")
    assert rf_ac["exit_code"] == 0 and rf_ac["marker"]["status"] == "confirmed"
    assert "confirmation DM failed" in rf_ac["stderr"]
    assert w.run_watchdog("19:26", R_RIGHT)["fetches"] == []
    checked.append("dm_failures_every_step")

    # 5. Failed first wrong_game -> cutoff passes during fetch -> rewrites dm_failed to alerted (A5.3)
    w = new_world("dm_failed_then_cutoff")
    assert w.run_watchdog("18:30", R_WRONG, dm_mode="failure")["marker"]["status"] == "dm_failed"
    rc = w.run_watchdog("19:34", R_WRONG, elapsed_sec=61.0)
    assert rc["exit_code"] == 1 and rc["marker"]["status"] == "alerted" and rc["marker"]["escalations"] == []
    checked.append("dm_failed_then_cutoff")

    # 6. Interleaved present_unverified transitions
    # 6a: wrong_game (sent) -> present_unverified -> right game -> MUST send all-clear!
    w = new_world("wg_then_pu_then_right")
    w.run_watchdog("18:30", R_WRONG)
    r_pu = w.run_watchdog("18:40", R_XW_GAP)
    assert r_pu["marker"]["status"] == "present_unverified" and r_pu["marker"]["escalations"] == ["wrong_game"]
    r_ok = w.run_watchdog("18:50", R_RIGHT)
    assert r_ok["exit_code"] == 0 and r_ok["dm_sent"] == [NORMAL_ALL_CLEAR.format(day=DEFAULT_DAY)]
    checked.append("wg_then_pu_then_right")

    # 6b: wrong_game (sent) -> present_unverified -> unbound -> stays unresolved (A1b)!
    w = new_world("wg_then_pu_then_unbound")
    w.run_watchdog("18:30", R_WRONG)
    w.run_watchdog("18:40", R_XW_GAP)
    r_u = w.run_watchdog("19:05", R_UNBOUND)
    assert r_u["exit_code"] == 1 and r_u["marker"]["status"] == "alerted" and len(r_u["dm_sent"]) == 1
    checked.append("wg_then_pu_then_unbound")

    # 6c: no_pick (sent) -> present_unverified -> wrong_game (DM fails) -> keeps ["initial"] -> right game sends all-clear!
    w = new_world("np_then_pu_then_failed_wg_then_right")
    w.run_watchdog("18:30", R_NONE)
    w.run_watchdog("18:40", R_XW_GAP)
    rf = w.run_watchdog("18:45", R_WRONG, dm_mode="failure")
    assert rf["marker"]["status"] == "dm_failed" and rf["marker"]["escalations"] == ["initial"]
    r_ok = w.run_watchdog("18:50", R_RIGHT)
    assert r_ok["dm_sent"] == [NORMAL_ALL_CLEAR.format(day=DEFAULT_DAY)]
    checked.append("np_then_pu_then_failed_wg_then_right")

    # 6d: wrong_game (sent) -> present_unverified -> no_pick (DM fails) -> keeps ["wrong_game"]!
    w = new_world("wg_then_pu_then_failed_np")
    w.run_watchdog("18:30", R_WRONG)
    w.run_watchdog("18:40", R_XW_GAP)
    rf = w.run_watchdog("18:45", R_NONE, dm_mode="failure")
    assert rf["marker"]["status"] == "dm_failed" and rf["marker"]["escalations"] == ["wrong_game"]
    r_retry = w.run_watchdog("18:46", R_NONE)
    assert len(r_retry["dm_sent"]) == 1 and r_retry["marker"]["escalations"] == ["initial", "wrong_game"]
    checked.append("wg_then_pu_then_failed_np")

    # 7. No DM recipient consumes tiers as base (A2 / A5.3)
    w = new_world("no_recipient")
    r_nr = w.run_watchdog("19:05", R_WRONG, dm_mode="none")
    assert r_nr["exit_code"] == 1 and r_nr["dm_attempted"] == []
    assert r_nr["marker"]["status"] == "alerted" and r_nr["marker"]["escalations"] == ["t30", "wrong_game"]
    checked.append("no_recipient")

    return {"sequences_verified": len(checked), "names": checked}


def check_item5_texts(root: Path) -> dict:
    verified = []

    def assert_no_unsafe_words(msg: str):
        for forbidden in ("min to submit", "min left to submit", "Fix it now", "Move the entry"):
            assert forbidden not in msg, f"Forbidden phrase {forbidden!r} in {msg!r}"

    # 1. Double with only primary affected, only DD affected, and both affected (naive UTC, Z UTC, ET offset)
    w = ReviewWorld(root / "txt_double")
    w.write_pick_and_decision(
        batter_id=1010,
        batter_name="Aaron Judge",
        game_pk=101,
        game_time="2027-03-24T23:40:00",
        dd=(1020, 201, "2027-03-25T02:05:00Z", "Juan Soto"),
    )
    w.write_capture([
        {"id": 77, "roundId": 1, "feedId": 101},
        {"id": 87, "roundId": 1, "feedId": 102},
        {"id": 78, "roundId": 1, "feedId": 201},
        {"id": 88, "roundId": 1, "feedId": 202},
    ])
    leg_p = "pick Aaron Judge: recommended game 101 (first pitch 7:40 PM ET), the entry is recorded in game 102"
    leg_d = "double-down Juan Soto: recommended game 201 (first pitch 10:05 PM ET), the entry is recorded in game 202"

    r = w.run_watchdog("18:30", [make_row(77, 1, 1), make_row(88, 2, 2)])
    assert r["dm_sent"] == [FIRST_ALERT.format(legs=leg_d)]
    assert_no_unsafe_words(r["dm_sent"][0])
    r_quiet = w.run_watchdog("18:45", [make_row(77, 1, 1), make_row(88, 2, 2)])
    assert r_quiet["stdout"] == f"check-pick-entered: {DEFAULT_DAY} game verification unresolved (double-down); re-verifying each run until cutoff\n"

    r_both_30 = w.run_watchdog("19:05", [make_row(87, 1, 1), make_row(88, 2, 2)])
    assert r_both_30["dm_sent"] == [REMINDER_ALERT.format(legs=f"{leg_p}; {leg_d}")]
    assert_no_unsafe_words(r_both_30["dm_sent"][0])
    r_both_quiet = w.run_watchdog("19:10", [make_row(87, 1, 1), make_row(88, 2, 2)])
    assert r_both_quiet["stdout"] == f"check-pick-entered: {DEFAULT_DAY} game verification unresolved (pick + double-down); re-verifying each run until cutoff\n"
    verified.append("double_one_and_both_legs_and_timezones")

    # 2. Missing names (None and "") in single and double
    for bad_name in (None, ""):
        w_n = ReviewWorld(root / f"txt_name_{bad_name is None}")
        w_n.write_pick_and_decision(batter_name=bad_name, dd=(1020, 201, DEFAULT_PITCH, bad_name))
        w_n.write_capture([UNIT_WRONG_1, UNIT_WRONG_2])
        r_n = w_n.run_watchdog("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)])
        expected_legs = (
            "pick batter 1010: recommended game 101 (first pitch 7:40 PM ET), the entry is recorded in game 102; "
            "double-down batter 1020: recommended game 201 (first pitch 7:40 PM ET), the entry is recorded in game 202"
        )
        assert r_n["dm_sent"] == [FIRST_ALERT.format(legs=expected_legs)]
        assert_no_unsafe_words(r_n["dm_sent"][0])
    verified.append("missing_names_null_and_empty")

    # 3. Unreadable game_time (both unpatched pre-fetch raise per A5.4, and LEGS fallback omission per A3)
    w_gt = ReviewWorld(root / "txt_bad_gt_unpatched")
    w_gt.write_pick_and_decision(game_time="not-a-timestamp")
    w_gt.write_capture([UNIT_WRONG_1])
    r_gt_err = w_gt.run_watchdog("18:30", [make_row()])
    assert r_gt_err["exception"] == "ValueError" and r_gt_err["fetches"] == []

    anchor_pitch = datetime(2027, 3, 24, 19, 40, tzinfo=ZoneInfo("America/New_York"))
    for idx, bad_gt in enumerate((None, "", "not-iso", 12345)):
        w_fb = ReviewWorld(root / f"txt_bad_gt_fb_{idx}")
        w_fb.write_pick_and_decision(game_time=bad_gt)
        w_fb.write_capture([UNIT_WRONG_1])
        r_fb = w_fb.run_watchdog(
            "18:30",
            [make_row()],
            extra_patches=[("bts.scheduler._earliest_pick_game_et", lambda _d: anchor_pitch)],
        )
        expected_no_pitch = "pick Batter 1010: recommended game 101, the entry is recorded in game 102"
        assert r_fb["dm_sent"] == [FIRST_ALERT.format(legs=expected_no_pitch)], (bad_gt, r_fb["dm_sent"])
        assert_no_unsafe_words(r_fb["dm_sent"][0])
    verified.append("unreadable_game_time_both_paths")

    # 4. locked_unconfirmed with path containing spaces (shlex.quote) + double with 1 affected leg
    w_lu = ReviewWorld(root / "dir with spaces" / "lu_case")
    w_lu.write_pick_and_decision(
        batter_name="Aaron Judge",
        dd=(1020, 201, DEFAULT_PITCH, "Juan Soto"),
        delivery="locked_unconfirmed",
    )
    w_lu.write_capture([UNIT_RIGHT_1, UNIT_WRONG_2])
    r_lu = w_lu.run_watchdog("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)])
    expected_lu = LOCKED_UNCONFIRMED_ALERT.format(
        names="Aaron Judge + DD Juan Soto",
        legs="double-down Juan Soto: recommended game 201 (first pitch 7:40 PM ET), the entry is recorded in game 202",
        qdir=quote(str(w_lu.picks_dir)),
    )
    assert r_lu["dm_sent"] == [expected_lu]
    assert "'" in quote(str(w_lu.picks_dir))
    assert_no_unsafe_words(r_lu["dm_sent"][0])
    verified.append("locked_unconfirmed_quoted_path_and_double")

    # 5. Case (b) with NO wrong_game leg across all three DM templates (First alert, Reminder, locked_unconfirmed)
    w_cb = ReviewWorld(root / "txt_case_b_first")
    w_cb.write_pick_and_decision(dd=(1020, 201, DEFAULT_PITCH, "Juan Soto"))
    w_cb.write_capture([UNIT_WRONG_1, UNIT_RIGHT_2], "20270324T200000Z")
    assert w_cb.run_watchdog("18:30", [make_row(77, 1, 1), make_row(78, 2, 2)], dm_mode="failure")["marker"]["status"] == "dm_failed"
    w_cb.write_capture([UNIT_RIGHT_1, UNIT_RIGHT_2], "20270324T210000Z")
    r_cb1 = w_cb.run_watchdog("18:45", [make_row(77, 1, 1), make_row(78, 2, 2)])
    leg_cb = "pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET)"
    assert r_cb1["dm_sent"] == [FIRST_ALERT.format(legs=leg_cb)]
    r_cb30 = w_cb.run_watchdog("19:05", [make_row(77, 1, 1), make_row(78, 2, 2)])
    assert r_cb30["dm_sent"] == [REMINDER_ALERT.format(legs=leg_cb)]
    w_cb.write_pick_and_decision(dd=(1020, 201, DEFAULT_PITCH, "Juan Soto"), delivery="locked_unconfirmed")
    r_cb15 = w_cb.run_watchdog("19:20", [make_row(77, 1, 1), make_row(78, 2, 2)])
    assert r_cb15["dm_sent"] == [LOCKED_UNCONFIRMED_ALERT.format(
        names="Batter 1010 + DD Juan Soto",
        legs=leg_cb,
        qdir=quote(str(w_cb.picks_dir)),
    )]
    verified.append("case_b_no_wrong_game_leg_all_three_templates")

    return {"text_families_verified": len(verified), "names": verified}


def bad_marker_worker(out_file: Path):
    res = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label, raw in [
            ("json_null", b"null"),
            ("json_list", b"[]"),
            ("invalid_utf8", b"\xff\xfe"),
            ("int_escalations", json.dumps({"date": DEFAULT_DAY, "status": "alerted", "escalations": 42}).encode()),
            ("invalid_json_brace", b"{"),
        ]:
            w = ReviewWorld(root / label)
            w.write_pick_and_decision()
            w.write_capture([UNIT_WRONG_1])
            w.set_marker(raw_bytes=raw)
            r = w.run_watchdog("18:30", [make_row()])
            res[label] = {"exit_code": r["exit_code"], "exception": r["exception"], "marker_status": (r["marker"] or {}).get("status")}

        w_dir = ReviewWorld(root / "marker_is_dir")
        w_dir.write_pick_and_decision()
        w_dir.write_capture([UNIT_WRONG_1])
        w_dir.marker_path.parent.mkdir(parents=True, exist_ok=True)
        w_dir.marker_path.mkdir()
        (w_dir.marker_path / "keep").write_text("x")
        r_dir = w_dir.run_watchdog("18:30", [])
        res["marker_is_nonempty_dir_no_pick"] = {
            "exception": r_dir["exception"],
            "dm_sent_count": len(r_dir["dm_sent"]),
            "receipt_marker_status": r_dir["receipts"][0]["marker_status"] if r_dir["receipts"] else None,
            "receipt_detail": r_dir["receipts"][0]["detail"] if r_dir["receipts"] else None,
        }
    out_file.write_text(json.dumps(res))


def fifo_child_worker(world_root: Path, mode: str):
    w = ReviewWorld(world_root)
    sent_flag = world_root / "dm_sent.flag"
    w.run_watchdog(
        "18:30",
        [make_row()] if mode == "match" else [],
        extra_patches=[("bts.dm.send_dm", lambda _r, _m: sent_flag.write_text("1"))],
    )


def check_item6_standing(root: Path) -> dict:
    standing = {}

    def gz(obj):
        return gzip.compress(json.dumps(obj).encode("utf-8"), mtime=0)

    capture_cases = [
        ("not_gzip", b"plain bytes", "BadGzipFile"),
        ("truncated_gzip", gz({"units": [UNIT_WRONG_1]})[:12], "EOFError"),
        ("invalid_utf8", gzip.compress(b"\xff"), "UnicodeDecodeError"),
        ("invalid_json", gzip.compress(b"{bad"), "JSONDecodeError"),
        ("top_list", gz([]), "AttributeError"),
        ("top_null", gz(None), "AttributeError"),
        ("top_int", gz(7), "AttributeError"),
        ("units_int", gz({"units": 1}), "TypeError"),
        ("units_bool_true", gz({"units": True}), "TypeError"),
        ("directory_at_capture", "DIR", "IsADirectoryError"),
        ("dangling_symlink", "DANGLING", "FileNotFoundError"),
        ("chmod_000", "CHMOD000", "PermissionError"),
        ("units_absent", gz({}), None),
        ("units_null", gz({"units": None}), None),
        ("units_false", gz({"units": False}), None),
        ("units_empty_str", gz({"units": ""}), None),
        ("units_dict", gz({"units": {"id": 77, "feedId": 102, "roundId": 1}}), None),
        ("units_str", gz({"units": "xyz"}), None),
        ("units_bad_elements", gz({"units": [None, False, 5, "s", [], {"id": "77", "feedId": 102, "roundId": 1}, {"id": True, "feedId": 102, "roundId": 1}]}), None),
    ]
    cap_verified = 0
    for name, payload, expected_exc in capture_cases:
        w = ReviewWorld(root / f"cap_{name}")
        w.write_pick_and_decision()
        cpath = w.units_dir / "20270324T210000Z.json.gz"
        if payload == "DIR":
            cpath.mkdir()
        elif payload == "DANGLING":
            cpath.symlink_to(w.root / "nonexistent.json.gz")
        elif payload == "CHMOD000":
            cpath.write_bytes(gz({"units": [UNIT_WRONG_1]}))
            cpath.chmod(0)
        else:
            cpath.write_bytes(payload)
        try:
            r = w.run_watchdog("18:30", [make_row()])
        finally:
            if payload == "CHMOD000":
                cpath.chmod(0o600)
        if expected_exc is not None:
            assert r["exit_code"] == 0, (name, r)
            assert r["marker"]["status"] == "confirmed" and r["marker"]["reason"] == "match_game_unchecked", (name, r)
            assert r["stderr"] == (
                f"check-pick-entered: game check unavailable ({expected_exc}); batter match only\n"
                f"check-pick-entered: entry receipt unavailable ({expected_exc})\n"
            ), (name, r["stderr"])
            assert r["stdout"] == f"check-pick-entered: {DEFAULT_DAY} batter entered; the game could not be checked (match_game_unchecked)\n"
            assert r["receipts"] == []
        else:
            assert r["exit_code"] == 0, (name, r)
            assert r["marker"]["status"] == "confirmed" and r["marker"]["reason"] == "match", (name, r)
            assert r["stderr"] == "", (name, r["stderr"])
            assert len(r["receipts"]) == 1 and r["receipts"][0]["qualification"]["slots"][0]["state"] == "unit_unverified"
        cap_verified += 1
    standing["malformed_captures_verified"] = cap_verified

    fifo_results = {}
    for link in (False, True):
        for mode in ("no_pick", "match"):
            tag = f"{'sym_' if link else ''}fifo_{mode}"
            w_f = ReviewWorld(root / tag)
            w_f.write_pick_and_decision()
            cpath = w_f.units_dir / "20270324T210000Z.json.gz"
            if link:
                real_fifo = w_f.root / "real.fifo"
                os.mkfifo(real_fifo)
                cpath.symlink_to(real_fifo)
            else:
                os.mkfifo(cpath)
            env = dict(os.environ, PYTHONPATH=f"{HEAD_SRC}:{HERE}", TZ="America/New_York")
            timed_out = False
            try:
                subprocess.run(
                    [sys.executable, __file__, "--fifo-child", str(w_f.root), mode],
                    env=env,
                    timeout=1.5,
                    capture_output=True,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
            assert timed_out, f"Expected FIFO timeout for {tag}"
            m = w_f.read_marker()
            dm_fired = (w_f.root / "dm_sent.flag").exists()
            if mode == "no_pick":
                assert dm_fired and m is not None and m["status"] == "alerted" and m["reason"] == "no_pick"
            else:
                assert not dm_fired and m is None
            fifo_results[tag] = {"timed_out": timed_out, "dm_fired": dm_fired, "marker": m["status"] if m else None}
    standing["fifo_checks"] = fifo_results

    for exc_cls, expected_detail, expected_code in [
        (KeyboardInterrupt, "raised KeyboardInterrupt", None),
        (SystemExit, None, 19),
    ]:
        w_i = ReviewWorld(root / f"int_{exc_cls.__name__}")
        w_i.write_pick_and_decision()
        w_i.write_capture([UNIT_WRONG_1])
        w_i.run_watchdog("18:26", [])
        prior_before = w_i.read_marker()
        real_lu = entry_receipt.load_units
        calls = []

        def fault_once(*a, _cls=exc_cls, _real=real_lu):
            calls.append(1)
            if len(calls) == 1:
                raise (_cls(19) if _cls is SystemExit else _cls())
            return _real(*a)

        r_i = w_i.run_watchdog("18:30", [make_row()], extra_patches=[("bts.entry_receipt.load_units", fault_once)])
        assert len(calls) == 2
        assert w_i.read_marker() == prior_before
        assert r_i["dm_attempted"] == []
        new_rec = [x for x in r_i["receipts"] if x["attempt_id"] != prior_before["receipt"]][0]
        assert new_rec["detail"] == expected_detail and new_rec["exit_code"] == expected_code

    w_ib = ReviewWorld(root / "int_both")
    w_ib.write_pick_and_decision()
    w_ib.write_capture([UNIT_WRONG_1])
    r_ib = w_ib.run_watchdog(
        "18:30",
        [make_row()],
        extra_patches=[("bts.entry_receipt.load_units", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))],
    )
    assert r_ib["marker"] is None and r_ib["dm_attempted"] == [] and r_ib["receipts"] == []

    w_im = ReviewWorld(root / "int_after_dm")
    w_im.write_pick_and_decision()
    w_im.write_capture([UNIT_WRONG_1])
    r_im1 = w_im.run_watchdog(
        "18:30",
        [make_row()],
        extra_patches=[("bts.cli._atomic_write_json", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))],
    )
    assert len(r_im1["dm_sent"]) == 1 and r_im1["marker"] is None
    r_im2 = w_im.run_watchdog("18:31", [make_row()])
    assert r_im2["dm_sent"] == r_im1["dm_sent"] and r_im2["marker"]["escalations"] == ["wrong_game"]
    standing["interrupt_checks"] = "pass"

    w_m1 = ReviewWorld(root / "midnight", "2027-07-04")
    w_m2 = ReviewWorld(root / "midnight", "2027-07-05")
    w_m1.write_pick_and_decision(game_time="2027-07-05T00:50:00-04:00")
    w_m2.write_pick_and_decision(game_time="2027-07-05T00:50:00-04:00")
    w_m1.write_capture([UNIT_WRONG_1, {"id": 78, "roundId": 2, "feedId": 101}], "20270704T235000Z")
    rounds_mid = {1: date(2027, 7, 4), 2: date(2027, 7, 5)}
    rm_2359 = w_m1.run_watchdog("23:59", [make_row(77, 1, 1)], rounds=rounds_mid, elapsed_sec=120.0)
    assert rm_2359["marker"]["date"] == "2027-07-04" and rm_2359["marker"]["reason"] == "wrong_game"
    rm_0005 = w_m2.run_watchdog("00:05", [make_row(77, 1, 1), make_row(78, 1, 1, rnd=2)], rounds=rounds_mid)
    assert rm_0005["marker"]["date"] == "2027-07-05" and rm_0005["marker"]["status"] == "confirmed"

    w_y1 = ReviewWorld(root / "year_a66", "2027-12-31")
    w_y2 = ReviewWorld(root / "year_a66", "2028-01-01")
    w_y1.write_pick_and_decision(game_time="2028-01-01T00:50:00-05:00")
    w_y2.write_pick_and_decision(game_time="2028-01-01T00:50:00-05:00")
    w_y1.write_capture([UNIT_WRONG_1, {"id": 78, "roundId": 2, "feedId": 102}], "20280101T041500Z")
    rounds_yr = {1: date(2027, 12, 31), 2: date(2028, 1, 1)}
    ry_dec31 = w_y1.run_watchdog("23:50", [make_row(77, 1, 1)], rounds=rounds_yr)
    assert ry_dec31["exit_code"] == 0 and ry_dec31["receipts"][0]["qualification"]["slots"][0]["state"] == "unit_unverified"
    ry_jan01 = w_y2.run_watchdog("00:05", [make_row(78, 1, 1, rnd=2)], rounds=rounds_yr)
    assert ry_jan01["exit_code"] == 1 and ry_jan01["marker"]["reason"] == "wrong_game"
    standing["midnight_and_year_boundary"] = "pass"

    b_m = root / "bad_markers_base.json"
    h_m = root / "bad_markers_head.json"
    for src_dir, out_f in ((BASE_SRC, b_m), (HEAD_SRC, h_m)):
        env = dict(os.environ, PYTHONPATH=f"{src_dir}:{HERE}", TZ="America/New_York")
        subprocess.run([sys.executable, __file__, "--marker-worker", str(out_f)], env=env, timeout=30, check=True)
    bm_data = json.loads(b_m.read_text())
    hm_data = json.loads(h_m.read_text())
    for k in ("json_null", "json_list", "invalid_utf8", "int_escalations", "marker_is_nonempty_dir_no_pick"):
        assert bm_data[k] == hm_data[k], (k, bm_data[k], hm_data[k])
    assert bm_data["invalid_json_brace"]["exception"] is None and hm_data["invalid_json_brace"]["exception"] is None
    assert hm_data["invalid_json_brace"]["marker_status"] == "alerted"

    w_m0 = ReviewWorld(root / "marker_chmod000")
    w_m0.write_pick_and_decision()
    w_m0.write_capture([UNIT_WRONG_1])
    w_m0.set_marker({"date": DEFAULT_DAY, "status": "confirmed", "reason": "match"})
    w_m0.marker_path.chmod(0)
    r_m0 = w_m0.run_watchdog("18:30", [make_row()])
    assert r_m0["exit_code"] == 1 and r_m0["marker"]["status"] == "alerted" and r_m0["marker"]["reason"] == "wrong_game"
    standing["bad_markers_verified"] = list(hm_data.keys()) + ["chmod_000_marker"]

    return standing


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        r4 = check_item4_tiers(root)
        r5 = check_item5_texts(root)
        r6 = check_item6_standing(root)
    summary = {"item4_tiers": r4, "item5_texts": r5, "item6_standing": r6}
    (HERE / "probe_03_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--marker-worker":
        bad_marker_worker(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--fifo-child":
        fifo_child_worker(Path(sys.argv[2]), sys.argv[3])
    else:
        main()
