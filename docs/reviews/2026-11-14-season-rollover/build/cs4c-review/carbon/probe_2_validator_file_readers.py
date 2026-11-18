"""Adversarial Probe 2 (Items 3 & 4): Pure function, validator oracle, generative checks,
pick file encoding/re-saves, live-forward scripts, and all pick readers.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time as time_mod
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REVIEW_DIR))

from netguard import forbid_network

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
SCHEMA = "bts_pick_automatic_reading_v1"
KEYS = ("schema", "recorded_at", "age_hours", "stale", "reason")


# --- Independent reference specification from kickoff-cs4c.md prose ---

def _oracle_bounded_str(val: object) -> str | None:
    if not isinstance(val, str):
        return None
    if len(val) > 128:
        return None
    for ch in val:
        code = ord(ch)
        if code < 0x20 or code > 0x7E:
            return None
    return val


def _oracle_parse_utc(val: str | None) -> datetime | None:
    if val is None:
        return None
    try:
        dt = datetime.fromisoformat(val)
    except ValueError:
        return None
    if dt.utcoffset() is None:
        return None
    try:
        return dt.astimezone(timezone.utc)
    except OverflowError:
        return None


def oracle_automatic_reading_record(s, now: datetime) -> dict | None:
    if s.observation_kind != "automatic":
        return None
    rec = _oracle_bounded_str(s.contest_recorded_at)
    rec_utc = _oracle_parse_utc(rec)
    if rec_utc is None:
        return {
            "schema": "bts_pick_automatic_reading_v1",
            "recorded_at": rec,
            "age_hours": None,
            "stale": True,
            "reason": "unreadable",
        }
    now_utc = now.astimezone(timezone.utc)
    elapsed = now_utc - rec_utc
    if elapsed < timedelta(minutes=-5):
        reason = "future"
    elif elapsed > timedelta(hours=14):
        reason = "old"
    else:
        reason = None
    age_hours = round(elapsed.total_seconds() / 3600, 2)
    return {
        "schema": "bts_pick_automatic_reading_v1",
        "recorded_at": rec,
        "age_hours": age_hours,
        "stale": reason is not None,
        "reason": reason,
    }


def oracle_valid_automatic_reading(val: object) -> dict | None:
    if not isinstance(val, dict) or set(val.keys()) != set(KEYS):
        return None
    if val["schema"] != SCHEMA:
        return None
    rec = val["recorded_at"]
    if rec is not None and _oracle_bounded_str(rec) is None:
        return None
    stale = val["stale"]
    if type(stale) is not bool:
        return None
    reason = val["reason"]
    if reason is not None and (type(reason) is not str or reason not in ("old", "future", "unreadable")):
        return None
    if stale != (reason is not None):
        return None
    age = val["age_hours"]
    usable = _oracle_parse_utc(rec) is not None
    if reason == "unreadable":
        if age is not None or usable:
            return None
    else:
        if not usable:
            return None
        if type(age) is not float or not math.isfinite(age):
            return None
        if abs(age) > 87_660_000:
            return None
        if round(age, 2) != age:
            return None
        if reason == "old" and not (age >= 14.0):
            return None
        if reason == "future" and not (age <= -0.08):
            return None
        if reason is None and not (-0.08 <= age <= 14.0):
            return None
    return {k: val[k] for k in KEYS}


def _audit_child_worker() -> None:
    """Runs in an isolated child process with sys.addaudithook so no process-wide hook leaks."""
    from bts import contest_state

    st = contest_state.DecisionStreakState(
        7, False, True, "contest", "fresh", 0, False,
        observation_kind="automatic", contest_recorded_at="2027-07-14T22:35:00-04:00",
    )
    events = []

    def hook(event, args):
        if event in {"open", "os.listdir", "os.scandir", "os.stat"}:
            events.append((event, args))

    class ForbiddenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            raise AssertionError("datetime.now called")

        @classmethod
        def utcnow(cls):
            raise AssertionError("datetime.utcnow called")

        @classmethod
        def today(cls):
            raise AssertionError("datetime.today called")

    sys.addaudithook(hook)
    with patch.object(contest_state, "datetime", ForbiddenClock), patch.object(
        time_mod, "time", Mock(side_effect=AssertionError("time.time called"))
    ):
        for k in ("automatic", "dated", "legacy_manual", "model"):
            res = contest_state.automatic_reading_record(replace(st, observation_kind=k), NOW)
            if k == "automatic":
                assert res is not None and contest_state.valid_automatic_reading(res) == res
            else:
                assert res is None
        for bad in (None, "", "bad", "2027-07-15\n18:35Z", "x" * 129, 42):
            res = contest_state.automatic_reading_record(
                replace(st, contest_recorded_at=bad), NOW  # type: ignore[arg-type]
            )
            assert contest_state.valid_automatic_reading(res) == res
    assert events == [], f"Unexpected I/O events: {events}"


def test_pure_function_and_validator() -> None:
    from bts import contest_state, picks

    assert contest_state.AUTOMATIC_READING_STALE_HOURS == 14

    # Audit hook check in isolated child subprocess
    env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--audit-child"],
        env=env, check=True,
    )

    base_state = contest_state.DecisionStreakState(7, False, True, "contest", "fresh", 0, False)

    # 1. Kinds, boundaries (±1us), grey zones, Z/z/offsets, DST, bad strings, non-strings, extreme now
    test_pairs: list[tuple[object, object, datetime]] = []
    for k in ("automatic", "dated", "legacy_manual", "model", "custom"):
        test_pairs.append((k, "2027-07-14T22:35:00-04:00", NOW))

    # ±1 microsecond around -5 minutes (-300s) and +14 hours (+50400s), plus rounding grey zones
    now_utc = NOW.astimezone(timezone.utc)
    for sec, us in [
        (-300, -1), (-300, 0), (-300, 1),
        (50400, -1), (50400, 0), (50400, 1),
        (-306, 0), (-305, 0), (-289, 0), (-288, 0),
        (-1, 0), (0, 0), (1, 0),
        (50382, 0), (50383, 0), (50417, 0), (50418, 0),
    ]:
        inst = now_utc - timedelta(seconds=sec, microseconds=us)
        test_pairs.append(("automatic", inst.isoformat().replace("+00:00", "Z"), NOW))
        test_pairs.append(("automatic", inst.isoformat().replace("+00:00", "z"), NOW))
        test_pairs.append(("automatic", inst.astimezone(ET).isoformat(), NOW))

    # Offsets & sub-minute offsets & fractional seconds
    for ts in [
        "2027-07-15T22:35:00+00:00",
        "2027-07-15T22:35:00-00:00",
        "2027-07-16T04:05:00+05:30",
        "2027-07-16T12:35:00+14:00",
        "2027-07-14T22:36:00-23:59",
        "2027-07-15T22:35:30+00:00:30",
        "2027-07-15T22:35:00.123Z",
        "2027-07-15T22:35:00.123456Z",
    ]:
        test_pairs.append(("automatic", ts, NOW))

    # DST spring-forward and fall-back instants
    test_pairs.extend([
        ("automatic", "2027-03-13T13:30:00-05:00", datetime(2027, 3, 14, 3, 30, tzinfo=ET)),
        ("automatic", "2027-03-13T13:30:00-05:00", datetime(2027, 3, 14, 4, 30, 0, 1, tzinfo=ET)),
        ("automatic", "2027-11-06T13:30:00-04:00", datetime(2027, 11, 7, 2, 30, tzinfo=ET)),
        ("automatic", "2027-11-06T13:30:00-04:00", datetime(2027, 11, 7, 3, 30, tzinfo=ET)),
        ("automatic", "2027-11-07T01:30:00-04:00", datetime(2027, 11, 7, 1, 30, tzinfo=ET, fold=0)),
        ("automatic", "2027-11-07T01:30:00-04:00", datetime(2027, 11, 7, 1, 30, tzinfo=ET, fold=1)),
    ])

    # Bad strings: empty, spaces, naive, non-ISO, 128 vs 129, non-ASCII, all control chars, \n, \ud800, UTC overflow
    bad_strings: list[object] = [
        None, "", "   ",
        "2027-07-15T18:35:00", "2027-07-15",
        "not-iso", "2027-02-29T12:00:00Z", "2027-07-15T25:00:00Z", "2027-07-15T18:35:00+24:00",
        "x" * 128, " " * 128, "~" * 128,
        "x" * 129, "2027-07-15T18:35:00Z" + (" " * 109),
        "2027-07-15T18:35:00Zé", "2027-07-15T18:35:00Z\u0080",
        "2027-07-15\n18:35:00Z", "2027-07-15\r18:35:00Z", "2027-07-15\t18:35:00Z",
        "2027-07-15\ud80018:35:00Z", "\ud800",
        "0001-01-01T00:00:00+00:01", "9999-12-31T23:59:59-00:01",
        "0001-01-01T00:00:00Z", "9999-12-31T23:59:59.999999Z",
        0, 1, -1, 3.14, True, False, [], {}, b"2027-07-15T18:35:00Z",
    ]
    for c in list(range(0x00, 0x20)) + [0x7F]:
        bad_strings.append(f"2027-07-15T18:35:00Z{chr(c)}")
    for bs in bad_strings:
        test_pairs.append(("automatic", bs, NOW))

    # Extreme valid UTC-representable `now`
    min_utc = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    max_utc = datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
    test_pairs.extend([
        ("automatic", "9999-12-31T23:59:59.999999Z", min_utc),
        ("automatic", "0001-01-01T00:00:00Z", max_utc),
        ("automatic", "0001-01-01T00:00:00Z", min_utc),
        ("automatic", "9999-12-31T23:59:59.999999Z", max_utc),
    ])

    for kind, rec_in, now_in in test_pairs:
        st = replace(base_state, observation_kind=str(kind), contest_recorded_at=rec_in)  # type: ignore[arg-type]
        got = contest_state.automatic_reading_record(st, now_in)
        exp = oracle_automatic_reading_record(st, now_in)
        assert got == exp, f"Mismatch for {(kind, rec_in, now_in)}: {got} != {exp}"
        if got is not None:
            assert tuple(got.keys()) == KEYS
            assert type(got["stale"]) is bool
            if got["age_hours"] is not None:
                assert type(got["age_hours"]) is float
            assert contest_state.valid_automatic_reading(got) == got

    # Check extreme overflowing `now` (outside caller precondition) raises OverflowError cleanly
    bad_now = datetime(9999, 12, 31, 23, 59, tzinfo=timezone(timedelta(hours=-1)))
    try:
        contest_state.automatic_reading_record(
            replace(base_state, observation_kind="automatic", contest_recorded_at="2027-07-15T00:00:00Z"),
            bad_now,
        )
        raise AssertionError("Expected OverflowError on non-UTC-representable now")
    except OverflowError:
        pass

    # 2. Exhaustive validator matrix (valid + invalid, including unmasked cases!)
    good_iso = "2027-07-14T22:35:00-04:00"
    base_old = {"schema": SCHEMA, "recorded_at": good_iso, "age_hours": 20.0, "stale": True, "reason": "old"}
    base_fresh = {"schema": SCHEMA, "recorded_at": good_iso, "age_hours": 5.0, "stale": False, "reason": None}
    base_fut = {"schema": SCHEMA, "recorded_at": good_iso, "age_hours": -1.0, "stale": True, "reason": "future"}
    base_unr = {"schema": SCHEMA, "recorded_at": "bad", "age_hours": None, "stale": True, "reason": "unreadable"}

    candidates: list[object] = [
        # Valid shapes
        base_old, base_fresh, base_fut, base_unr,
        dict(base_unr, recorded_at=None),
        dict(base_unr, recorded_at=""),
        dict(base_unr, recorded_at="2027-07-15T18:35:00"),  # naive
        dict(base_unr, recorded_at="0001-01-01T00:00:00+00:01"),  # UTC underflow
        dict(base_unr, recorded_at="9999-12-31T23:59:59-00:01"),  # UTC overflow
        dict(base_unr, recorded_at="x" * 128),
        dict(base_old, age_hours=14.0),
        dict(base_fresh, age_hours=14.0),
        dict(base_fresh, age_hours=-0.08),
        dict(base_fut, age_hours=-0.08),
        dict(base_fresh, age_hours=0.0),
        dict(base_fresh, age_hours=-0.0),
        dict(base_old, age_hours=87_660_000.0),
        dict(base_fut, age_hours=-87_660_000.0),
        dict(base_old, recorded_at="2027-07-14T22:35:00Z"),
        # Invalid non-dicts & key sets
        None, True, False, 0, 1.5, "str", [], [base_old],
        *[({k: v for k, v in base_old.items() if k != m}) for m in KEYS],
        dict(base_old, extra=1),
        # Invalid schema
        dict(base_old, schema=None), dict(base_old, schema=1), dict(base_old, schema=[]),
        dict(base_old, schema={}), dict(base_old, schema="bts_pick_automatic_reading_v2"),
        # Unmasked invalid recorded_at on BOTH unreadable and old/fresh!
        dict(base_unr, recorded_at="x" * 129),
        dict(base_unr, recorded_at="\ud800"),
        dict(base_unr, recorded_at="bad\nline"),
        dict(base_unr, recorded_at="bad\x7f"),
        dict(base_unr, recorded_at="badé"),
        dict(base_unr, recorded_at=123),
        dict(base_unr, recorded_at=[]),
        dict(base_old, recorded_at="2027-07-14\n22:35:00-04:00"),
        dict(base_old, recorded_at="2027-07-14\ud80022:35:00-04:00"),
        dict(base_old, recorded_at="x" * 129),
        dict(base_old, recorded_at=None),
        dict(base_old, recorded_at="bad"),
        dict(base_old, recorded_at="2027-07-14T22:35:00"),  # naive with non-unreadable
        dict(base_old, recorded_at="0001-01-01T00:00:00+00:01"),  # overflow with non-unreadable
        dict(base_fresh, recorded_at=None),
        dict(base_fut, recorded_at="bad"),
        # Type swaps: bool for float (unmasked on fresh!), int for float, float for bool, int for bool
        dict(base_fresh, age_hours=True),
        dict(base_fresh, age_hours=False),
        dict(base_old, age_hours=True),
        dict(base_fut, age_hours=False),
        dict(base_fresh, age_hours=0),
        dict(base_fresh, age_hours=5),
        dict(base_old, age_hours=20),
        dict(base_fut, age_hours=-1),
        dict(base_old, stale=1),
        dict(base_old, stale=1.0),
        dict(base_fresh, stale=0),
        dict(base_fresh, stale=0.0),
        dict(base_old, stale="true"),
        dict(base_old, stale=[]),
        # Stale / reason / age inconsistencies
        dict(base_old, stale=False),
        dict(base_fresh, stale=True),
        dict(base_fut, stale=False),
        dict(base_unr, stale=False),
        dict(base_old, reason=None),
        dict(base_fresh, reason="old"),
        dict(base_old, reason="other"),
        dict(base_old, reason=1),
        dict(base_old, reason=["old"]),
        dict(base_old, reason={"k": "old"}),
        dict(base_unr, recorded_at=good_iso),  # usable time on unreadable
        dict(base_unr, age_hours=0.0),         # non-None age on unreadable
        dict(base_unr, age_hours=None, recorded_at=good_iso),
        dict(base_old, age_hours=None),
        dict(base_fresh, age_hours=None),
        dict(base_fut, age_hours=None),
        # Non-finite, out-of-bounds (+ and -), unrounded, boundary mismatches
        dict(base_old, age_hours=float("inf")),
        dict(base_old, age_hours=float("-inf")),
        dict(base_fut, age_hours=float("-inf")),
        dict(base_fresh, age_hours=float("nan")),
        dict(base_old, age_hours=float("nan")),
        dict(base_old, age_hours=87_660_000.01),
        dict(base_fut, age_hours=-87_660_000.01),
        dict(base_old, age_hours=1e20),
        dict(base_fut, age_hours=-1e20),
        dict(base_fresh, age_hours=0.001),
        dict(base_old, age_hours=14.001),
        dict(base_fut, age_hours=-0.085),
        dict(base_old, age_hours=13.99),
        dict(base_old, age_hours=0.0),
        dict(base_old, age_hours=-0.08),
        dict(base_fut, age_hours=-0.07),
        dict(base_fut, age_hours=0.0),
        dict(base_fut, age_hours=14.0),
        dict(base_fresh, age_hours=-0.09),
        dict(base_fresh, age_hours=14.01),
    ]

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                        0.82, [], False, 101, f"{DAY}T23:10:00Z")
        dp_template = asdict(picks.DailyPick(DAY, NOW.isoformat(), p1, None, None))
        shadow_min_template = {
            "date": DAY, "run_time": NOW.isoformat(), "pick": dp_template["pick"],
            # Deliberately omit double_down and runner_up to test shadow loader tolerance!
        }

        for idx, cand in enumerate(candidates):
            shuffled = (
                dict(reversed(list(cand.items()))) if isinstance(cand, dict) else cand
            )
            got = contest_state.valid_automatic_reading(shuffled)
            exp = oracle_valid_automatic_reading(shuffled)
            assert got == exp, f"Validator mismatch #{idx} ({cand!r}): {got} != {exp}"
            if got is not None:
                assert tuple(got.keys()) == KEYS

            # Test through load_pick, load_pick_bytes, and load_shadow_pick (when JSON-encodable)
            body = dict(dp_template, automatic_reading=shuffled)
            sbody = dict(shadow_min_template, automatic_reading=shuffled)
            raw = json.dumps(body).encode("utf-8")
            sraw = json.dumps(sbody).encode("utf-8")
            (tdir / f"{DAY}.json").write_bytes(raw)
            (tdir / f"{DAY}.shadow.json").write_bytes(sraw)

            lp = picks.load_pick(DAY, tdir)
            lpb, held = picks.load_pick_bytes(DAY, tdir)
            lsp = picks.load_shadow_pick(DAY, tdir)
            assert held == raw
            for obj in (lp, lpb, lsp):
                assert obj is not None
                assert obj.automatic_reading == exp
                if exp is not None:
                    assert tuple(obj.automatic_reading.keys()) == KEYS

    # 3. Generative test: 2,500 random records from automatic_reading_record + JSON round-trip
    rng = random.Random(20261118)
    for _ in range(2500):
        year = rng.choice([1, 2, 1970, 2026, 2027, 2028, 9998, 9999])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        sec = rng.randint(0, 59)
        us = rng.randint(0, 999999)
        off_min = rng.randint(-1439, 1439)
        tz = timezone(timedelta(minutes=off_min))
        dt = datetime(year, month, day, hour, minute, sec, us, tzinfo=tz)
        raw_ts = dt.isoformat()
        if off_min == 0 and rng.random() < 0.5:
            raw_ts = raw_ts.replace("+00:00", "Z")
        now_y = rng.choice([1, 2027, 9999])
        now_dt = datetime(now_y, rng.randint(1, 12), rng.randint(1, 28),
                          rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59),
                          rng.randint(0, 999999), tzinfo=timezone.utc)
        st = replace(base_state, observation_kind="automatic", contest_recorded_at=raw_ts)
        rec = contest_state.automatic_reading_record(st, now_dt)
        assert rec == oracle_automatic_reading_record(st, now_dt)
        assert contest_state.valid_automatic_reading(rec) == rec
        decoded = json.loads(json.dumps(rec))
        assert contest_state.valid_automatic_reading(decoded) == rec

    # IEEE-754 hundredths check around powers of 2 up to 87,660,000
    for exp_pow in range(0, 27):
        anchor = float(1 << exp_pow)
        if anchor > 87_660_000:
            break
        for delta_cents in range(-50, 51):
            val = round(anchor + delta_cents * 0.01, 2)
            if abs(val) <= 87_660_000:
                assert round(val, 2) == val

    print(
        f"[PASS] Pure function ({len(test_pairs)} edge cases + child audit hook), "
        f"validator ({len(candidates)} shapes across all 3 loaders), and 2,500 generative cases passed."
    )


def test_pick_file_resaves_live_forward_and_readers() -> None:
    import pandas as pd
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, dm, entry_receipt, orchestrator, picks, posting, scheduler, strategy, web
    from bts.experiment import artifacts as exp_artifacts
    from bts.health import (
        alert, fallback_defer, late_delivery, mdp_policy_alignment, pick_entry,
        pitcher_sparsity, post_failure, postponed_pick, predicted_vs_realized,
        projected_lineup, same_team_corr,
    )
    from bts.state import export as state_export, regenerate as state_regen
    from scripts import live_forward_capture_once as lf_capture
    from scripts import live_forward_provenance_inventory as lf_inventory
    from scripts.audit.season_ledger.sources import pick_files as ledger_picks

    old_rec = {
        "schema": SCHEMA,
        "recorded_at": "2027-07-14T22:35:00-04:00",
        "age_hours": 20.0,
        "stale": True,
        "reason": "old",
    }
    fresh_rec = {
        "schema": SCHEMA,
        "recorded_at": "2027-07-15T16:35:00-04:00",
        "age_hours": 2.0,
        "stale": False,
        "reason": None,
    }

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pdir = root / "picks"
        pdir.mkdir(parents=True)

        p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                        0.82, [], False, 101, f"{DAY}T23:10:00Z")
        dp = picks.attach_provenance(picks.DailyPick(DAY, NOW.isoformat(), p1, None, None))

        # 1. Pre-5a canonical file re-save exact bytes AND 5a canonical file re-save exact bytes
        full_dict = asdict(dp)
        pre5a_dict = {k: v for k, v in full_dict.items() if k not in {"chosen_on", "automatic_reading"}}
        cs5a_dict = {k: v for k, v in full_dict.items() if k != "automatic_reading"}

        raw_pre5a = json.dumps(pre5a_dict, indent=2).encode("utf-8")
        (pdir / f"{DAY}.json").write_bytes(raw_pre5a)
        loaded_pre5a = picks.load_pick(DAY, pdir)
        assert loaded_pre5a.chosen_on is None and loaded_pre5a.automatic_reading is None
        picks.save_pick(loaded_pre5a, pdir)
        assert (pdir / f"{DAY}.json").read_bytes() == (
            raw_pre5a[:-2] + b',\n  "chosen_on": null,\n  "automatic_reading": null\n}'
        )

        raw_5a = json.dumps(cs5a_dict, indent=2).encode("utf-8")
        (pdir / f"{DAY}.json").write_bytes(raw_5a)
        loaded_5a = picks.load_pick(DAY, pdir)
        assert loaded_5a.automatic_reading is None
        picks.save_pick(loaded_5a, pdir)
        assert (pdir / f"{DAY}.json").read_bytes() == (
            raw_5a[:-2] + b',\n  "automatic_reading": null\n}'
        )

        # Normal and shadow key order
        loaded_5a.automatic_reading = old_rec
        picks.save_pick(loaded_5a, pdir)
        picks.save_shadow_pick(loaded_5a, pdir)
        for fname in (f"{DAY}.json", f"{DAY}.shadow.json"):
            d = json.loads((pdir / fname).read_text())
            assert list(d)[-2:] == ["chosen_on", "automatic_reading"]
            assert d["automatic_reading"] == old_rec

        # 2. Survival through every named re-save via real functions!
        acct = pdir / "account_state"
        acct.mkdir(parents=True, exist_ok=True)
        (acct / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v3",
            "season": 2027, "effective_date": DAY,
            "active_streak": 7, "best_streak": 9, "saver_state": "used",
            "recorded_at": NOW.isoformat(),
            "expires_at": f"{DAY}T23:59:59.999999-04:00",
            "source": "manual_cli",
        }))
        st_dated = contest_state.load_decision_streak_state(pdir, now=NOW, require_contest_state=True)
        dp.chosen_on = picks.chosen_on_record(st_dated, required=True)
        dp.automatic_reading = old_rec

        # (a) attach_provenance + save_pick
        picks.attach_provenance(dp)
        picks.save_pick(dp, pdir)
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec

        # (b) optional attempt + confirmed DM save preserves seeded automatic_reading
        cfg_opt = {
            "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(root / "hb"),
                             "models_dir": str(root / "models")},
            "bluesky": {"dm_recipient": "operator.test"},
            "scheduler": {"contest_state_required": False, "pick_delivery": "dm",
                          "live_forward_capture_on_lock": False},
            "tiers": [],
        }
        s_state = scheduler.SchedulerState(
            date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
            pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        attempt_seen = []
        with patch.object(scheduler, "_now_et", lambda: NOW), patch.object(
            scheduler, "_delivery_game_statuses", lambda _d: {}
        ), patch.object(
            dm, "send_dm",
            lambda r, t: (attempt_seen.append(picks.load_pick(DAY, pdir).automatic_reading), "dm-1")[1],
        ), contextlib.redirect_stderr(io.StringIO()):
            assert scheduler._deliver_and_lock_pick(
                picks.load_pick(DAY, pdir), cfg_opt, pdir, s_state, DAY, "opt",
            ) is True
        assert attempt_seen == [old_rec]
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec

        # (c) already-delivered recovery save & private save
        with patch.object(scheduler, "_now_et", lambda: NOW), contextlib.redirect_stderr(io.StringIO()):
            assert scheduler._deliver_and_lock_pick(
                picks.load_pick(DAY, pdir), cfg_opt, pdir, s_state, DAY, "recov",
            ) is True
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec

        dp_priv = picks.load_pick(DAY, pdir)
        dp_priv.notification_sent = False
        dp_priv.notification_id = None
        dp_priv.delivery_attempted = False
        picks.save_pick(dp_priv, pdir)
        cfg_priv = dict(cfg_opt, scheduler={"contest_state_required": True, "pick_delivery": "private",
                                            "live_forward_capture_on_lock": False})
        with patch.object(scheduler, "_now_et", lambda: NOW), patch.object(
            scheduler, "_delivery_game_statuses", lambda _d: {}
        ), contextlib.redirect_stderr(io.StringIO()):
            assert scheduler._deliver_and_lock_pick(
                picks.load_pick(DAY, pdir), cfg_priv, pdir, s_state, DAY, "priv",
            ) is True
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec

        # (d) bts preview, bts run, bts orchestrate reuse saves (Warmup unlocked & In-Progress locked catch-up!)
        from bts.model import predict
        preds = pd.DataFrame([{
            "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
            "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
            "p_hit_pa": 0.24, "p_game_hit": 0.82, "flags": "", "projected_lineup": False,
            "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
        }])

        class PinDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

        cfg_live = {
            "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(root / "hb"),
                             "models_dir": str(root / "models")},
            "bluesky": {"dm_recipient": "operator.test"},
            "scheduler": {"contest_state_required": True, "pick_delivery": "public",
                          "live_forward_capture_on_lock": False},
            "tiers": [],
        }
        toml_path = root / "cfg.toml"
        toml_path.write_text(
            'tiers = []\n[orchestrator]\n'
            f'picks_dir = {json.dumps(str(pdir))}\n'
            f'heartbeat_path = {json.dumps(str(root / "hb"))}\n'
            f'models_dir = {json.dumps(str(root / "models"))}\n'
            '[bluesky]\ndm_recipient = "operator.test"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "public"\n'
        )

        for status_mode in ("Warmup", "In Progress"):
            st_map = {101: {
                "abstract": "P" if status_mode == "Warmup" else "L",
                "detailed": status_mode,
                "code": "PW" if status_mode == "Warmup" else "I",
            }}
            for cmd in ("preview", "run", "orchestrate"):
                # Reset posted flag so each command actually re-saves
                cur = picks.load_pick(DAY, pdir)
                cur.bluesky_posted = False
                cur.bluesky_uri = None
                cur.automatic_reading = old_rec
                picks.save_pick(cur, pdir)
                with patch.object(contest_state, "datetime", PinDT), patch.object(
                    cli, "_now_et", lambda: NOW
                ), patch.object(
                    predict, "run_pipeline", Mock(return_value=preds)
                ), patch.object(
                    orchestrator, "run_cascade", Mock(return_value=(preds, "offline"))
                ), patch.object(
                    picks, "get_game_statuses_detailed", lambda *_: st_map
                ), patch.object(
                    strategy, "get_game_statuses_detailed", lambda *_: st_map
                ), patch.object(
                    posting, "post_to_bluesky", Mock(return_value="at://cli")
                ), patch.object(
                    posting, "should_post_now", Mock(return_value=True)
                ):
                    cli_args = (
                        ["orchestrate", "--date", DAY, "--config", str(toml_path)]
                        if cmd == "orchestrate"
                        else [cmd, "--date", DAY, "--picks-dir", str(pdir), "--models-dir", str(root / "models")]
                    )
                    res = CliRunner().invoke(cli.cli, cli_args, catch_exceptions=False)
                    assert res.exit_code == 0, res.output
                after_cmd = picks.load_pick(DAY, pdir)
                assert after_cmd.automatic_reading == old_rec, f"{cmd} ({status_mode}) lost automatic_reading"
                if cmd != "preview":
                    assert after_cmd.bluesky_posted is True

        # (e) Grading: save_nonterminal_result, run_result_polling, cli check-results (incl shadow!), reconcile_results
        cur = picks.load_pick(DAY, pdir)
        cur.result = None
        cur.slot_results = None
        picks.save_pick(cur, pdir)
        picks.save_shadow_pick(cur, pdir)

        scheduler.save_nonterminal_result(DAY, pdir, "unresolved")
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec

        with patch.object(scheduler, "poll_game_result", lambda *_: "final"), patch.object(
            picks, "resolve_daily_slot_results", lambda *_: {"primary": "hit"}
        ), contextlib.redirect_stderr(io.StringIO()):
            assert scheduler.run_result_polling(101, DAY, pdir) == "final"
        assert picks.load_pick(DAY, pdir).result == "hit"
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec

        # CLI check-results on unscored pick + shadow pick
        cur = picks.load_pick(DAY, pdir)
        cur.result = None
        cur.slot_results = None
        picks.save_pick(cur, pdir)
        sh = picks.load_shadow_pick(DAY, pdir)
        sh.result = None
        sh.slot_results = None
        picks.save_shadow_pick(sh, pdir)
        with patch.object(picks, "resolve_daily_slot_results", lambda *_: {"primary": "hit"}), patch.object(
            cli, "resolve_daily_slot_results", lambda *_: {"primary": "hit"}, create=True
        ), patch("bts.posting.reply_to_bluesky", lambda *_: "at://reply"):
            cr = CliRunner().invoke(
                cli.cli, ["check-results", "--date", DAY, "--picks-dir", str(pdir), "--no-reply"],
                catch_exceptions=False,
            )
            assert cr.exit_code == 0, cr.output
        assert picks.load_pick(DAY, pdir).result == "hit"
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec
        assert picks.load_shadow_pick(DAY, pdir).result == "hit"
        assert picks.load_shadow_pick(DAY, pdir).automatic_reading == old_rec

        # reconcile_results flipping hit -> miss before 08:00 ET next morning
        with patch.object(picks, "resolve_daily_slot_results", lambda *_: {"primary": "miss"}):
            flips = picks.reconcile_results(
                pdir, lookback_days=2, clock=lambda: NOW + timedelta(hours=8)
            )
            assert len(flips) == 1
        assert picks.load_pick(DAY, pdir).result == "miss"
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec

        # (f) All 4 archive saves: deferred_fallback, refused_delivery (late), refused_delivery (void), stale_pick
        graded = picks.load_pick(DAY, pdir)
        with patch.object(scheduler, "_now_et", lambda: NOW), patch.object(
            alert, "send_dm", lambda *_: "a"
        ), contextlib.redirect_stderr(io.StringIO()):
            picks.save_pick(graded, pdir)
            a_def = scheduler._defer_pick_at_fallback(pdir, DAY, graded, reason="test")
            picks.save_pick(graded, pdir)
            scheduler._refuse_late_delivery(
                graded, cfg_live, pdir, s_state, DAY, "late", NOW - timedelta(minutes=5), NOW
            )
            a_late, = (pdir / DAY).glob("refused_delivery_*.json")
            a_late_data = json.loads(a_late.read_text())
            a_late.unlink()
            picks.save_pick(graded, pdir)
            scheduler._refuse_void_game_delivery(
                graded, cfg_live, pdir, s_state, DAY, "void", {101: "Postponed"}
            )
            a_void, = (pdir / DAY).glob("refused_delivery_*.json")
            a_void_data = json.loads(a_void.read_text())
            # stale_pick via _archive_and_remove_pick
            picks.save_pick(graded, pdir)
            a_stale = scheduler._archive_and_remove_pick(
                pdir, DAY, graded, prefix="stale_pick", key="stale_pick",
                at_key="staled_at", reason="committed_game_past_cutoff",
            )

        for arch_data, expected_last in [
            (json.loads(a_def.read_text()), "deferred_fallback"),
            (a_late_data, "refused_delivery"),
            (a_void_data, "refused_delivery"),
            (json.loads(a_stale.read_text()), "stale_pick"),
        ]:
            assert list(arch_data)[-3:] == ["chosen_on", "automatic_reading", expected_last]
            assert arch_data["automatic_reading"] == old_rec

        # 3. Both live-forward scripts
        base_doc = {k: v for k, v in asdict(graded).items() if k != "automatic_reading"}
        null_doc = dict(base_doc, automatic_reading=None)
        old_doc = dict(base_doc, automatic_reading=old_rec)
        fresh_doc = dict(base_doc, automatic_reading=fresh_rec)
        changed_old_doc = dict(base_doc, automatic_reading=dict(old_rec, age_hours=20.01))

        lf_dir = root / "lf_picks"
        lf_dir.mkdir()
        manifest_path = root / "manifest.json"
        pick_file = lf_dir / f"{DAY}.json"

        pairs = [
            (base_doc, null_doc, False),
            (null_doc, base_doc, False),
            (base_doc, old_doc, True),
            (null_doc, old_doc, True),
            (old_doc, base_doc, True),
            (old_doc, null_doc, True),
            (old_doc, fresh_doc, True),
            (old_doc, changed_old_doc, True),
        ]
        for d_before, d_after, expect_drift in pairs:
            assert (lf_capture.decision_snapshot(d_before) != lf_capture.decision_snapshot(d_after)) is expect_drift
            assert (lf_inventory._decision_snapshot(d_before) != lf_inventory._decision_snapshot(d_after)) is expect_drift

            raw_before = json.dumps(d_before, indent=2).encode("utf-8")
            raw_after = json.dumps(d_after, indent=2).encode("utf-8")
            sha_before = hashlib.sha256(raw_before).hexdigest()
            sha_after = hashlib.sha256(raw_after).hexdigest()
            assert sha_before != sha_after

            pick_file.write_bytes(raw_after)
            snap = {"present": True, "source_sha256": sha_before, "production_pick_json": d_before}
            manifest_path.write_text(json.dumps({"production_pick_snapshot": snap}))
            cap_state = lf_capture.existing_snapshot_state(
                manifest_path=manifest_path, pick_path=pick_file, expected_date=DAY,
            )
            inv_state = lf_inventory._current_pick_snapshot_summary(
                snapshot=snap, date_key=DAY, picks_dir=lf_dir,
            )
            assert cap_state["current_pick_sha256"] == inv_state["current_pick_sha256"] == sha_after
            assert cap_state["snapshot_matches_current_pick"] is False
            assert inv_state["matches_current_pick"] is False
            assert cap_state["snapshot_decision_matches_current_pick"] is (not expect_drift)
            assert inv_state["decision_matches_current_pick"] is (not expect_drift)

        # 4. All other pick readers across 4 variants (absent, null, fresh_rec, old_rec)
        variants = [
            ("absent", base_doc),
            ("null", null_doc),
            ("fresh", fresh_doc),
            ("stale", old_doc),
        ]
        projected_outputs = []
        for vname, vdoc in variants:
            vdir = root / f"readers_{vname}"
            vdir.mkdir()
            raw_v = json.dumps(vdoc, indent=2).encode("utf-8")
            vpath = vdir / f"{DAY}.json"
            vpath.write_bytes(raw_v)
            (vdir / f"{DAY}.shadow.json").write_bytes(raw_v)
            daily_decision.write_decision(
                DAY, vdir, action="single", source="heuristic",
                primary=scheduler._row_from_daily(graded.pick),
                delivery_status="delivered", scoreable=True, now=NOW,
            )

            # Readers
            r_dates = contest_state._resolved_pick_dates(vdir)
            r_latest = contest_state.latest_resolved_pick_date(vdir)
            r_gap = contest_state.resolved_pick_settlement_gap(vdir, date(2027, 7, 14))
            r_miss = contest_state._has_unconfirmed_miss(vdir, date(2027, 7, 14))
            r_replay = picks._replay_season_streak(vdir, 2027)
            r_hist = state_export._pick_to_historical(json.loads(raw_v))
            r_regen = state_regen._hist_to_pick_file(r_hist)
            assert "automatic_reading" not in r_hist and "automatic_reading" not in r_regen

            dp_loaded, held = picks.load_pick_bytes(DAY, vdir)
            rec_obj = entry_receipt.EntryReceipt(
                et_date=DAY, started_at=NOW, clock=lambda: NOW,
                expected_username=None, picks_dir=vdir,
            )
            rec_obj.selection(dp_loaded, held, None, None)
            rec_sel = dict(rec_obj.record()["selection"])
            assert rec_sel.pop("pick_file_sha256") == hashlib.sha256(raw_v).hexdigest()

            led_row, = ledger_picks.parse_pick_file(vpath.name, raw_v).rows
            assert led_row["content_sha256"] == hashlib.sha256(raw_v).hexdigest()
            assert json.loads(led_row["record_raw_json"]) == vdoc
            led_proj = {
                k: v for k, v in led_row.items()
                if k not in {"obs_id", "content_sha256", "record_raw_json"}
            }

            exp_snap = exp_artifacts.load_production_pick_snapshot(vpath, expected_date=DAY)
            assert exp_snap["source_sha256"] == hashlib.sha256(raw_v).hexdigest()
            exp_proj = {
                k: v for k, v in exp_snap.items()
                if k not in {"source_sha256", "production_pick_json"}
            }

            web_picks = web.load_all_picks(vdir)
            assert len(web_picks) == 1
            web_proj = {k: v for k, v in web_picks[0].items() if k != "automatic_reading"}

            today_d = date(2027, 7, 15)
            h_checks = {
                "post_failure": post_failure.check(vdir, today=today_d, now=NOW + timedelta(hours=5)),
                "late_delivery": late_delivery.check(vdir, today=today_d),
                "fallback_defer": fallback_defer.check(vdir, today=today_d),
                "postponed_pick": postponed_pick.check(vdir, today=today_d),
                "projected_lineup": projected_lineup.check(vdir, today=today_d),
                "same_team_corr": same_team_corr.check(vdir, today=today_d),
                "pitcher_sparsity": pitcher_sparsity.check(vdir, today=today_d),
                "predicted_vs_realized": predicted_vs_realized.check(vdir, today=today_d),
                "mdp_policy_alignment": mdp_policy_alignment.check(vdir, today=today_d),
            }

            projected_outputs.append({
                "r_dates": r_dates, "r_latest": r_latest, "r_gap": r_gap, "r_miss": r_miss,
                "r_replay": r_replay, "r_hist": r_hist, "r_regen": r_regen,
                "rec_sel": rec_sel, "led_proj": led_proj, "exp_proj": exp_proj,
                "web_proj": web_proj, "h_checks": h_checks,
            })

        for other in projected_outputs[1:]:
            assert other == projected_outputs[0]

    print(
        "[PASS] Pick file encoding (pre-5a & 5a exact bytes), all 11 re-save paths, "
        "both live-forward scripts, and 20+ pick readers verified."
    )


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--audit-child":
        _audit_child_worker()
        return

    with forbid_network(allow_loopback=False):
        test_pure_function_and_validator()
        test_pick_file_resaves_live_forward_and_readers()
    print("PROBE_2_ALL_PASSED")


if __name__ == "__main__":
    main()
