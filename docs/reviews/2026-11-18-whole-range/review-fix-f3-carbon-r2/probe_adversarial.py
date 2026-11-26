"""Fresh Carbon adversarial checks for F3 round 2: extreme dates/timestamps, DST, symlinks, partial reads, boundaries, parity, and R64."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
BASE_SRC = Path("/tmp/wr-f3-r2-scratch").resolve() / "base" / "src"
F3_SRC = REPO / "src"
ET = ZoneInfo("America/New_York")
DAY = "2027-04-02"


def full_snapshot(root: Path) -> dict[str, tuple[int, object]]:
    if not root.exists():
        return {}
    snap: dict[str, tuple[int, object]] = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        rel = str(p.relative_to(root))
        if stat.S_ISREG(st.st_mode):
            snap[rel] = (st.st_mode, p.read_bytes())
        elif stat.S_ISLNK(st.st_mode):
            snap[rel] = (st.st_mode, os.readlink(p))
        else:
            snap[rel] = (st.st_mode, None)
    return snap


def make_skip(date: str = DAY, streak: int = 30, saver: bool = False) -> dict:
    from tests.test_day_outcome import _decision
    return _decision("skip", scoreable=False) | {"date": date, "streak": streak, "saver_available": saver}


def make_dated(day: str = DAY, *, streak: int = 1, saver: str = "used", recorded_at: str | None = None, **over) -> dict:
    from bts import contest_state
    d = dt.date.fromisoformat(day)
    end = dt.datetime.combine(d, dt.time.max, ET).isoformat()
    base = {
        "schema_version": contest_state.DATED_MANUAL_SCHEMA,
        "season": d.year,
        "effective_date": day,
        "expires_at": end,
        "active_streak": streak,
        "best_streak": max(streak, 30),
        "saver_state": saver,
        "recorded_at": recorded_at or f"{day}T14:00:00Z",
        "source": "manual_cli",
    }
    return base | over


def make_switch(day: str = DAY, *, status: str = "revoked", revoked_at: str | None = None, **over) -> dict:
    base = {
        "schema_version": "bts_season_contract_v1",
        "season": int(day[:4]),
        "status": status,
        "activation_id": "a" * 32,
        "switched_at": f"{day}T12:00:00Z" if status == "automatic" else None,
        "revoked_at": f"{day}T20:00:00Z" if revoked_at is None and status == "revoked" else revoked_at,
    }
    return base | over


def invoke_cli(root: Path, *, day: str = DAY, decision: dict | bytes | None = None,
               intent: str = "enter", dm_recipient: str | None = "operator.bsky.social",
               schedule=({},), raw_config: str | None = None, create_picks: bool = True) -> dict:
    from click.testing import CliRunner
    from bts import cli, daily_decision, dm, scheduler

    picks = root / "picks"
    if create_picks:
        (picks / "account_state").mkdir(parents=True, exist_ok=True)
        if decision is not None:
            dp = daily_decision.decision_path(day, picks)
            dp.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(decision, bytes):
                dp.write_bytes(decision)
            else:
                dp.write_text(json.dumps(decision))
    cfg = root / "config.toml"
    if raw_config is not None:
        cfg.write_text(raw_config)
    else:
        delivery = "private" if intent == "research" else "dm"
        cfg.write_text(f'[scheduler]\nentry_intent = "{intent}"\npick_delivery = "{delivery}"\n')

    sends = []

    def fetch(_d):
        if isinstance(schedule, Exception):
            raise schedule
        return list(schedule)

    def send(r, m):
        sends.append([r, m])
        return "ok"

    args = ["day-outcome", "--date", day, "--config", str(cfg),
            "--picks-dir", str(picks), "--expected-username", "stonehengee"]
    if dm_recipient is not None:
        args.extend(["--dm-recipient", dm_recipient])

    before = full_snapshot(picks)
    with patch.object(scheduler, "fetch_schedule", side_effect=fetch), \
         patch.object(dm, "send_dm", side_effect=send):
        res = CliRunner().invoke(cli.cli, args)
    assert full_snapshot(picks) == before, f"writes detected in {picks}"
    return {
        "exit": res.exit_code,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "sends": sends,
        "exc": type(res.exception).__name__ if res.exception else None,
    }


def check_extreme_dates_and_iso_grammar() -> None:
    print("=== A1. Extreme effective_date in evidence & ISO-8601 grammar edges ===", flush=True)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Extreme effective_date (0001-01-01 and 9999-12-31) inside current/archive while day=2027-04-02
        extreme_other_days = [
            make_dated("9999-12-31", expires_at="9999-12-31T23:59:59.999999-05:00"),
            make_dated("9999-12-31", expires_at="9999-12-31T23:59:59.999999-14:00"),
            make_dated("9999-12-31", expires_at="9999-12-31T23:59:59.999999+14:00"),
            make_dated("0001-01-01", expires_at="0001-01-01T23:59:59.999999-04:56:02"),
            make_dated("0001-01-01", expires_at="0001-01-01T00:00:00+14:00"),
        ]
        for idx, doc in enumerate(extreme_other_days):
            sub = root / f"ext_other_{idx}"
            (sub / "picks/account_state").mkdir(parents=True)
            (sub / "picks/account_state/contest_streak.manual.json").write_text(json.dumps(doc))
            (sub / "picks/account_state/contest_streak.manual.revoked-x.json").write_text(json.dumps(doc))
            res = invoke_cli(sub, decision=make_skip())
            assert res["exit"] == 0 and json.loads(res["stdout"])["state"] == "deliberate_skip", (idx, res)

        # Unusual ISO-8601 timestamp spellings accepted or rejected consistently
        iso_cases = [
            ("lowercase_z", f"{DAY}T14:00:00z", True),
            ("compact_iso", "20270402T140000Z", True),
            ("space_sep", f"{DAY} 14:00:00-04:00", True),
            ("arbitrary_sep_with_Z", f"{DAY}x14:00:00Z", True),
            ("plus_as_date_time_sep_naive", f"{DAY}+14:00", False),  # naive -> rejected by validator
            ("plus_as_sep_with_Z", f"{DAY}+14:00Z", True),
            ("subminute_offset", f"{DAY}T14:00:00.123456+05:45:30.123456", True),
            ("date_only_Z", f"{DAY}Z", False),
        ]
        for label, stamp, should_be_valid in iso_cases:
            sub = root / f"iso_{label}"
            (sub / "picks/account_state").mkdir(parents=True)
            (sub / "picks/account_state/contest_streak.manual.json").write_text(
                json.dumps(make_dated(recorded_at=stamp))
            )
            res = invoke_cli(sub, decision=make_skip())
            expected_code = 1 if should_be_valid else 0
            assert res["exit"] == expected_code, (label, stamp, res)

            sub_sw = root / f"iso_sw_{label}"
            (sub_sw / "picks/account_state").mkdir(parents=True)
            (sub_sw / "picks/account_state/season_contract.json").write_text(
                json.dumps(make_switch(revoked_at=stamp))
            )
            res_sw = invoke_cli(sub_sw, decision=make_skip())
            # Note: subminute_offset +05:45:30 at 14:00 is 08:14:30Z = 04:14 ET on DAY (same day)
            assert res_sw["exit"] == expected_code, (label, stamp, res_sw)

    print("A1 passed: extreme effective_dates and ISO-8601 spellings handled without exception.")


def check_unhashable_and_hostile_fields() -> None:
    print("=== A2. Unhashable / hostile JSON field types and NaN/Infinity ===", flush=True)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        dated_corruptions = [
            {"schema_version": []},
            {"effective_date": []},
            {"effective_date": "2027-4-2"},
            {"season": True},
            {"season": {}},
            {"expires_at": []},
            {"active_streak": True},
            {"active_streak": []},
            {"active_streak": 401},
            {"best_streak": True},
            {"best_streak": []},
            {"saver_state": []},
            {"saver_state": {}},
            {"recorded_at": []},
            {"source": []},
            {"reason": []},
            {"reason": 123},
        ]
        for idx, patch_fields in enumerate(dated_corruptions):
            sub = root / f"dated_bad_{idx}"
            (sub / "picks/account_state").mkdir(parents=True)
            (sub / "picks/account_state/contest_streak.manual.json").write_text(
                json.dumps(make_dated() | patch_fields)
            )
            res = invoke_cli(sub, decision=make_skip())
            assert res["exit"] == 0 and json.loads(res["stdout"])["state"] == "deliberate_skip", (patch_fields, res)

        switch_corruptions = [
            {"schema_version": []},
            {"season": True},
            {"season": []},
            {"status": []},
            {"status": {}},
            {"activation_id": []},
            {"activation_id": "A" * 32},
            {"switched_at": []},
            {"revoked_at": []},
            {"revoked_at": None},
            {"note": []},
            # Valid automatic switch with overflowing switched_at and a same-day revoked_at carried over
            {"status": "automatic", "switched_at": "0001-01-01T00:00:00Z", "revoked_at": f"{DAY}T20:00:00Z"},
            {"status": "automatic", "switched_at": "9999-12-31T23:59:59-12:00", "revoked_at": f"{DAY}T20:00:00Z"},
        ]
        for idx, patch_fields in enumerate(switch_corruptions):
            sub = root / f"sw_bad_{idx}"
            (sub / "picks/account_state").mkdir(parents=True)
            (sub / "picks/account_state/season_contract.json").write_text(
                json.dumps(make_switch() | patch_fields)
            )
            res = invoke_cli(sub, decision=make_skip())
            assert res["exit"] == 0 and json.loads(res["stdout"])["state"] == "deliberate_skip", (patch_fields, res)

        # Non-standard JSON NaN / Infinity in extra fields
        sub_nan = root / "nan_extra"
        (sub_nan / "picks/account_state").mkdir(parents=True)
        raw_nan = json.dumps(make_dated(streak=5))[:-1] + ', "extra": NaN, "inf": Infinity}'
        (sub_nan / "picks/account_state/contest_streak.manual.json").write_text(raw_nan)
        res_nan = invoke_cli(sub_nan, decision=make_skip())
        assert res_nan["exit"] == 1 and "records streak 5" in json.loads(res_nan["stdout"])["detail"]

    print("A2 passed: all unhashable/boolean/overflow/NaN JSON inputs handled safely.")


def check_dst_and_saver_boundaries() -> None:
    print("=== A3. ET DST spring-forward / fall-back and saver/streak boundaries ===", flush=True)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Spring forward: 2027-03-14 (02:00 EST -> 03:00 EDT)
        # 2027-03-14T03:59:00Z = 2027-03-13 22:59 EST (not 03-14); 2027-03-14T04:59:00Z = 2027-03-13 23:59 EST;
        # 2027-03-14T05:00:00Z = 2027-03-14 00:00 EST; 2027-03-14T07:05:00Z = 2027-03-14 03:05 EDT!
        for label, stamp, exp_code, exp_time in [
            ("before_midnight_est", "2027-03-14T04:59:00Z", 0, None),
            ("at_midnight_est", "2027-03-14T05:00:00Z", 1, "00:00 ET"),
            ("after_spring_jump", "2027-03-14T07:05:00Z", 1, "03:05 ET"),
        ]:
            sub = root / f"dst_spring_{label}"
            (sub / "picks/account_state").mkdir(parents=True)
            (sub / "picks/account_state/season_contract.json").write_text(
                json.dumps(make_switch("2027-03-14", revoked_at=stamp))
            )
            res = invoke_cli(sub, day="2027-03-14", decision=make_skip("2027-03-14"))
            assert res["exit"] == exp_code, (label, res)
            if exp_time:
                assert exp_time in json.loads(res["stdout"])["detail"], res

        # Fall back: 2027-11-07 (01:30 EDT = -04:00 = 05:30Z, vs 01:30 EST = -05:00 = 06:30Z)
        sub_fall = root / "dst_fall"
        (sub_fall / "picks/account_state").mkdir(parents=True)
        (sub_fall / "picks/account_state/contest_streak.manual.revoked-edt.json").write_text(
            json.dumps(make_dated("2027-11-07", streak=2, recorded_at="2027-11-07T01:30:00-04:00"))
        )
        (sub_fall / "picks/account_state/contest_streak.manual.revoked-est.json").write_text(
            json.dumps(make_dated("2027-11-07", streak=7, recorded_at="2027-11-07T01:30:00-05:00"))
        )
        (sub_fall / "picks/account_state/season_contract.json").write_text(
            json.dumps(make_switch("2027-11-07", revoked_at="2027-11-07T06:30:00Z"))
        )
        res_fall = invoke_cli(sub_fall, day="2027-11-07", decision=make_skip("2027-11-07"))
        detail_fall = json.loads(res_fall["stdout"])["detail"]
        assert res_fall["exit"] == 1 and "records streak 7" in detail_fall and "at 01:30 ET" in detail_fall

        # Streak 0, 9, 10 policy_saver_available boundaries + omitted --dm-recipient
        saver_cases = [
            # (skip_streak, skip_saver, obs_streak, obs_saver_state, expect_review)
            (0, False, 0, "used", False),
            (0, False, 0, "not_earned", True),   # streak 0 < 10 -> not_earned implies available=True != False!
            (9, True, 9, "not_earned", False),   # 9 < 10 -> available=True == True
            (10, False, 10, "not_earned", False),  # 10 >= 10 -> available=False == False
            (10, True, 10, "not_earned", True),  # 10 >= 10 -> available=False != True
            (1000, False, 30, "used", True),     # large skip streak formatting
        ]
        for idx, (s_strk, s_sav, o_strk, o_sav, exp_rev) in enumerate(saver_cases):
            sub = root / f"sav_{idx}"
            (sub / "picks/account_state").mkdir(parents=True)
            (sub / "picks/account_state/contest_streak.manual.json").write_text(
                json.dumps(make_dated(streak=o_strk, saver=o_sav))
            )
            res = invoke_cli(sub, decision=make_skip(streak=s_strk, saver=s_sav), dm_recipient=None)
            assert res["exit"] == int(exp_rev) and res["sends"] == [], (idx, res)
            if exp_rev:
                assert json.loads(res["stdout"])["state"] == "skip_state_review"

        # Installed mismatch + revoke on D + NO matching archive => BOTH installed reason AND revocation reason!
        sub_both = root / "installed_plus_revoke"
        (sub_both / "picks/account_state").mkdir(parents=True)
        (sub_both / "picks/account_state/contest_streak.manual.json").write_text(json.dumps(make_dated(streak=4)))
        (sub_both / "picks/account_state/season_contract.json").write_text(json.dumps(make_switch()))
        res_both = invoke_cli(sub_both, decision=make_skip())
        det_both = json.loads(res_both["stdout"])["detail"]
        assert ("The installed dated observation for 2027-04-02 records streak 4" in det_both
                and "The season switch records a revocation on 2027-04-02 at 16:00 ET." in det_both), det_both

    print("A3 passed: DST transitions, streak=0/9/10 saver boundaries, no-recipient CLI, and installed+revoke verified.")


def check_filesystem_symlinks_and_partial_reads() -> None:
    print("=== A4. Filesystem, symlinks, account_state non-dir, and 7-byte partial reads ===", flush=True)
    from bts import day_outcome

    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        # account_state is a regular file (not a directory)
        sub_file = root / "acct_is_file"
        (sub_file / "picks" / DAY).mkdir(parents=True)
        (sub_file / "picks" / DAY / "decision.json").write_text(json.dumps(make_skip()))
        (sub_file / "picks" / "account_state").write_text("not a dir")
        res = invoke_cli(sub_file, create_picks=False)
        assert res["exit"] == 0 and json.loads(res["stdout"])["state"] == "deliberate_skip"

        # account_state is a FIFO
        sub_fifo_dir = root / "acct_is_fifo"
        (sub_fifo_dir / "picks" / DAY).mkdir(parents=True)
        (sub_fifo_dir / "picks" / DAY / "decision.json").write_text(json.dumps(make_skip()))
        os.mkfifo(sub_fifo_dir / "picks" / "account_state")
        res = invoke_cli(sub_fifo_dir, create_picks=False)
        assert res["exit"] == 0 and json.loads(res["stdout"])["state"] == "deliberate_skip"

        # Symlink loop, broken symlink, symlink to FIFO, symlink to dir, and symlink to valid file
        sub_sym = root / "symlinks"
        acct = sub_sym / "picks" / "account_state"
        acct.mkdir(parents=True)
        fifo_target = sub_sym / "ext.fifo"
        os.mkfifo(fifo_target)
        dir_target = sub_sym / "ext_dir"
        dir_target.mkdir()
        os.symlink(acct / "contest_streak.manual.revoked-loop2.json", acct / "contest_streak.manual.revoked-loop1.json")
        os.symlink(acct / "contest_streak.manual.revoked-loop1.json", acct / "contest_streak.manual.revoked-loop2.json")
        os.symlink(sub_sym / "missing.json", acct / "contest_streak.manual.json")
        os.symlink(fifo_target, acct / "season_contract.json")
        os.symlink(dir_target, acct / "contest_streak.manual.revoked-dir.json")
        res_sym = invoke_cli(sub_sym, decision=make_skip())
        assert res_sym["exit"] == 0 and json.loads(res_sym["stdout"])["state"] == "deliberate_skip"

        # Add a symlink to a valid differing archive alongside all the bad symlinks -> still alerts!
        valid_target = sub_sym / "real_archive.json"
        valid_target.write_text(json.dumps(make_dated(streak=8)))
        os.symlink(valid_target, acct / "contest_streak.manual.revoked-good.json")
        res_sym2 = invoke_cli(sub_sym, decision=make_skip())
        assert res_sym2["exit"] == 1 and "records streak 8" in json.loads(res_sym2["stdout"])["detail"]

        # gather() with str picks_dir vs Path picks_dir
        cfg_d = {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}
        o_str = day_outcome.gather(DAY, str(sub_sym / "picks"), cfg_d, expected_username="stonehengee",
                                   fetch_schedule=lambda _: [{}])
        o_path = day_outcome.gather(DAY, sub_sym / "picks", cfg_d, expected_username="stonehengee",
                                    fetch_schedule=lambda _: [{}])
        assert o_str == o_path and o_str.state == "skip_state_review"

        # Partial os.read returning at most 7 bytes per call at exact 65536 and 65537 bytes
        base_doc = make_dated(streak=6) | {"pad": ""}
        enc_head = json.dumps(base_doc, separators=(",", ":")).encode()
        for target_size, expect_parsed in [(65536, True), (65537, False)]:
            blob = enc_head[:-2] + b"p" * (target_size - len(enc_head)) + enc_head[-2:]
            assert len(blob) == target_size
            p = root / f"partial_{target_size}.json"
            p.write_bytes(blob)
            st_target = p.stat()
            real_read = os.read
            seen = []

            def tiny_read(fd, n):
                chunk = real_read(fd, min(n, 7))
                if os.path.samestat(os.fstat(fd), st_target):
                    seen.append(len(chunk))
                return chunk

            with patch.object(os, "read", side_effect=tiny_read):
                parsed = day_outcome._read_evidence(p)
            if expect_parsed:
                assert isinstance(parsed, dict) and parsed["active_streak"] == 6
                assert sum(seen) == 65536 and seen[-1] == 0
            else:
                assert parsed is None
                assert sum(seen) == 65537

    print("A4 passed: symlinks, non-dir account_state, str/Path picks_dir, and 7-byte partial reads verified.")


def run_extra_parity_child() -> None:
    import bts
    from tests.test_day_outcome import _decision
    print(f"bts.__file__={bts.__file__}")

    results = []
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases = [
            # Absent picks_dir completely
            ("absent_picks_dir", dict(day=DAY, decision=None, create_picks=False)),
            # Invalid date string on non-skip and skip
            ("bad_date_no_decision", dict(day="not-a-date", decision=None)),
            ("bad_date_skip", dict(day="not-a-date", decision=make_skip("not-a-date"))),
            ("leap_invalid_skip", dict(day="2027-02-29", decision=make_skip("2027-02-29"))),
            # Extreme date 0001-01-01 and 9999-12-31 on non-deliberate_skip
            ("year1_no_decision", dict(day="0001-01-01", decision=None)),
            ("year9999_no_games", dict(day="9999-12-31", decision=make_skip("9999-12-31"), schedule=())),
            # Config where scheduler section is a list (intent_unknown)
            ("scheduler_list_config", dict(day=DAY, decision=make_skip(), raw_config="scheduler = []\n")),
            # Invalid UTF-8 in decision.json (raises UnicodeDecodeError -> ValueError -> exit 2 could-not-run DM!)
            ("invalid_utf8_decision", dict(day=DAY, decision=b"\xff\xfe")),
            # Single pick with runner-up in double_down (W8: pick text names only primary)
            ("single_with_runner_up", dict(
                day=DAY,
                decision=_decision("single", double={"batter_name": "Runner Up"}),
            )),
        ]
        for name, kw in cases:
            sub = root / name
            if kw.get("create_picks", True):
                acct = sub / "picks/account_state"
                acct.mkdir(parents=True, exist_ok=True)
                (acct / "contest_streak.manual.json").write_text(json.dumps(make_dated()))
                (acct / "season_contract.json").write_text(json.dumps(make_switch()))
            res = invoke_cli(sub, **kw)
            results.append({"case": name, **res})
    print(json.dumps(results, sort_keys=True))


def check_extra_parity_and_r64() -> None:
    print("=== A5. Extra paired base-vs-F3 cases and R64 text audit ===", flush=True)
    outs = {}
    for label, src in [("base", BASE_SRC), ("f3", F3_SRC)]:
        env = os.environ | {
            "TZ": "America/New_York",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{src}:{REPO}",
        }
        proc = subprocess.run(
            [sys.executable, __file__, "--parity-child"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=30, check=True,
        )
        lines = proc.stdout.splitlines()
        assert Path(lines[0].split("=", 1)[1]).resolve().is_relative_to(src.resolve())
        outs[label] = (lines[0], lines[1])

    print(f"base: {outs['base'][0]}")
    print(f"f3:   {outs['f3'][0]}")
    assert outs["base"][1] == outs["f3"][1], "extra parity cases differed between base and F3!"
    parsed = json.loads(outs["f3"][1])
    print(f"A5 parity: {len(parsed)}/{len(parsed)} tricky cases (absent picks_dir, invalid dates, UTF-8 decision exit 2, W8) matched byte for byte.")

    # R64 check across all 5 rendered message shapes
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        shapes = {
            "installed_only": {"contest_streak.manual.json": make_dated(streak=0, saver="not_earned")},
            "archive_only": {"contest_streak.manual.revoked-1.json": make_dated(streak=1, saver="used")},
            "revoke_only": {"season_contract.json": make_switch()},
            "installed_and_revoke": {
                "contest_streak.manual.json": make_dated(streak=12, saver="active"),
                "season_contract.json": make_switch(),
            },
            "archive_and_revoke": {
                "contest_streak.manual.revoked-1.json": make_dated(streak=12, saver="used"),
                "season_contract.json": make_switch(),
            },
        }
        forbidden_tokens = ("B8a", "DIR", "set-contest-streak", "revoke --season", "restart", "Nothing needs doing")
        for name, files in shapes.items():
            sub = root / name
            acct = sub / "picks/account_state"
            acct.mkdir(parents=True)
            for fn, doc in files.items():
                (acct / fn).write_text(json.dumps(doc))
            res = invoke_cli(sub, decision=make_skip(streak=30, saver=False))
            assert res["exit"] == 1 and len(res["sends"]) == 1
            dm_text = res["sends"][0][1]
            assert dm_text.startswith(f"BTS day check {DAY}: SKIP STATE REVIEW. ")
            assert dm_text.endswith("Check MLB for unsettled entries and verify the current streak, season best and saver.")
            for tok in forbidden_tokens:
                assert tok not in dm_text, (name, tok, dm_text)
            print(f"R64 [{name}]: {dm_text}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--parity-child":
        run_extra_parity_child()
    else:
        check_extreme_dates_and_iso_grammar()
        check_unhashable_and_hostile_fields()
        check_dst_and_saver_boundaries()
        check_filesystem_symlinks_and_partial_reads()
        check_extra_parity_and_r64()
