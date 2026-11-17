"""Adversarial CLI, message recovery, status, revoke, and operator sequence probe (Items 5, 6, 7, 8).

Every network DNS/connect event is forbidden via sys.addaudithook; fetch, DM,
posting, and schedule transports are captured offline; filesystem operations are
audited via sys.addaudithook.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

AUDIT = {
    "active": False,
    "denied_net": [],
    "events": [],
    "write_opens": [],
    "mutations": [],
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


def _hook(event: str, args: tuple) -> None:
    if event in _NET_EVENTS:
        AUDIT["denied_net"].append((event, repr(args)))
        raise RuntimeError(f"network forbidden by audit hook: {event}")
    if not AUDIT["active"]:
        return
    if event == "open" and args:
        path_s = str(args[0])
        mode = str(args[1]) if len(args) > 1 else "r"
        AUDIT["events"].append(("open", path_s, mode))
        if any(ch in mode for ch in "wax+"):
            AUDIT["write_opens"].append((path_s, mode))
    elif event == "os.stat" and args:
        AUDIT["events"].append(("stat", str(args[0])))
    elif event in ("os.mkdir", "os.unlink", "os.rmdir"):
        AUDIT["events"].append((event, str(args[0])))
        AUDIT["mutations"].append((event, str(args[0])))
    elif event in ("os.rename", "os.replace"):
        AUDIT["events"].append((event, str(args[0]), str(args[1])))
        AUDIT["mutations"].append((event, str(args[0]), str(args[1])))


sys.addaudithook(_hook)

from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas  # noqa: F401 - initialize real datetime types before clock patching
import httpx
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import bts.cli as cli
import bts.contest_fetch as cf
import bts.contest_state as cs
import bts.dm as dm_mod
import bts.health.contest_state as hc
import bts.leaderboard.auth as auth
import bts.posting as posting_mod
import bts.saver_state as ss
import bts.season_contract as sc
import bts.util as util

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
ID = "a" * 32


def _reset_audit() -> None:
    AUDIT["events"].clear()
    AUDIT["write_opens"].clear()
    AUDIT["mutations"].clear()


@contextmanager
def audited():
    _reset_audit()
    AUDIT["active"] = True
    try:
        yield AUDIT
    finally:
        AUDIT["active"] = False


@contextmanager
def captured_env(now_box: list[dt.datetime], *, profile_fn=None, rounds_fn=None):
    """Patch wall clock + capture all fetch, DM, and posting transports."""
    sent_dms = []
    sent_posts = []

    class FakeDatetime(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            cur = now_box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

    def default_profile(*_a, **_kw):
        return {
            "activeStreak": 7,
            "seasonBestStreak": 9,
            "predictions": [{"roundId": 1, "result": "hit"}],
        }

    def default_rounds():
        cur_d = now_box[0].astimezone(ET).date()
        return {1: cur_d - dt.timedelta(days=1)}

    with (
        patch.object(dt, "datetime", FakeDatetime),
        patch.object(httpx, "get", side_effect=AssertionError("uncaptured httpx.get")),
        patch.object(auth, "load_session_cookies", return_value={}),
        patch.object(auth, "extract_uid", return_value="u1"),
        patch.object(
            auth,
            "fetch_login_session",
            side_effect=lambda **_: SimpleNamespace(user_id=50311, username="stonehengee", xsid="x"),
        ),
        patch.object(cf, "fetch_profile", side_effect=profile_fn or default_profile),
        patch.object(cli, "_fetch_rounds", side_effect=rounds_fn or default_rounds),
        patch.object(dm_mod, "send_dm", side_effect=lambda r, t: sent_dms.append((r, t)) or "dm-id"),
        patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: sent_posts.append(t) or "uri"),
    ):
        yield {"dms": sent_dms, "posts": sent_posts}


def _tree(root: Path) -> dict[str, bytes | str]:
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = f"SYMLINK->{os.readlink(p)}"
        elif p.is_file():
            out[rel] = p.read_bytes()
        elif p.is_dir():
            out[rel] = "DIR"
    return out


def _put_json(picks: Path, name: str, obj) -> Path:
    p = picks / "account_state" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, bytes):
        p.write_bytes(obj)
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _sw(**kw):
    return {
        "schema_version": sc.SCHEMA,
        "season": 2027,
        "status": "automatic",
        "activation_id": ID,
        "switched_at": "2027-07-15T16:00:00Z",
        "revoked_at": None,
        "note": None,
    } | kw


def _auto(**kw):
    return {
        "schema_version": sc.AUTO_SCHEMA,
        "season": 2027,
        "activation_id": ID,
        "active_streak": 12,
        "best_streak": 15,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "recorded_at": "2027-07-15T15:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
    } | kw


def _dated(day: str = "2027-07-15", **kw):
    d = dt.date.fromisoformat(day)
    return {
        "schema_version": cs.DATED_MANUAL_SCHEMA,
        "effective_date": day,
        "season": d.year,
        "expires_at": REAL_DT.combine(d, dt.time.max, ET).isoformat(),
        "active_streak": 12,
        "best_streak": 15,
        "saver_state": "used",
        "recorded_at": "2027-07-15T15:00:00Z",
        "source": "manual_cli",
    } | kw


def _extract_cmds(msg: str) -> list[tuple[str, list[str]]]:
    raw_cmds = re.findall(r"`(bts [^`]+)`", msg)
    return [(raw, shlex.split(raw)) for raw in raw_cmds]


def probe_item5_messages() -> dict:
    """Item 5: Every reason & sub-case of reason (5), shell quoting, complete alternative paths."""
    runner = CliRunner()
    now_box = [NOW]
    summary = {"cases_tested": 0, "precedence_checks": [], "opening_night_auto_path": {}}

    cases = [
        ("corrupt_check1", [], _auto(), None),
        ("corrupt_check2", _sw(schema_version="bad"), _auto(), None),
        ("corrupt_check3", _sw(season=True), _auto(), None),
        ("corrupt_check4", _sw(status="bad"), _auto(), None),
        ("corrupt_check5", _sw(activation_id="G" * 32), _auto(), None),
        ("corrupt_check6", _sw(switched_at=None), _auto(), None),
        ("corrupt_check7", _sw(revoked_at="naive"), _auto(), None),
        ("corrupt_check8", _sw(note=123), _auto(), None),
        ("legacy_unexpired", _sw(), _auto(), {"active_streak": 5, "override_expires_at": "2027-07-16T16:00:00Z"}),
        ("legacy_expired_no_auto", _sw(), None, {"active_streak": 5, "override_expires_at": "2027-07-14T16:00:00Z"}),
        ("legacy_no_exp_no_auto", None, None, {"active_streak": 5}),
        ("absent_switch", None, _auto(), None),
        ("other_season_switch", _sw(season=2026), _auto(), None),
        ("revoked_switch", _sw(status="revoked", revoked_at="2027-07-15T16:00:00Z"), _auto(), None),
        ("missing_auto", _sw(), None, None),
        ("r5_auto_v1", _sw(), _auto(schema_version="bts_contest_streak_auto_v1"), None),
        ("r5_schema_missing", _sw(), {k: v for k, v in _auto().items() if k != "schema_version"}, None),
        ("r5_schema_null", _sw(), _auto(schema_version=None), None),
        ("r5_schema_int", _sw(), _auto(schema_version=99), None),
        ("r5_schema_list", _sw(), _auto(schema_version=["v2"]), None),
        ("r5_schema_unknown", _sw(), _auto(schema_version="mystery_v0"), None),
        ("r5_other_season", _sw(), _auto(season=2026), None),
        ("r5_other_id", _sw(), _auto(activation_id="b" * 32), None),
    ]

    with captured_env(now_box):
        for name, sw_val, auto_val, man_val in cases:
            for alt in ("dated", "automatic"):
                with tempfile.TemporaryDirectory() as tmp:
                    picks = Path(tmp) / "dir space 'single' \"double\" & picks"
                    picks.mkdir()
                    if sw_val is not None:
                        _put_json(picks, "season_contract.json", sw_val)
                    if auto_val is not None:
                        _put_json(picks, "contest_streak.json", auto_val)
                    if man_val is not None:
                        _put_json(picks, "contest_streak.manual.json", man_val)

                    try:
                        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                        raise AssertionError(f"expected refusal for {name}")
                    except cs.ContestStateError as exc:
                        msg = str(exc)

                    cmds = _extract_cmds(msg)
                    assert cmds, f"no commands in {name}"
                    for raw, argv in cmds:
                        idx = argv.index("--picks-dir")
                        assert argv[idx + 1] == str(picks)
                        # Verify real /bin/sh quoting round-trip on spaces, ', ", and &
                        quoted_part = raw.split("--picks-dir ", 1)[1]
                        sh_out = subprocess.check_output(
                            ["/bin/sh", "-c", f"printf '%s' {quoted_part}"], text=True
                        )
                        assert sh_out == str(picks), (sh_out, str(picks))

                    # Verify required boilerplate on every message
                    assert "TEMPLATE:" in msg and "Read N, B and S from MLB" in msg
                    assert "One repair can expose another blocker." in msg
                    assert "A non-zero fetch exit does not prove no observation was installed." in msg
                    assert "Status output is not itself a verification of what MLB shows." in msg

                    if alt == "dated":
                        dated_argv = next(argv for _, argv in cmds if argv[1] == "set-contest-streak")
                        filled = [{"N": "4", "B": "9", "S": "not_earned"}.get(tok, tok) for tok in dated_argv]
                        res = runner.invoke(cli.cli, filled[1:])
                        assert res.exit_code == 0, res.output
                        acc = cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                        assert acc.streak == 4 and acc.saver_available is True
                    else:
                        if name.startswith("legacy_"):
                            # Verify enable + fetch indeed CANNOT clear unexpired legacy override
                            if name == "legacy_unexpired":
                                assert runner.invoke(
                                    cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]
                                ).exit_code == 0
                                assert runner.invoke(
                                    cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]
                                ).exit_code == 0
                                try:
                                    cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                    raise AssertionError("unexpired legacy should still block!")
                                except cs.ContestStateError as exc2:
                                    assert "legacy manual override" in str(exc2)
                            # Remove legacy override -> exposes next blocker or covered auto
                            (picks / "account_state/contest_streak.manual.json").unlink()
                            if name == "legacy_unexpired":
                                # We already enabled + fetched above, so removing legacy now accepts!
                                acc = cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                assert acc.streak == 7
                            elif name == "legacy_expired_no_auto":
                                # Switch was automatic, auto absent -> exposes reason 4!
                                try:
                                    cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                    raise AssertionError("expected reason 4")
                                except cs.ContestStateError as exc3:
                                    assert "no automatic observation" in str(exc3)
                                fetch_argv = next(
                                    argv for _, argv in _extract_cmds(str(exc3)) if argv[1] == "fetch-contest-streak"
                                )
                                assert runner.invoke(cli.cli, fetch_argv[1:]).exit_code == 0
                                assert cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW).streak == 7
                            else:
                                # Switch was absent, auto absent -> exposes reason 3!
                                try:
                                    cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                    raise AssertionError("expected reason 3")
                                except cs.ContestStateError as exc3:
                                    assert "switch is absent" in str(exc3)
                                for _, argv in _extract_cmds(str(exc3)):
                                    if argv[1] != "set-contest-streak":
                                        assert runner.invoke(cli.cli, argv[1:]).exit_code == 0
                                assert cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW).streak == 7
                        elif name.startswith("corrupt_"):
                            auto_cmds = [argv for _, argv in cmds if argv[1] != "set-contest-streak"]
                            # 1. Inspect status first while corrupt -> exits 1!
                            r_st1 = runner.invoke(cli.cli, auto_cmds[0][1:])
                            assert r_st1.exit_code == 1 and "switch: corrupt" in r_st1.output
                            # 2. Remove corrupt switch -> exposes reason 3 (absent)
                            (picks / "account_state/season_contract.json").unlink()
                            try:
                                cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                raise AssertionError("expected absent after unlink")
                            except cs.ContestStateError as exc_abs:
                                assert "switch is absent" in str(exc_abs)
                            # 3. Unswitched fetch -> ledger only, still reason 3 (absent)
                            assert runner.invoke(cli.cli, auto_cmds[1][1:]).exit_code == 0
                            try:
                                cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                raise AssertionError("expected still absent after unswitched fetch")
                            except cs.ContestStateError as exc_abs2:
                                assert "switch is absent" in str(exc_abs2)
                            # 4. Status -> exit 0, still reason 3
                            r_st2 = runner.invoke(cli.cli, auto_cmds[2][1:])
                            assert r_st2.exit_code == 0 and "live decision now: refused" in r_st2.output
                            # 5. Enable -> exposes reason 5 (old activation_id on auto_val)!
                            assert runner.invoke(cli.cli, auto_cmds[3][1:]).exit_code == 0
                            try:
                                cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                raise AssertionError("expected activation_id mismatch after enable")
                            except cs.ContestStateError as exc_id:
                                assert "activation_id" in str(exc_id)
                            # 6. Switched fetch -> accepted!
                            assert runner.invoke(cli.cli, auto_cmds[4][1:]).exit_code == 0
                            assert cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW).streak == 7
                        elif name in ("absent_switch", "other_season_switch", "revoked_switch"):
                            auto_cmds = [argv for _, argv in cmds if argv[1] != "set-contest-streak"]
                            # [fetch, status, enable, fetch]
                            assert runner.invoke(cli.cli, auto_cmds[0][1:]).exit_code == 0
                            # Still reason 3 after unswitched fetch!
                            try:
                                cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                raise AssertionError("still reason 3")
                            except cs.ContestStateError as exc_mid1:
                                assert ("absent" if name == "absent_switch" else
                                        "other_season" if name == "other_season_switch" else "revoked") in str(exc_mid1)
                            assert runner.invoke(cli.cli, auto_cmds[1][1:]).exit_code == 0
                            assert runner.invoke(cli.cli, auto_cmds[2][1:]).exit_code == 0
                            # After enable before 2nd fetch: reason 5 (activation_id mismatch)!
                            try:
                                cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                                raise AssertionError("expected reason 5 after enable")
                            except cs.ContestStateError as exc_mid2:
                                assert "activation_id" in str(exc_mid2)
                            assert runner.invoke(cli.cli, auto_cmds[3][1:]).exit_code == 0
                            assert cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW).streak == 7
                        else:
                            # Reason 4 and Reason 5: single switched fetch command
                            auto_cmds = [argv for _, argv in cmds if argv[1] != "set-contest-streak"]
                            assert len(auto_cmds) == 1
                            assert runner.invoke(cli.cli, auto_cmds[0][1:]).exit_code == 0
                            assert cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW).streak == 7

                    summary["cases_tested"] += 1

        # Multi-reason precedence verification
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            # 1 vs 2: corrupt switch + unexpired legacy manual -> reason 1
            _put_json(picks, "season_contract.json", [])
            _put_json(picks, "contest_streak.manual.json", {"active_streak": 3, "override_expires_at": "2027-07-16T16:00:00Z"})
            m12 = sc.live_automatic_refusal(picks, NOW, cs.load_contest_streak_state(picks, now=NOW, require_dated_auto=True))
            assert "corrupt season switch, check 1" in m12.splitlines()[0]
            summary["precedence_checks"].append("1_over_2")

            # 2 vs 3: absent switch + unexpired legacy manual -> reason 2
            (picks / "account_state/season_contract.json").unlink()
            m23 = sc.live_automatic_refusal(picks, NOW, cs.load_contest_streak_state(picks, now=NOW, require_dated_auto=True))
            assert "legacy manual override" in m23.splitlines()[0]
            summary["precedence_checks"].append("2_over_3")

            # 3 vs 4: revoked switch + no observations -> reason 3
            (picks / "account_state/contest_streak.manual.json").unlink()
            _put_json(picks, "season_contract.json", _sw(status="revoked", revoked_at="2027-07-15T16:00:00Z"))
            m34 = sc.live_automatic_refusal(picks, NOW, None)
            assert "switch is revoked" in m34.splitlines()[0]
            summary["precedence_checks"].append("3_over_4")

            # 5a vs 5b vs 5c: schema before season before activation_id
            _put_json(picks, "season_contract.json", _sw())
            _put_json(picks, "contest_streak.json", _auto(schema_version="bts_contest_streak_auto_v1", season=2026, activation_id="b" * 32))
            m5abc = sc.live_automatic_refusal(picks, NOW, cs.load_contest_streak_state(picks, now=NOW, require_dated_auto=True))
            assert "schema_version" in m5abc.splitlines()[0]
            _put_json(picks, "contest_streak.json", _auto(season=2026, activation_id="b" * 32))
            m5bc = sc.live_automatic_refusal(picks, NOW, cs.load_contest_streak_state(picks, now=NOW, require_dated_auto=True))
            assert "observation season is 2026, not 2027" in m5bc.splitlines()[0]
            summary["precedence_checks"].extend(["5_schema_over_season_and_id", "5_season_over_id"])

    # Opening Night with predictions=[] (no settled rounds yet -> source_date=None)
    opener_box = [REAL_DT(2027, 3, 24, 15, 0, 0, tzinfo=ET)]
    with captured_env(
        opener_box,
        profile_fn=lambda *_: {"activeStreak": 0, "seasonBestStreak": 0, "predictions": []},
        rounds_fn=lambda: {1: dt.date(2027, 3, 24)},
    ):
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            _put_json(picks, "contest_streak.json", _auto(schema_version="bts_contest_streak_auto_v1", active_streak=18, best_streak=22, source_date="2026-09-27"))
            _put_json(picks, "saver_state.json", {"season": 2026, "state": "active"})
            try:
                cs.load_decision_streak_state(picks, require_contest_state=True, now=opener_box[0])
                raise AssertionError("expected refusal")
            except cs.ContestStateError as exc:
                cmds = _extract_cmds(str(exc))
            for _, argv in cmds:
                if argv[1] != "set-contest-streak":
                    assert runner.invoke(cli.cli, argv[1:]).exit_code == 0
            st = cs.load_decision_streak_state(picks, require_contest_state=True, now=opener_box[0])
            hlth = hc.check(picks, expected=True, now=opener_box[0])
            summary["opening_night_auto_path"] = {
                "accepted_streak": st.streak,
                "status": st.status,
                "saver_flag_state": st.saver_flag_state,
                "saver_available": st.saver_available,
                "health_alerts": [(a.level, a.source, a.message.splitlines()[0]) for a in hlth],
            }

    return summary


def probe_item6_status() -> dict:
    """Item 6: status in every state, --season vs actual Y, read-only & creation-free audit."""
    runner = CliRunner()
    now_box = [NOW]
    checked = []

    with captured_env(now_box) as transports:
        # 1. Cold nonexistent picks directory
        with tempfile.TemporaryDirectory() as tmp:
            cold = Path(tmp) / "nonexistent_picks"
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(cold)])
            assert res.exit_code == 0, res.output
            assert aud["write_opens"] == [] and aud["mutations"] == []
            assert _tree(Path(tmp)) == before
            assert "live decision now: refused; Live decision refused: season 2027 switch is absent." in res.output
            checked.append("cold_absent")

        # 2. Accepted covered auto + --season 2026 (inspected switch is other_season, live uses 2027!)
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            _put_json(picks, "season_contract.json", _sw())
            _put_json(picks, "contest_streak.json", _auto())
            _put_json(picks, "saver_state.json", {"season": 2027, "state": "active"})
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks), "--season", "2026"])
            assert res.exit_code == 0
            assert "switch: other_season" in res.output
            real = cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
            expected_line = (
                f"live decision now: accepted; source={real.source}; streak={real.streak}; "
                f"status={real.status}; saver={real.saver_flag_state}"
            )
            assert expected_line in res.output
            assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
            checked.append("accepted_covered_with_season_2026")

        # 3. Valid 2026 switch + --season 2026 (inspected switch is automatic, live refuses for 2027!)
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            _put_json(picks, "season_contract.json", _sw(season=2026))
            _put_json(picks, "contest_streak.json", _auto(season=2026))
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks), "--season", "2026"])
            assert res.exit_code == 0
            assert "switch: automatic" in res.output
            assert "live decision now: refused; Live decision refused: season 2027 switch is other_season (file season 2026)." in res.output
            assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
            checked.append("other_season_file_with_season_2026")

        # 4. Accepted dated today beside CORRUPT switch -> exit 1 (corrupt switch), live decision accepted!
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            _put_json(picks, "season_contract.json", [])
            _put_json(picks, "contest_streak.manual.json", _dated())
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
            assert res.exit_code == 1
            assert "switch: corrupt" in res.output
            assert "live decision now: accepted; source=contest; streak=12; status=fresh; saver=used" in res.output
            assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
            checked.append("accepted_dated_beside_corrupt_switch_exits_1")

        # 5. Unavailable: non-ContestStateError (overlong int AND deep recursion in saver_state.json)
        for kind, payload in [
            ("overlong_int", b'{"n":' + (b"9" * 5000) + b"}"),
            ("deep_recursion", b"[" * 5000 + b"]" * 5000),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                picks.mkdir()
                _put_json(picks, "season_contract.json", _sw())
                _put_json(picks, "contest_streak.json", _auto())
                _put_json(picks, "saver_state.json", payload)
                before = _tree(Path(tmp))
                with audited() as aud:
                    res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
                assert res.exit_code == 0, res.output
                assert "live decision now: unavailable;" in res.output
                assert res.output.index("saver:") < res.output.index("live decision now:") < res.output.index("ledger:")
                assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
                checked.append(f"unavailable_{kind}")

        assert transports["dms"] == [] and transports["posts"] == []

    return {"checked": checked}


def probe_item7_revoke() -> dict:
    """Item 7: revoke usage errors, audit-hook step order, all failure modes, carry rules, New Year, non-re-coverability."""
    runner = CliRunner()
    now_box = [NOW]
    report = {}

    with captured_env(now_box):
        # 1. Usage errors with ZERO state reads/stats
        for bad_args in [
            [],
            ["--season", "2026", "--reason", "test"],
            ["--season", "notint", "--reason", "test"],
            ["--season", "2027"],
            ["--season", "2027", "--reason", ""],
            ["--season", "2027", "--reason", "  \t\n  "],
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                _put_json(picks, "season_contract.json", _sw())
                with audited() as aud:
                    res = runner.invoke(cli.cli, ["season-contract", "revoke", *bad_args, "--picks-dir", str(picks)])
                assert res.exit_code == 2, (bad_args, res.exit_code, res.output)
                state_touches = [
                    ev for ev in aud["events"]
                    if any("account_state" in str(x) or "health_state" in str(x) for x in ev[1:])
                ]
                assert state_touches == [], (bad_args, state_touches)
        report["usage_zero_reads"] = True

        # 2. Exact 3-step order via sys.addaudithook
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "season_contract.json", _sw())
            _put_json(picks, "contest_streak.manual.json", _dated())
            _put_json(picks, "saver_state.json", {"season": 2027, "state": "used"})
            with audited() as aud:
                res = runner.invoke(
                    cli.cli,
                    ["season-contract", "revoke", "--season", "2027", "--reason", " audit ", "--picks-dir", str(picks)],
                )
            assert res.exit_code == 0, res.output
            relevant = []
            for ev in aud["events"]:
                joined = " ".join(str(x) for x in ev)
                if "account_state" not in joined:
                    continue
                if ev[0] == "open" and ev[1].endswith("season_contract.json") and "r" in ev[2]:
                    relevant.append("read_prior_switch")
                elif ev[0] == "os.replace" and ev[2].endswith("season_contract.json"):
                    relevant.append("write_revoked_switch")
                elif ev[0] == "stat" and ev[1].endswith("contest_streak.manual.json"):
                    relevant.append("lstat_manual_src")
                elif ev[0] == "stat" and "contest_streak.manual.revoked-" in ev[1]:
                    relevant.append("lstat_manual_dst")
                elif ev[0] == "os.rename" and ev[1].endswith("contest_streak.manual.json"):
                    relevant.append("rename_manual")
                elif ev[0] == "stat" and ev[1].endswith("saver_state.json"):
                    relevant.append("lstat_saver_src")
                elif ev[0] == "stat" and "saver_state.revoked-" in ev[1]:
                    relevant.append("lstat_saver_dst")
                elif ev[0] == "os.rename" and ev[1].endswith("saver_state.json"):
                    relevant.append("rename_saver")
            expected_order = [
                "read_prior_switch",
                "write_revoked_switch",
                "lstat_manual_src",
                "lstat_manual_dst",
                "rename_manual",
                "lstat_saver_src",
                "lstat_saver_dst",
                "rename_saver",
            ]
            assert relevant == expected_order, relevant
            report["audit_step_order"] = relevant

        # 3. Every step failing (switch dir, readonly folder, existing targets of all 4 types, lstat errors)
        stamp = "20270715T160000000000Z"
        for target_kind in ("regular", "directory", "valid_symlink", "dangling_symlink"):
            for stage in ("manual", "saver"):
                for manual_present in ([True] if stage == "manual" else [True, False]):
                    with tempfile.TemporaryDirectory() as tmp:
                        picks = Path(tmp) / "picks"
                        _put_json(picks, "season_contract.json", _sw())
                        auto_p = _put_json(picks, "contest_streak.json", _auto())
                        ledger_p = _put_json(picks, "contest_ledger.jsonl", {"row": 1})
                        fstat_p = picks.parent / "health_state/contest_streak_fetch_status.json"
                        fstat_p.parent.mkdir(parents=True, exist_ok=True)
                        fstat_p.write_text('{"ok": true}', encoding="utf-8")
                        untouched_before = {p: p.read_bytes() for p in (auto_p, ledger_p, fstat_p)}

                        if manual_present:
                            _put_json(picks, "contest_streak.manual.json", _dated())
                        _put_json(picks, "saver_state.json", {"season": 2027, "state": "used"})

                        stem = "contest_streak.manual" if stage == "manual" else "saver_state"
                        tgt = picks / "account_state" / f"{stem}.revoked-{stamp}.json"
                        if target_kind == "regular":
                            tgt.write_text("keep_me", encoding="utf-8")
                        elif target_kind == "directory":
                            tgt.mkdir()
                        elif target_kind == "valid_symlink":
                            real_f = picks / "account_state/other_real.txt"
                            real_f.write_text("real", encoding="utf-8")
                            tgt.symlink_to(real_f)
                        else:
                            tgt.symlink_to(picks / "account_state/does_not_exist")

                        with audited() as aud:
                            res = runner.invoke(
                                cli.cli,
                                ["season-contract", "revoke", "--season", "2027", "--reason", "fail", "--picks-dir", str(picks)],
                            )
                        assert res.exit_code == 1, (stage, target_kind, res.output)
                        assert sc.switch_state(picks, NOW).state == "revoked"
                        if stage == "manual":
                            assert "manual file was not set aside" in res.output
                            assert "saver flag was not attempted" in res.output
                            assert (picks / "account_state/contest_streak.manual.json").exists()
                            assert (picks / "account_state/saver_state.json").exists()
                            assert not any("saver_state" in str(e) for e in aud["events"])
                        else:
                            assert "saver flag was not set aside" in res.output
                            if manual_present:
                                assert "renamed to" in res.output
                                assert not (picks / "account_state/contest_streak.manual.json").exists()
                                assert (picks / "account_state" / f"contest_streak.manual.revoked-{stamp}.json").exists()
                            else:
                                assert "absent" in res.output
                            assert (picks / "account_state/saver_state.json").exists()
                        # Target was NOT clobbered
                        if target_kind == "regular":
                            assert tgt.read_text(encoding="utf-8") == "keep_me"
                        elif target_kind == "directory":
                            assert tgt.is_dir()
                        else:
                            assert tgt.is_symlink()
                        for p, b in untouched_before.items():
                            assert p.read_bytes() == b
        report["target_collision_all_types_verified"] = True

        # Read-only folder at step (a)
        if os.geteuid() != 0:
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                _put_json(picks, "season_contract.json", _sw())
                _put_json(picks, "contest_streak.manual.json", _dated())
                _put_json(picks, "saver_state.json", {"season": 2027, "state": "used"})
                acct = picks / "account_state"
                acct.chmod(0o555)
                try:
                    res = runner.invoke(
                        cli.cli,
                        ["season-contract", "revoke", "--season", "2027", "--reason", "ro", "--picks-dir", str(picks)],
                    )
                    assert res.exit_code == 1
                    assert "was NOT revoked" in res.output and "nothing renamed" in res.output
                    assert sc.switch_state(picks, NOW).state == "automatic"
                    assert (acct / "contest_streak.manual.json").exists()
                    assert (acct / "saver_state.json").exists()
                finally:
                    acct.chmod(0o755)
            report["readonly_folder_step_a_verified"] = True

        # Dangling SOURCE symlinks for both manual and saver
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            acct = picks / "account_state"
            acct.mkdir(parents=True)
            (acct / "contest_streak.manual.json").symlink_to(acct / "gone1")
            (acct / "saver_state.json").symlink_to(acct / "gone2")
            res = runner.invoke(
                cli.cli,
                ["season-contract", "revoke", "--season", "2027", "--reason", "dangling", "--picks-dir", str(picks)],
            )
            assert res.exit_code == 0, res.output
            assert not (acct / "contest_streak.manual.json").is_symlink()
            assert not (acct / "saver_state.json").is_symlink()
            assert (acct / f"contest_streak.manual.revoked-{stamp}.json").is_symlink()
            assert (acct / f"saver_state.revoked-{stamp}.json").is_symlink()
        report["dangling_sources_verified"] = True

        # Carry rules across all prior switch states, including revoked with null vs non-null switched_at
        prior_cases = [
            ("absent", None, False, None),
            ("corrupt_obj", [], False, None),
            ("corrupt_schema", _sw(schema_version="v0"), False, None),
            ("corrupt_season", _sw(season=True), False, None),
            ("corrupt_status", _sw(status="x"), False, None),
            ("corrupt_id", _sw(activation_id="123"), False, None),
            ("corrupt_sw_at", _sw(switched_at=None), False, None),
            ("corrupt_rev_at", _sw(revoked_at="bad"), False, None),
            ("corrupt_note", _sw(note={}), False, None),
            ("other_season", _sw(season=2026), False, None),
            ("automatic_same_y", _sw(), True, "2027-07-15T16:00:00Z"),
            ("revoked_null_sw_at", _sw(status="revoked", activation_id="d" * 32, switched_at=None, revoked_at="2027-07-14T00:00:00Z"), True, None),
            ("revoked_with_sw_at", _sw(status="revoked", activation_id="e" * 32, switched_at="2027-03-25T14:00:00Z", revoked_at="2027-07-14T00:00:00Z"), True, "2027-03-25T14:00:00Z"),
        ]
        for label, prior_val, expect_carry, expect_sw_at in prior_cases:
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                picks.mkdir()
                if prior_val is not None:
                    _put_json(picks, "season_contract.json", prior_val)
                res = runner.invoke(
                    cli.cli,
                    ["season-contract", "revoke", "--season", "2027", "--reason", "carry", "--picks-dir", str(picks)],
                )
                assert res.exit_code == 0, (label, res.output)
                snap = sc.switch_state(picks, NOW).snapshot
                assert set(snap.keys()) == {
                    "schema_version", "season", "status", "activation_id", "switched_at", "revoked_at", "note"
                }
                if expect_carry:
                    assert snap["activation_id"] == prior_val["activation_id"]
                else:
                    assert sc.valid_activation_id(snap["activation_id"])
                    if isinstance(prior_val, dict) and "activation_id" in prior_val:
                        assert snap["activation_id"] != prior_val["activation_id"]
                assert snap["switched_at"] == expect_sw_at

                # Also test executing BOTH printed templates from revoke output!
                revoke_cmds = _extract_cmds(res.output)
                assert len(revoke_cmds) == 3  # dated template, saver template, status
                saver_cmd = [
                    "not_earned" if tok == "<state" else tok
                    for tok in revoke_cmds[1][1]
                    if tok not in ("read", "from", "MLB>")
                ]
                assert runner.invoke(cli.cli, saver_cmd[1:]).exit_code == 0
                assert ss.load_saver_state(picks, season=2027).state == "not_earned"
        report["carry_and_templates_verified"] = len(prior_cases)

        # Cannot make pre-revoke observation covered again without restoring season_contract.json
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "season_contract.json", _sw())
            auto_p = _put_json(picks, "contest_streak.json", _auto())
            pre_revoke_auto_bytes = auto_p.read_bytes()
            assert cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW).streak == 12

            # Revoke -> try unswitched fetch, enable, restore auto bytes, revoke again, enable again, restore auto bytes
            assert runner.invoke(cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "r1", "--picks-dir", str(picks)]).exit_code == 0
            for step_action in [
                lambda: runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]),
                lambda: auto_p.write_bytes(pre_revoke_auto_bytes),
                lambda: runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]),
                lambda: auto_p.write_bytes(pre_revoke_auto_bytes),
                lambda: runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]),
                lambda: runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]),
                lambda: auto_p.write_bytes(pre_revoke_auto_bytes),
            ]:
                step_action()
                if auto_p.read_bytes() == pre_revoke_auto_bytes:
                    try:
                        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                        raise AssertionError("pre-revoke automatic observation became covered again!")
                    except cs.ContestStateError:
                        pass
        report["pre_revoke_auto_never_recovered_without_switch_restore"] = True

    return report


def _step_snapshot(picks: Path, now: dt.datetime) -> dict:
    try:
        dec = cs.load_decision_streak_state(picks, require_contest_state=True, now=now)
        live_res = {
            "accepted": True,
            "streak": dec.streak,
            "status": dec.status,
            "saver_flag_state": dec.saver_flag_state,
            "policy_saver": dec.saver_available,
            "best_status": dec.best_status,
        }
    except cs.ContestStateError as exc:
        live_res = {"accepted": False, "reason_line": str(exc).splitlines()[0]}
    alerts = hc.check(picks, expected=True, now=now)
    return {
        "live": live_res,
        "health": [(a.level, a.source, a.message.splitlines()[0]) for a in alerts],
    }


def probe_item8_sequences() -> dict:
    """Item 8: All 5 operator sequences (a)-(e) recording live, health, and policy saver at each step."""
    runner = CliRunner()
    out = {}

    # (a) Season start
    day1 = REAL_DT(2027, 3, 24, 14, 0, 0, tzinfo=ET)
    day2 = REAL_DT(2027, 3, 25, 14, 0, 0, tzinfo=ET)
    now_box = [day1]
    with captured_env(now_box):
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "contest_streak.json", _auto(schema_version="bts_contest_streak_auto_v1", active_streak=18, best_streak=22, source_date="2026-09-27"))
            _put_json(picks, "saver_state.json", {"season": 2026, "state": "active"})
            seq_a = {}
            seq_a["1_day1_initial"] = _step_snapshot(picks, now_box[0])
            assert not seq_a["1_day1_initial"]["live"]["accepted"]

            assert runner.invoke(
                cli.cli,
                ["set-contest-streak", "--date", "2027-03-24", "--streak", "0", "--best-streak", "0", "--saver", "not_earned", "--picks-dir", str(picks)],
            ).exit_code == 0
            seq_a["2_day1_dated"] = _step_snapshot(picks, now_box[0])
            assert seq_a["2_day1_dated"]["live"]["accepted"] and seq_a["2_day1_dated"]["live"]["policy_saver"] is True

            now_box[0] = day2
            seq_a["3_day2_before_obs"] = _step_snapshot(picks, now_box[0])
            assert not seq_a["3_day2_before_obs"]["live"]["accepted"]

            assert runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]).exit_code == 0
            seq_a["4_day2_evidence_fetch"] = _step_snapshot(picks, now_box[0])
            assert not seq_a["4_day2_evidence_fetch"]["live"]["accepted"]

            st_res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
            assert st_res.exit_code == 0 and "ledger: readable; rows: 1" in st_res.output

            assert runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]).exit_code == 0
            seq_a["5_day2_after_enable_before_fetch"] = _step_snapshot(picks, now_box[0])
            assert "schema_version" in seq_a["5_day2_after_enable_before_fetch"]["live"]["reason_line"]

            assert runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]).exit_code == 0
            seq_a["6_day2_after_switched_fetch"] = _step_snapshot(picks, now_box[0])
            assert seq_a["6_day2_after_switched_fetch"]["live"]["accepted"]
            # Notice: fetch with 2027 source_date and best_streak=9 < 10 cold-inited saver_state.json to not_earned!
            assert seq_a["6_day2_after_switched_fetch"]["live"]["saver_flag_state"] == "not_earned"
            assert seq_a["6_day2_after_switched_fetch"]["live"]["policy_saver"] is True
            out["a_season_start"] = seq_a

    # (b) Restore without revoke (accepted limit), then revoke, dated, enable, next day refused on old ID, fetch, accepted
    now_box = [NOW]
    with captured_env(now_box):
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "season_contract.json", _sw())
            _put_json(picks, "contest_streak.json", _auto(active_streak=12, best_streak=15))
            _put_json(picks, "saver_state.json", {"season": 2027, "state": "active"})
            snap_sw = (picks / "account_state/season_contract.json").read_bytes()
            snap_auto = (picks / "account_state/contest_streak.json").read_bytes()
            snap_saver = (picks / "account_state/saver_state.json").read_bytes()

            # Simulate restoring snapshot without revoke:
            (picks / "account_state/season_contract.json").write_bytes(snap_sw)
            (picks / "account_state/contest_streak.json").write_bytes(snap_auto)
            (picks / "account_state/saver_state.json").write_bytes(snap_saver)
            seq_b = {}
            seq_b["1_restored_without_revoke"] = _step_snapshot(picks, now_box[0])
            assert seq_b["1_restored_without_revoke"]["live"]["accepted"]

            assert runner.invoke(cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "snapshot restore", "--picks-dir", str(picks)]).exit_code == 0
            seq_b["2_after_revoke"] = _step_snapshot(picks, now_box[0])
            assert "revoked" in seq_b["2_after_revoke"]["live"]["reason_line"]

            assert runner.invoke(cli.cli, ["set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "15", "--saver", "used", "--picks-dir", str(picks)]).exit_code == 0
            seq_b["3_dated_today"] = _step_snapshot(picks, now_box[0])
            assert seq_b["3_dated_today"]["live"]["accepted"] and seq_b["3_dated_today"]["live"]["policy_saver"] is False

            assert runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]).exit_code == 0
            seq_b["4_after_enable_same_day"] = _step_snapshot(picks, now_box[0])
            assert seq_b["4_after_enable_same_day"]["live"]["accepted"]  # today's dated still governs!

            now_box[0] = NOW + dt.timedelta(days=1)
            seq_b["5_next_day_before_fetch"] = _step_snapshot(picks, now_box[0])
            assert "activation_id" in seq_b["5_next_day_before_fetch"]["live"]["reason_line"]

            assert runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]).exit_code == 0
            seq_b["6_next_day_after_fetch"] = _step_snapshot(picks, now_box[0])
            assert seq_b["6_next_day_after_fetch"]["live"]["accepted"]
            # Saver flag was quarantined at revoke and best_streak=9 < 10 cold-inited to not_earned (or if best>=10 stays uninitialized!)
            out["b_restore_and_revoke"] = seq_b

    # (c) Fetch failing BEFORE vs AFTER replacing contest_streak.json following re-enable
    now_box = [NOW]
    with captured_env(now_box):
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "season_contract.json", _sw(status="revoked", revoked_at=NOW.isoformat()))
            _put_json(picks, "contest_streak.json", _auto())
            assert runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]).exit_code == 0
            seq_c = {}
            real_write = cli._atomic_write_json

            with patch.object(cli, "_atomic_write_json", side_effect=lambda p, v: (_ for _ in ()).throw(OSError("disk full")) if p.name == "contest_streak.json" else real_write(p, v)):
                r_pre = runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
            assert r_pre.exit_code == 1
            seq_c["1_fail_before_replace"] = _step_snapshot(picks, now_box[0])
            assert "activation_id" in seq_c["1_fail_before_replace"]["live"]["reason_line"]

            with patch.object(cli, "_atomic_write_json", side_effect=lambda p, v: (_ for _ in ()).throw(OSError("status fail")) if p.name == "contest_streak_fetch_status.json" else real_write(p, v)):
                r_post = runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
            assert r_post.exit_code == 1
            seq_c["2_fail_after_replace"] = _step_snapshot(picks, now_box[0])
            assert seq_c["2_fail_after_replace"]["live"]["accepted"]
            out["c_fetch_failures"] = seq_c

    # (d) December 31 23:59:59.999999 ET -> January 1 00:00:00.000000 ET
    dec31_end = REAL_DT(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)
    jan1_start = dec31_end + dt.timedelta(microseconds=1)
    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        _put_json(picks, "season_contract.json", _sw())
        _put_json(picks, "contest_streak.json", _auto(source_date="2027-12-31"))
        _put_json(picks, "saver_state.json", {"season": 2027, "state": "active"})
        seq_d = {
            "1_auto_dec31_last_us": _step_snapshot(picks, dec31_end),
            "2_auto_jan1_first_us": _step_snapshot(picks, jan1_start),
        }
        assert seq_d["1_auto_dec31_last_us"]["live"]["accepted"] and seq_d["1_auto_dec31_last_us"]["live"]["policy_saver"] is True
        assert "other_season (file season 2027)" in seq_d["2_auto_jan1_first_us"]["live"]["reason_line"]

        _put_json(picks, "contest_streak.manual.json", _dated("2027-12-31", saver_state="active"))
        seq_d["3_dated_dec31_last_us"] = _step_snapshot(picks, dec31_end)
        seq_d["4_dated_jan1_first_us"] = _step_snapshot(picks, jan1_start)
        assert seq_d["3_dated_dec31_last_us"]["live"]["accepted"]
        assert "other_season (file season 2027)" in seq_d["4_dated_jan1_first_us"]["live"]["reason_line"]
        out["d_new_year"] = seq_d

    # (e) Legacy override left on disk (unexpired AND expired) across enable, fetch, revoke, dated
    now_box = [NOW]
    with captured_env(now_box):
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "contest_streak.manual.json", {"active_streak": 6, "override_expires_at": "2027-07-20T00:00:00Z"})
            _put_json(picks, "contest_streak.json", _auto(schema_version="bts_contest_streak_auto_v1"))
            seq_e = {}
            seq_e["1_unexpired_legacy_absent_switch"] = _step_snapshot(picks, now_box[0])
            assert "legacy manual override" in seq_e["1_unexpired_legacy_absent_switch"]["live"]["reason_line"]

            assert runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]).exit_code == 0
            assert runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]).exit_code == 0
            seq_e["2_unexpired_legacy_after_enable_and_fetch"] = _step_snapshot(picks, now_box[0])
            assert "legacy manual override" in seq_e["2_unexpired_legacy_after_enable_and_fetch"]["live"]["reason_line"]

            # Expired legacy beside covered auto -> accepted, with health WARN on legacy override!
            _put_json(picks, "contest_streak.manual.json", {"active_streak": 6, "override_expires_at": "2027-07-10T00:00:00Z"})
            seq_e["3_expired_legacy_beside_covered_auto"] = _step_snapshot(picks, now_box[0])
            assert seq_e["3_expired_legacy_beside_covered_auto"]["live"]["accepted"]
            assert any("legacy/expired manual override" in h[2] for h in seq_e["3_expired_legacy_beside_covered_auto"]["health"])

            # Revoke sets aside the legacy manual file -> subsequent re-enable + fetch is completely clean!
            assert runner.invoke(cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "clean legacy", "--picks-dir", str(picks)]).exit_code == 0
            assert runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]).exit_code == 0
            assert runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]).exit_code == 0
            seq_e["4_after_revoke_reenable_fetch"] = _step_snapshot(picks, now_box[0])
            assert seq_e["4_after_revoke_reenable_fetch"]["live"]["accepted"]
            assert not any("legacy" in h[2] for h in seq_e["4_after_revoke_reenable_fetch"]["health"])
            out["e_legacy_override_across_sequences"] = seq_e

    return out


if __name__ == "__main__":
    results = {
        "item5_messages": probe_item5_messages(),
        "item6_status": probe_item6_status(),
        "item7_revoke": probe_item7_revoke(),
        "item8_sequences": probe_item8_sequences(),
        "denied_network": AUDIT["denied_net"],
    }
    assert AUDIT["denied_net"] == []
    out_file = Path(__file__).resolve().parent / "cli_and_sequences_report.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        "item5_cases": results["item5_messages"]["cases_tested"],
        "item6_checked": len(results["item6_status"]["checked"]),
        "item7_revoke": {k: v for k, v in results["item7_revoke"].items() if k != "audit_step_order"},
        "item8_sequences": list(results["item8_sequences"].keys()),
        "denied_network": len(results["denied_network"]),
    }, indent=2))
