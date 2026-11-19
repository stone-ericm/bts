"""Item 2 probe: status verdict vs live loader across ordinary, adversarial, FIFO & directory states."""
import datetime as dt
import json
import os
import signal
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import httpx
from click.testing import CliRunner

REVIEW = Path(__file__).resolve().parent
REV = sys.argv[1]
assert REV in ("base", "r1", "head"), REV
ROOT = REVIEW.parents[1] if REV == "head" else REVIEW / REV
sys.path.insert(0, str(ROOT / "src"))

import pandas  # noqa: F401 - import before datetime monkeypatching
from bts import cli, contest_fetch as cf, contest_state as cs, saver_state as saver, season_contract as sc
from bts.leaderboard import auth

expected_src = (ROOT / "src").resolve()
for mod in (cli, cf, cs, saver, sc):
    assert Path(mod.__file__).resolve().is_relative_to(expected_src), (REV, mod.__file__)

REAL_DT = dt.datetime
ET = ZoneInfo("America/New_York")
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
ID = "a" * 32


class BlockedError(BaseException):
    pass


@contextmanager
def bounded(seconds=0.08):
    def handler(*_):
        raise BlockedError("blocked_on_fifo")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@contextmanager
def offline_env(now=NOW, predictions=None):
    class PinnedDT(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    preds = [{"roundId": 1, "result": "hit"}] if predictions is None else predictions
    send_mock = Mock(return_value="dm-ack")
    profile_mock = Mock(return_value={
        "activeStreak": 7, "seasonBestStreak": 9, "predictions": preds,
    })
    with (
        patch.object(dt, "datetime", PinnedDT),
        patch.object(cs, "datetime", PinnedDT),
        patch.object(saver, "datetime", PinnedDT),
        patch.object(sc, "datetime", PinnedDT),
        patch.object(httpx, "get", Mock(side_effect=AssertionError("network forbidden"))),
        patch.object(auth, "load_session_cookies", lambda: {}),
        patch.object(auth, "extract_uid", lambda _: "u"),
        patch.object(auth, "fetch_login_session", lambda **_: SimpleNamespace(
            user_id=1, username="offline", xsid="x")),
        patch.object(cf, "fetch_profile", profile_mock),
        patch.object(cli, "_fetch_rounds", lambda: {1: (now.date() - dt.timedelta(days=1))}),
        patch("bts.dm.send_dm", send_mock),
    ):
        yield send_mock, profile_mock


def make_switch(now=NOW, **overrides):
    utc_str = now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "bts_season_contract_v1",
        "season": now.astimezone(ET).year,
        "status": "automatic",
        "activation_id": ID,
        "switched_at": utc_str,
        "revoked_at": None,
        "note": None,
    } | overrides


def make_auto(now=NOW, **overrides):
    utc_str = now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 12,
        "best_streak": 15,
        "source": "mlb_bts_profile",
        "source_date": (now.astimezone(ET).date() - dt.timedelta(days=1)).isoformat(),
        "recorded_at": utc_str,
        "season": now.astimezone(ET).year,
        "activation_id": ID,
    } | overrides


def make_dated(now=NOW, **overrides):
    d = now.astimezone(ET).date() if isinstance(now, dt.datetime) else now
    exp = REAL_DT.combine(d, dt.time.max, ET).isoformat()
    rec = REAL_DT(d.year, d.month, d.day, 12, 0, 0, tzinfo=ET).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": d.isoformat(),
        "season": d.year,
        "expires_at": exp,
        "active_streak": 1,
        "best_streak": 15,
        "saver_state": "not_earned",
        "recorded_at": rec,
        "source": "manual_cli",
    } | overrides


def write_item(path: Path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.is_file() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        path.rmdir()
    if body == "fifo":
        os.mkfifo(path)
    elif body == "directory":
        path.mkdir()
    elif isinstance(body, bytes):
        path.write_bytes(body)
    elif body is not None:
        path.write_text(json.dumps(body), encoding="utf-8")


def tree_snapshot(root: Path):
    snap = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        rel = p.relative_to(root).as_posix()
        payload = p.read_bytes() if stat.S_ISREG(st.st_mode) else None
        # Directories don't change bytes; track mode, size, mtime_ns and regular bytes
        snap[rel] = (stat.S_IFMT(st.st_mode), st.st_mtime_ns, payload)
    return snap


def run_status(picks: Path, *extra):
    try:
        with bounded():
            res = CliRunner().invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks), *extra])
            lines = res.output.replace(str(picks), "<PICKS>").splitlines()
            auto_lines = [ln for ln in lines if ln.startswith("automatic path alone:")]
            return {
                "outcome": "completed",
                "exit": res.exit_code,
                "exception": type(res.exception).__name__ if res.exception else None,
                "auto_lines": auto_lines,
                "has_expiry_warning": any("expires at midnight ET" in ln for ln in lines),
                "lines": lines,
            }
    except BlockedError:
        return {"outcome": "blocked", "auto_lines": []}


def run_live_without_selected_dated(picks: Path, now: dt.datetime, had_in_effect_dated: bool):
    manual_path = picks / "account_state/contest_streak.manual.json"
    saved = manual_path.read_bytes() if (had_in_effect_dated and manual_path.is_file()) else None
    if saved is not None:
        manual_path.unlink()
    try:
        with bounded():
            st = cs.load_decision_streak_state(picks, require_contest_state=True, now=now)
            rec = cs.automatic_reading_record(st, now)
            stamp = rec["recorded_at"] if rec and rec["reason"] != "unreadable" else "unavailable"
            return {
                "outcome": "accepted",
                "streak": st.streak,
                "status": st.status,
                "saver": st.saver_flag_state,
                "expected_line": f"automatic path alone: accepted (streak {st.streak}, recorded {stamp})",
            }
    except BlockedError:
        return {"outcome": "blocked"}
    except cs.ContestStateError as exc:
        first = str(exc).replace(str(picks), "<PICKS>").splitlines()[0]
        return {
            "outcome": "refused",
            "exc_type": "ContestStateError",
            "expected_line": f"automatic path alone: refused: {first}",
        }
    except Exception as exc:
        detail = (str(exc) or type(exc).__name__).replace(str(picks), "<PICKS>").splitlines()[0]
        return {
            "outcome": "error",
            "exc_type": type(exc).__name__,
            "expected_line": f"automatic path alone: refused: {detail}",
        }
    finally:
        if saved is not None:
            manual_path.write_bytes(saved)


results = {"ordinary": {}, "special": {}, "cutover_fifo": {}}

with TemporaryDirectory(dir=REVIEW, prefix=f"status-r2-{REV}-") as tmp:
    root = Path(tmp)

    def check_case(name, *, now=NOW, sw=True, auto=True, manual=None, saver_val=None,
                   model_val=None, picks_map=None, extra_args=(), category="ordinary"):
        picks = root / category / name / "picks"
        (picks / "account_state").mkdir(parents=True)
        write_item(picks / "account_state/season_contract.json", make_switch(now) if sw is True else sw)
        write_item(picks / "account_state/contest_streak.json", make_auto(now) if auto is True else auto)
        write_item(picks / "account_state/contest_streak.manual.json", make_dated(now) if manual is True else manual)
        write_item(picks / "account_state/saver_state.json", saver_val)
        write_item(picks / "streak.json", model_val)
        for day_str, body in (picks_map or {}).items():
            write_item(picks / f"{day_str}.json", body)

        in_effect_dated = (
            isinstance(manual, dict)
            and manual.get("schema_version") == "bts_contest_streak_manual_v3"
            and manual.get("effective_date") == now.astimezone(ET).date().isoformat()
        ) or (manual is True)

        before = tree_snapshot(picks.parent)
        with offline_env(now):
            st_res = run_status(picks, *extra_args)
            after = tree_snapshot(picks.parent)
            assert before == after, f"status modified files/mtimes in {name}"
            live_res = run_live_without_selected_dated(picks, now, in_effect_dated)

        if REV == "head" and category == "ordinary":
            assert st_res["outcome"] == "completed", (name, st_res)
            assert len(st_res["auto_lines"]) == 1, (name, st_res)
            assert st_res["auto_lines"][0] == live_res["expected_line"], (name, st_res["auto_lines"][0], live_res)
            if live_res["outcome"] != "accepted":
                assert "automatic path alone: accepted" not in "\n".join(st_res["lines"]), name

        results[category][name] = {"status": st_res, "live_no_dated": live_res}

    # 1. Full ordinary + adversarial regular-file matrix
    check_case("dated-absent-auto-accepted", auto=True, manual=None)
    check_case("dated-absent-auto-refused-v1", auto=make_auto(schema_version="bts_contest_streak_auto_v1"), manual=None)
    check_case("dated-present-auto-accepted", auto=True, manual=True)
    check_case("dated-present-auto-refused-v1", auto=make_auto(schema_version="bts_contest_streak_auto_v1"), manual=True)
    check_case("dated-present-auto-absent", auto=None, manual=True)
    check_case("expired-dated", auto=True, manual=make_dated(NOW - dt.timedelta(days=1)))
    check_case("future-dated-tomorrow", auto=True, manual=make_dated(NOW + dt.timedelta(days=1)))
    check_case("winning-legacy", auto=True, manual={
        "active_streak": 3, "override_expires_at": (NOW + dt.timedelta(days=2)).isoformat(),
    })
    check_case("expired-legacy-with-auto", auto=True, manual={
        "active_streak": 3, "override_expires_at": (NOW - dt.timedelta(days=2)).isoformat(),
    })
    check_case("expired-legacy-no-auto", auto=None, manual={
        "active_streak": 3, "override_expires_at": (NOW - dt.timedelta(days=2)).isoformat(),
    })
    check_case("corrupt-manual", auto=True, manual=b"{")
    check_case("corrupt-auto-masked", auto=b"[]", manual=True)
    check_case("corrupt-switch-masked", sw=b"\xff", auto=True, manual=True)
    check_case("wrong-activation-id-masked", auto=make_auto(activation_id="b" * 32), manual=True)
    check_case("missing-recorded-at", auto=make_auto(recorded_at=None), manual=True)
    check_case("naive-recorded-at", auto=make_auto(recorded_at="2027-07-15T11:30:00"), manual=True)
    check_case("stale-recorded-at", auto=make_auto(recorded_at=(NOW - dt.timedelta(hours=16)).isoformat()), manual=True)
    check_case("future-recorded-at", auto=make_auto(recorded_at=(NOW + dt.timedelta(hours=2)).isoformat()), manual=True)
    check_case("extreme-recorded-at", auto=make_auto(recorded_at="9999-12-31T23:59:59-01:00"), manual=True)
    check_case("uninitialized-saver-absent", auto=True, manual=True, saver_val=None)
    check_case("uninitialized-saver-other-season", auto=True, manual=True, saver_val={"season": 2026, "state": "active"})
    check_case("saver-huge-int-masked", auto=True, manual=True, saver_val=b"9" * 5000)
    check_case("saver-deep-nesting-masked", auto=True, manual=True, saver_val=b"[" * 3000 + b"]" * 3000)
    check_case("settlement-gap", auto=True, manual=True, picks_map={
        "2027-07-15": {"result": "hit"}, "2027-07-16": {"result": "miss"},
    })
    check_case("other-season-flag-july", auto=True, manual=True, extra_args=("--season", "2026"))
    dec31 = REAL_DT(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    jan01 = dec31 + dt.timedelta(seconds=2)
    check_case("dec31-before-midnight-other-season", now=dec31, sw=make_switch(dec31),
               auto=make_auto(dec31), manual=make_dated(dec31), extra_args=("--season", "2026"))
    check_case("jan01-after-midnight-other-season", now=jan01, sw=make_switch(dec31),
               auto=make_auto(dec31), manual=make_dated(dec31), extra_args=("--season", "2027"))

    # 2. FIFO and directory at all 7 paths (both unmasked and masked by dated where applicable)
    special_paths = [
        ("manual", "account_state/contest_streak.manual.json"),
        ("auto", "account_state/contest_streak.json"),
        ("switch", "account_state/season_contract.json"),
        ("saver", "account_state/saver_state.json"),
        ("ledger", "account_state/contest_ledger.jsonl"),
        ("model", "streak.json"),
        ("pick", "2027-07-15.json"),
    ]
    for label, rel in special_paths:
        for kind in ("fifo", "directory"):
            for masked in ([False] if label == "manual" else [False, True]):
                case_name = f"{label}-{kind}-{'masked' if masked else 'unmasked'}"
                picks = root / "special" / case_name / "picks"
                (picks / "account_state").mkdir(parents=True)
                write_item(picks / "account_state/season_contract.json", make_switch(NOW))
                write_item(picks / "account_state/contest_streak.json", make_auto(NOW))
                if masked:
                    write_item(picks / "account_state/contest_streak.manual.json", make_dated(NOW))
                write_item(picks / rel, kind)
                before = tree_snapshot(picks.parent)
                with offline_env(NOW):
                    st_res = run_status(picks)
                    after = tree_snapshot(picks.parent)
                    assert before == after, f"status modified special state in {case_name}"
                    live_res = run_live_without_selected_dated(picks, NOW, masked)
                if REV == "head":
                    # Never prints false "accepted" when live blocks or raises
                    if live_res["outcome"] != "accepted":
                        assert "automatic path alone: accepted" not in "\n".join(st_res.get("lines", [])), (
                            case_name, st_res, live_res
                        )
                    # On the 6 live-opened paths, status and live agree on block/refuse/accept
                    if label != "ledger":
                        if live_res["outcome"] == "blocked":
                            assert st_res["outcome"] == "blocked", (case_name, st_res, live_res)
                        else:
                            assert st_res["outcome"] == "completed", (case_name, st_res, live_res)
                            assert st_res["auto_lines"] == [live_res["expected_line"]], (case_name, st_res, live_res)
                results["special"][case_name] = {"status": st_res, "live_no_dated": live_res}

    # 3. Round-1 Finding 1 exact cutover reproduction:
    # 12:00 ET July 15, 2027: switch + today's dated observation + FIFO at saver_state.json;
    # switched fetch with active=7, best=9, predictions=[] exits 0 and installs matching auto-v2;
    # today's live selection succeeds on dated; check status and next-day live selection.
    c_picks = root / "cutover_fifo" / "picks"
    (c_picks / "account_state").mkdir(parents=True)
    write_item(c_picks / "account_state/season_contract.json", make_switch(NOW))
    write_item(c_picks / "account_state/contest_streak.manual.json", make_dated(NOW))
    write_item(c_picks / "account_state/saver_state.json", "fifo")
    with offline_env(NOW, predictions=[]):
        with bounded():
            f_res = CliRunner().invoke(cli.cli, [
                "fetch-contest-streak", "--picks-dir", str(c_picks), "--dm-recipient", "offline.test",
            ])
        assert f_res.exit_code == 0, f_res.output
        today_live = cs.load_decision_streak_state(c_picks, require_contest_state=True, now=NOW)
        assert today_live.observation_kind == "dated"
        st_cutover = run_status(c_picks)
    with offline_env(NOW + dt.timedelta(days=1), predictions=[]):
        next_day_live = run_live_without_selected_dated(c_picks, NOW + dt.timedelta(days=1), False)
    results["cutover_fifo"] = {
        "fetch_exit": f_res.exit_code,
        "today_live": today_live.observation_kind,
        "status": st_cutover,
        "next_day_live": next_day_live,
    }
    if REV == "head":
        assert st_cutover["outcome"] == "blocked" and next_day_live["outcome"] == "blocked"

out_file = REVIEW / f"status-{REV}.json"
out_file.write_text(json.dumps(results, indent=2) + "\n")
print(
    f"{REV}: {len(results['ordinary'])} ordinary cases, "
    f"{len(results['special'])} special cases, "
    f"cutover_fifo status={results['cutover_fifo']['status']['outcome']} "
    f"next_day={results['cutover_fifo']['next_day_live']['outcome']}"
)
