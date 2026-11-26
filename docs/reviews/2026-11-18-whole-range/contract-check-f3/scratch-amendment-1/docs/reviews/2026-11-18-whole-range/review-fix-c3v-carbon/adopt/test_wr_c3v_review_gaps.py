"""Regression tests closing surviving C3v condition-order and not-reached mutants."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import signal
import socket
import sys
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

from bts import picks, reconcile_receipt
from tests.test_wr_c3v_corpus import dump, pick_data, seed_streak
from tests.test_wr_c3v_reconcile import BASE, PinnedDatetime, base_modules, invoke, only_receipt, outcome, warning


ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, 0, tzinfo=ET)
TODAY = "2027-04-20"
YEST = "2027-04-19"
PAST = "2027-04-16"


@pytest.fixture(autouse=True)
def _guard_io_and_network(monkeypatch):
    class _DeadlineExceeded(BaseException):
        pass

    prev = signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_DeadlineExceeded("I/O deadline exceeded")))
    signal.alarm(15)

    def _no_net(*args, **kwargs):
        pytest.fail(f"Unexpected network call: {args}")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    monkeypatch.setattr(socket, "getaddrinfo", _no_net)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def test_today_resolved_hit_with_applied_marker_still_replays(tmp_path, monkeypatch, base_modules):
    """Kills M5b: checking today in load_applied_dates before r not in ('hit', 'miss', 'void')
    would block replay on today's already-scored pick."""
    root = tmp_path / "current" / "picks"
    target = seed_streak(root, streak=99, saver=False, season=2027, markers=(TODAY,))
    dump(root / f"{TODAY}.json", pick_data(TODAY, "hit"))

    old_picks, old_receipt, old_cli = base_modules
    old_root = tmp_path / "base" / "picks"
    shutil.copytree(root, old_root)

    with monkeypatch.context() as scoped:
        for mod in (old_picks, picks):
            scoped.setattr(mod, "datetime", PinnedDatetime)
            scoped.setattr(mod, "_git_head_sha", lambda *_a: BASE)
        with scoped.context() as archived:
            archived.setitem(sys.modules, "bts.picks", old_picks)
            expected = invoke(
                old_root, archived, clock=lambda: RUN, command=old_cli.cli,
                receipt_class=old_receipt.ReconcileReceipt, run_id="a" * 32,
            )
        actual = invoke(root, scoped, clock=lambda: RUN, run_id="a" * 32)

    assert actual.exit_code == expected.exit_code == 0
    assert actual.stderr == ""
    assert actual.stdout == "Reconciling last 0 days of picks...\nNo scoring changes detected. Streak: 1\n"
    assert target.read_bytes() == (old_root / "streak.json").read_bytes()
    assert only_receipt(root)["replay"] == {"state": "saved", "streak": 1, "saver_available": True}


def test_same_file_blocker_condition_precedence(tmp_path, monkeypatch, base_modules):
    """Kills M5c (today applied marker vs corrupt JSON) and M5d (load_pick None vs pending),
    and exercises real concurrent file disappearance on a hit without monkeypatching load_pick."""
    # (a) Corrupt today.json while TODAY is in applied_dates -> corrupt JSON reason wins over applied marker.
    r_a = tmp_path / "case_a" / "picks"
    seed_streak(r_a, streak=7, season=2027, markers=(TODAY,))
    (r_a / f"{TODAY}.json").write_text("{corrupt_today")
    res_a = invoke(r_a, monkeypatch, clock=lambda: RUN)
    assert res_a.exit_code == 0
    assert res_a.stderr == warning(TODAY, "pick file unreadable or not valid JSON")
    assert only_receipt(r_a)["replay"] == {
        "state": "unavailable",
        "blocked_date": TODAY,
        "reason": "pick file unreadable or not valid JSON",
    }

    # (b) Unresolved past pick when load_pick returns None -> pending reason wins over lost_hit.
    r_b = tmp_path / "case_b" / "picks"
    seed_streak(r_b, streak=7, season=2027)
    dump(r_b / f"{PAST}.json", pick_data(PAST, None))
    with monkeypatch.context() as mp:
        mp.setattr(picks, "load_pick", lambda *_a: None)
        res_b = invoke(r_b, mp, clock=lambda: RUN)
    assert res_b.exit_code == 0
    assert res_b.stderr == warning(PAST, "no final result on the pick")
    assert only_receipt(r_b)["replay"] == {
        "state": "unavailable",
        "blocked_date": PAST,
        "reason": "no final result on the pick",
    }

    # (c) Concurrent unlink of a resolved hit between f.read_text() and load_pick (no load_pick mock).
    r_c = tmp_path / "case_c" / "current" / "picks"
    target_c = seed_streak(r_c, streak=7, season=2027)
    before_c = target_c.read_bytes()
    dump(r_c / f"{PAST}.json", pick_data(PAST, "hit"))
    old_picks, old_receipt, old_cli = base_modules
    old_c = tmp_path / "case_c" / "base" / "picks"
    shutil.copytree(r_c, old_c)

    real_read_text = Path.read_text
    unlinked: set[str] = set()

    def _unlink_after_first_read(self_path: Path, *args, **kwargs):
        txt = real_read_text(self_path, *args, **kwargs)
        key = str(self_path)
        if self_path.name == f"{PAST}.json" and key not in unlinked:
            unlinked.add(key)
            self_path.unlink()
        return txt

    with monkeypatch.context() as mp:
        mp.setattr(Path, "read_text", _unlink_after_first_read)
        for mod in (old_picks, picks):
            mp.setattr(mod, "_git_head_sha", lambda *_a: BASE)
        with mp.context() as archived:
            archived.setitem(sys.modules, "bts.picks", old_picks)
            exp_c = invoke(
                old_c, archived, clock=lambda: RUN, command=old_cli.cli,
                receipt_class=old_receipt.ReconcileReceipt, run_id="a" * 32,
            )
        act_c = invoke(r_c, mp, clock=lambda: RUN, run_id="a" * 32)

    assert act_c.exit_code == exp_c.exit_code == 0
    assert act_c.stderr == warning(PAST, "hit pick could not be loaded")
    assert only_receipt(r_c)["replay"] == {
        "state": "unavailable",
        "blocked_date": PAST,
        "reason": "hit pick could not be loaded",
    }
    assert target_c.read_bytes() == (old_c / "streak.json").read_bytes() == before_c


def test_not_reached_with_corrections_uses_existing_recalculated_wording(tmp_path, monkeypatch):
    """Kills M6: when receipt.replay stays {'state': 'not_reached'} and corrections is non-empty,
    bts reconcile must preserve the existing 'Streak recalculated: N' wording."""
    corrections = [{"date": YEST, "batter": "Synthetic Batter", "old_result": "hit", "new_result": "miss"}]
    monkeypatch.setattr(picks, "reconcile_results", lambda *args, **kwargs: corrections)
    root = tmp_path / "picks"
    res = invoke(root, monkeypatch, clock=lambda: RUN)
    assert res.exit_code == 0, res.output
    assert res.stderr == ""
    assert res.stdout == (
        "Reconciling last 0 days of picks...\n"
        "CORRECTIONS FOUND (1):\n"
        f"  {YEST}: Synthetic Batter — hit -> miss\n"
        "Streak recalculated: 0\n"
    )
    assert only_receipt(root)["replay"] == {"state": "not_reached"}


@pytest.mark.parametrize("raw_streak", [None, b"{corrupt", b"[]"])
def test_missing_and_corrupt_streak_json_match_base_side_by_side(tmp_path, monkeypatch, base_modules, raw_streak):
    root = tmp_path / "current" / "picks"
    root.mkdir(parents=True)
    if raw_streak is not None:
        (root / "streak.json").write_bytes(raw_streak)
    dump(root / f"{PAST}.json", pick_data(PAST, None))

    old_picks, old_receipt, old_cli = base_modules
    old_root = tmp_path / "base" / "picks"
    shutil.copytree(root, old_root)

    with monkeypatch.context() as scoped:
        for mod in (old_picks, picks):
            scoped.setattr(mod, "datetime", PinnedDatetime)
            scoped.setattr(mod, "_git_head_sha", lambda *_a: BASE)
        assert outcome(lambda: picks._replay_season_streak(root, 2027, TODAY)) == outcome(
            lambda: old_picks._replay_season_streak(old_root, 2027, TODAY)
        )
        with scoped.context() as archived:
            archived.setitem(sys.modules, "bts.picks", old_picks)
            expected = invoke(
                old_root, archived, clock=lambda: RUN, command=old_cli.cli,
                receipt_class=old_receipt.ReconcileReceipt, run_id="a" * 32,
            )
        actual = invoke(root, scoped, clock=lambda: RUN, run_id="a" * 32)

    assert actual.exit_code == expected.exit_code
    assert (type(actual.exception) if actual.exception else None) is (
        type(expected.exception) if expected.exception else None
    )
    assert (root / "streak.json").exists() == (old_root / "streak.json").exists() == (raw_streak is not None)
    if raw_streak is not None:
        assert (root / "streak.json").read_bytes() == (old_root / "streak.json").read_bytes() == raw_streak
    else:
        assert actual.stdout.endswith("No scoring changes detected. Streak retained: 0\n")
    assert actual.stderr == warning(PAST, "no final result on the pick")


def test_failed_receipt_publication_still_prints_blocker_and_retains_streak(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    seed_streak(picks_dir, streak=7, season=2027)
    dump(picks_dir / f"{PAST}.json", pick_data(PAST, None))
    dump(picks_dir / f"{YEST}.json", pick_data(YEST, "hit", slot_results={"pick": "hit"}))

    health_dir = tmp_path / "data" / "health_state"
    health_dir.mkdir(parents=True)
    (health_dir / "reconcile_receipts").write_text("file blocking directory creation")

    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "miss"})
    res = invoke(picks_dir, monkeypatch, clock=lambda: RUN, lookback=8)
    assert res.exit_code == 0
    err_lines = res.stderr.splitlines()
    assert len(err_lines) == 2
    assert err_lines[0].startswith("reconcile: reconcile receipt unavailable (")
    assert err_lines[1] == (
        f"Season replay unavailable for {PAST} (no final result on the pick); streak retained, not recalculated."
    )
    assert res.stdout.endswith("Streak retained: 7\n")
    assert "recalculated" not in res.stdout.lower()
