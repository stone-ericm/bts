"""Round-2 reviewer adoptable regression tests for surviving scan and receipt-guard mutants."""
from __future__ import annotations

from datetime import datetime
import signal
import socket
from zoneinfo import ZoneInfo

import pytest

from bts import picks
from tests.test_wr_c3v_corpus import seed_streak
from tests.test_wr_c3v_reconcile import PENDING, base_modules, compare_reconcile, invoke, only_receipt, warning


ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, 0, tzinfo=ET)


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


def test_real_non_ascii_digit_pick_filename_blocks_like_base_and_renders_escaped(tmp_path, monkeypatch, base_modules):
    """Kills adding re.ASCII to _ISO_DATE_RE: real Unicode-digit stems accepted on 88b69a16
    still block identically while rendering escaped on a single ASCII stderr line."""
    root = tmp_path / "current" / "picks"
    target = seed_streak(root, streak=9, season=2027)
    before = target.read_bytes()
    stem = "2027-03-\u0661\u0666"
    (root / f"{stem}.json").write_text("{}")

    result, fold = compare_reconcile(root, monkeypatch, base_modules, clock=lambda: RUN, lookback=8)
    assert fold is None
    assert result.exit_code == 0, result.output
    assert result.stderr == warning(r"2027-03-\u0661\u0666", PENDING)
    assert result.stdout == (
        "Reconciling last 8 days of picks...\nNo scoring changes detected. Streak retained: 9\n"
    )
    assert only_receipt(root)["replay"] == {
        "state": "unavailable",
        "blocked_date": stem,
        "reason": PENDING,
    }
    assert target.read_bytes() == before


def test_unavailable_receipt_without_blocker_detail_does_not_raise(tmp_path, monkeypatch):
    """Kills dropping 'and \"blocked_date\" in replay' in cli.py when ReconcileReceipt.replayed(None)
    records a detail-less {'state': 'unavailable'} object."""
    def _bare_unavailable(*args, receipt, **kwargs):
        receipt.replayed(None)
        return []

    monkeypatch.setattr(picks, "reconcile_results", _bare_unavailable)
    root = tmp_path / "picks"
    res = invoke(root, monkeypatch, clock=lambda: RUN)
    assert res.exit_code == 0, res.output
    assert res.stderr == ""
    assert res.stdout == "Reconciling last 0 days of picks...\nNo scoring changes detected. Streak: 0\n"
    assert only_receipt(root)["replay"] == {"state": "unavailable"}
