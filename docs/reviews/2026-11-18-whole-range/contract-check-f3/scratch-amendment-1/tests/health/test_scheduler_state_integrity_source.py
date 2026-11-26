"""Health source surfacing quarantined scheduler state files (audit F3).

load_state quarantines a corrupt scheduler_state.json (*.corrupt-<ts>) and the
daemon proceeds with fresh state — operationally right, but silent. This source
makes the quarantine visible: day-state (pick_locked / skip context) was reset,
an operator should glance at the evidence file.
"""
from datetime import date

from bts.health.scheduler_state_integrity import check


def _quarantine(picks_dir, day, stamp="20260709T120000Z"):
    d = picks_dir / day
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"scheduler_state.json.corrupt-{stamp}"
    f.write_text("{torn")
    return f


TODAY = date(2026, 7, 9)


def test_no_quarantine_files_stays_quiet(tmp_path):
    (tmp_path / "2026-07-09").mkdir()
    assert check(tmp_path, today=TODAY) == []


def test_quarantine_today_warns_and_names_file(tmp_path):
    _quarantine(tmp_path, "2026-07-09")
    alerts = check(tmp_path, today=TODAY)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == "scheduler_state_integrity"
    assert "2026-07-09" in alerts[0].message
    assert "corrupt-20260709T120000Z" in alerts[0].message


def test_quarantine_within_lookback_warns(tmp_path):
    _quarantine(tmp_path, "2026-07-07")
    assert len(check(tmp_path, today=TODAY)) == 1


def test_quarantine_beyond_lookback_stays_quiet(tmp_path):
    _quarantine(tmp_path, "2026-07-01")
    assert check(tmp_path, today=TODAY) == []


def test_multiple_quarantines_single_alert(tmp_path):
    _quarantine(tmp_path, "2026-07-08", "20260708T010000Z")
    _quarantine(tmp_path, "2026-07-09", "20260709T100000Z")
    alerts = check(tmp_path, today=TODAY)
    assert len(alerts) == 1
    assert "2026-07-08" in alerts[0].message and "2026-07-09" in alerts[0].message
