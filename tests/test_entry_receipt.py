"""Watchdog plan P1: the pick-entry receipt from the existing check-pick-entered run (registration R4 and the producer
receipt contract, docs/sota_audit/2026-10-04-prereg-c1-watchdog.md; schema docs/ops/pick-entry-receipt-v1.md).

Every run publishes one receipt, whatever its outcome, and the run's behaviour is unchanged. That covers its DMs, its
marker statuses, its exit codes and its authenticated request counts. The auth, contest and DM leaves are patched
(the TestCheckPickEntered harness), so nothing here touches a real account or sends anything.
"""
import json
from datetime import datetime

import httpx
import pytest

import tests.test_cli_integration as _cli_tests     # not imported by name: pytest would collect the class again

H = _cli_tests.TestCheckPickEntered()
DATE = "2026-06-12"
PITCH = "2026-06-12T23:10:00+00:00"          # 19:10 ET; the submission cutoff is 19:05 ET
IN_WINDOW = "2026-06-12T18:30:00"            # 40 min to pitch


def receipts(tmp_path, date=DATE):
    d = tmp_path / "health_state" / "pick_entry_receipts" / date
    files = sorted(d.glob("*.json")) if d.exists() else []
    return [json.loads(f.read_text()) for f in files]


def only(tmp_path):
    (r,) = receipts(tmp_path)
    return r


def aware(s):
    t = datetime.fromisoformat(s)
    assert t.tzinfo is not None, s
    return t


def picks_dir(tmp_path, **kw):
    p = tmp_path / "picks"
    p.mkdir()
    if kw.pop("save", True):
        H._save_pick(p, DATE, PITCH, **kw)
    return p


def common(r):
    assert r["schema"] == "bts_pick_entry_receipt_v1" and len(r["attempt_id"]) == 32
    assert r["et_date"] == DATE and r["season"] == 2026 and r["producer"]["command"] == "check-pick-entered"
    assert aware(r["published_at"]) >= aware(r["started_at"])


def counting(monkeypatch):
    """Count every authenticated/contest request leaf (after the harness patches them)."""
    import bts.cli as climod
    import bts.contest_fetch as cf
    import bts.leaderboard.auth as auth
    n = {}
    for mod, name in ((auth, "fetch_login_session"), (cf, "fetch_profile"), (cf, "fetch_pending_predictions"),
                      (climod, "_fetch_rounds"), (climod, "_fetch_bts_to_mlb")):
        real = getattr(mod, name)

        def wrapped(*a, _real=real, _name=name, **k):
            n[_name] = n.get(_name, 0) + 1
            return _real(*a, **k)
        monkeypatch.setattr(mod, name, wrapped)
    return n


# ---- no-attempt outcomes ------------------------------------------------------------------------------------------

def test_no_pick_file(monkeypatch, tmp_path):
    H._setup(monkeypatch)
    picks = picks_dir(tmp_path, save=False)
    assert H._run(picks, IN_WINDOW).exit_code == 0
    r = only(tmp_path)
    common(r)
    assert r["outcome"] == "no_pick_file" and r["selection"] is None and r["observation"] is None


def test_not_committed(monkeypatch, tmp_path):
    H._setup(monkeypatch)
    picks = picks_dir(tmp_path, batter_id=1, delivered=False)
    assert H._run(picks, IN_WINDOW).exit_code == 0
    r = only(tmp_path)
    assert r["outcome"] == "not_committed" and r["observation"] is None and r["account"] is None


def test_outside_window_records_the_committed_selection_and_cutoff(monkeypatch, tmp_path):
    H._setup(monkeypatch)
    picks = picks_dir(tmp_path, batter_id=1, dd_batter_id=5)
    n = counting(monkeypatch)
    assert H._run(picks, "2026-06-12T12:00:00").exit_code == 0
    r = only(tmp_path)
    assert r["outcome"] == "outside_window" and n == {} and r["observation"] is None
    sel = r["selection"]
    assert [(s["role"], s["batter_id"], s["game_pk"]) for s in sel["slots"]] == [("pick", 1, 1), ("double_down", 5, 2)]
    assert sel["delivery"]["notification_id"] == "dm_x" and len(sel["pick_file_sha256"]) == 64
    assert aware(r["cutoff_at"]) == aware("2026-06-12T19:05:00-04:00")


# ---- attempted outcomes -------------------------------------------------------------------------------------------

def test_a_confirmed_observation_binds_account_selection_rows_and_times(monkeypatch, tmp_path):
    dms = H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    n = counting(monkeypatch)
    res = H._run(picks, IN_WINDOW)
    assert res.exit_code == 0 and dms == []
    assert n == {"fetch_login_session": 1, "fetch_profile": 1, "fetch_pending_predictions": 1, "_fetch_rounds": 1,
                 "_fetch_bts_to_mlb": 1}                                       # unchanged request count
    r = only(tmp_path)
    common(r)
    assert r["outcome"] == "observed"
    assert r["account"] == {"expected_username": None, "user_id": 50311, "username": "stonehengee"}
    obs = r["observation"]
    assert aware(r["started_at"]) <= aware(obs["response_completed_at"]) <= aware(r["published_at"])
    assert obs["before_cutoff"] is True
    assert obs["rows"]["pending"] == [H._pending(100)] and obs["rows"]["profile"] == []
    assert set(obs["sources"]) == {"profile_sha256", "pending_sha256", "rounds_sha256", "crosswalk_sha256"}
    assert obs["entered_bts_ids"] == [100] and obs["resolved_mlb_ids"] == [1]
    assert r["verifier"] == {"ok": True, "reason": "match", "required_mlb_ids": [1]}
    assert r["marker_status"] == "confirmed"
    assert H._status(tmp_path)["receipt"] == r["attempt_id"]


def test_an_observed_absence_is_a_successful_observation(monkeypatch, tmp_path):
    dms = H._setup(monkeypatch, pending=[], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    assert H._run(picks, IN_WINDOW).exit_code != 0 and len(dms) == 1             # behaviour unchanged
    r = only(tmp_path)
    assert r["outcome"] == "observed" and r["verifier"]["ok"] is False and r["verifier"]["reason"] == "no_pick"
    assert r["marker_status"] == "alerted" and r["exit_code"] == 1


def test_a_missing_double_down_leg_is_a_mismatch_observation(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1, 500: 5})
    picks = picks_dir(tmp_path, batter_id=1, dd_batter_id=5)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert r["verifier"]["reason"] == "mismatch" and r["verifier"]["required_mlb_ids"] == [1, 5]


@pytest.mark.parametrize("exc", [httpx.ConnectError("boom"), ValueError("bad shape")])
def test_a_fetch_failure_is_not_an_observation(monkeypatch, tmp_path, exc):
    import bts.contest_fetch as cf
    H._setup(monkeypatch)
    monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: (_ for _ in ()).throw(exc))
    picks = picks_dir(tmp_path, batter_id=1)
    assert H._run(picks, IN_WINDOW).exit_code == 0
    r = only(tmp_path)
    assert r["outcome"] == "fetch_failed" and r["error"]["type"] == type(exc).__name__
    assert r["observation"] is None and r["verifier"] is None and aware(r["failed_at"])
    assert not (tmp_path / "health_state" / "pick_entry_check.json").exists()       # no marker, as before


def test_an_identity_mismatch_records_the_observed_account_only(monkeypatch, tmp_path):
    H._setup(monkeypatch)
    H._patch_auth(monkeypatch, username="someone_else", user_id=7)
    picks = picks_dir(tmp_path, batter_id=1)
    assert H._run(picks, IN_WINDOW, extra=["--expected-username", "stonehengee"]).exit_code == 0
    r = only(tmp_path)
    assert r["outcome"] == "identity_mismatch" and r["observation"] is None
    assert r["account"] == {"expected_username": "stonehengee", "user_id": 7, "username": "someone_else"}


def test_already_confirmed_references_the_original_receipt_without_a_new_observation(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    first = only(tmp_path)
    n = counting(monkeypatch)
    assert H._run(picks, "2026-06-12T18:45:00").exit_code == 0
    second = [r for r in receipts(tmp_path) if r["attempt_id"] != first["attempt_id"]][0]
    assert n == {} and second["outcome"] == "already_confirmed"
    assert second["references"] == {"confirmed_by": first["attempt_id"]}
    assert second["observation"] is None and second["verifier"] is None


def test_a_late_observation_keeps_its_actual_completion_time(monkeypatch, tmp_path):
    """The fetch spans the cutoff (the existing 40-minute fake clock): the observation is late, not pre-cutoff."""
    import time as _time
    H._setup(monkeypatch, pending=[], crosswalk={100: 1})
    calls = {"n": 0}

    def _mono():
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 1000.0 + 40 * 60
    monkeypatch.setattr(_time, "monotonic", _mono)
    picks = picks_dir(tmp_path, batter_id=1)
    res = H._run(picks, IN_WINDOW)
    assert "cutoff passed during fetch" in res.output                        # the existing behaviour holds
    r = only(tmp_path)
    assert r["observation"]["before_cutoff"] is False
    assert aware(r["observation"]["response_completed_at"]) == aware("2026-06-12T19:10:00-04:00")


# ---- the contract around the receipt itself -----------------------------------------------------------------------

def test_no_cookie_or_token_is_stored(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    text = json.dumps(only(tmp_path))
    assert "x_1" not in text and "oktaid" not in text and "xsid" not in text and "cookie" not in text.lower()


def test_a_failed_publication_changes_nothing_else(monkeypatch, tmp_path):
    import bts.entry_receipt as er
    dms = H._setup(monkeypatch, pending=[], crosswalk={100: 1})
    monkeypatch.setattr(er, "_write_durable", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    picks = picks_dir(tmp_path, batter_id=1)
    res = H._run(picks, IN_WINDOW)
    assert res.exit_code != 0 and len(dms) == 1 and H._status(tmp_path)["status"] == "alerted"
    assert "receipt unavailable" in res.output and receipts(tmp_path) == []


def test_each_run_is_a_distinct_record(monkeypatch, tmp_path):
    H._setup(monkeypatch)
    picks = picks_dir(tmp_path, batter_id=1)
    for t in ("2026-06-12T11:00:00", "2026-06-12T11:15:00"):
        H._run(picks, t)
    rs = receipts(tmp_path)
    assert len(rs) == 2 and len({r["attempt_id"] for r in rs}) == 2
