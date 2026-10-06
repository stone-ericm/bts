"""Watchdog plan P1: the pick-entry receipt from the existing check-pick-entered run (registration R4 and the producer
receipt contract, docs/sota_audit/2026-10-04-prereg-c1-watchdog.md; schema docs/ops/pick-entry-receipt-v1.md;
producer review r1 C1-C6).

Every run publishes one receipt, whatever its outcome, and the run's behaviour is unchanged. That covers its DMs,
marker statuses, exit codes and authenticated request counts. The auth, contest and DM leaves are patched (the
TestCheckPickEntered harness), so nothing here touches a real account or sends anything.
"""
import gzip
import hashlib
import json
from datetime import datetime

import httpx
import pytest

import bts.entry_receipt as er
import tests.test_cli_integration as _cli_tests     # not imported by name: pytest would collect the class again

H = _cli_tests.TestCheckPickEntered()
DATE = "2026-06-12"
PITCH = "2026-06-12T23:10:00+00:00"          # 19:10 ET; the submission cutoff is 19:05 ET
IN_WINDOW = "2026-06-12T18:30:00"            # 40 min to pitch


def receipts(tmp_path, date=DATE):
    return er.discover(tmp_path / "picks", date)


def only(tmp_path):
    (r,) = receipts(tmp_path)
    return r


def aware(s):
    t = datetime.fromisoformat(s)
    assert t.tzinfo is not None, s
    return t


def picks_dir(tmp_path, **kw):
    p = tmp_path / "picks"
    p.mkdir(parents=True)
    if kw.pop("save", True):
        H._save_pick(p, DATE, PITCH, **kw)
    return p


def units_snapshot(tmp_path, units, name="20260612T120000Z"):
    d = tmp_path / "leaderboard" / "static_snapshots" / "units"
    d.mkdir(parents=True, exist_ok=True)
    raw = gzip.compress(json.dumps({"units": units}).encode(), mtime=0)
    (d / f"{name}.json.gz").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def common(r):
    assert r["schema"] == "bts_pick_entry_receipt_v1" and len(r["attempt_id"]) == 32
    assert r["et_date"] == DATE and r["season"] == 2026 and r["producer"]["command"] == "check-pick-entered"
    assert aware(r["published_at"]) >= aware(r["started_at"]) and r["degraded"] == []


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
    assert sel["delivery"]["notification_id"] == "dm_x"
    assert sel["pick_file_sha256"] == hashlib.sha256((picks / f"{DATE}.json").read_bytes()).hexdigest()
    assert aware(r["cutoff_at"]) == aware("2026-06-12T19:05:00-04:00")


# ---- attempted outcomes -------------------------------------------------------------------------------------------

def test_a_confirmed_observation_binds_account_selection_rows_times_and_game(monkeypatch, tmp_path):
    dms = H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    units_sha = units_snapshot(tmp_path, [{"id": 1, "feedId": 1, "roundId": 7, "homeSquadId": 3, "awaySquadId": 4}])
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
    assert set(obs["sources"]) == {"profile_sha256", "pending_sha256", "rounds_sha256", "crosswalk_sha256", "units"}
    assert [f["sha256"] for f in obs["sources"]["units"]["files"]] == [units_sha]
    assert r["verifier"] == {"ok": True, "reason": "match", "required_mlb_ids": [1], "game_qualified": False}
    q = r["qualification"]
    assert q["all_confirmed"] is True and q["slots"][0]["state"] == "confirmed"
    assert q["slots"][0]["row"] == {"roundId": 7, "unitId": 1, "playerId": 100, "number": 1}
    assert r["marker_status"] == "confirmed" and H._status(tmp_path)["receipt"] == r["attempt_id"]


@pytest.mark.parametrize("units, state", [
    ([{"id": 1, "feedId": 999, "roundId": 7}], "wrong_game"),                 # the unit names another game
    ([{"id": 2, "feedId": 1, "roundId": 7}], "unit_unverified"),             # the observed unit is unknown
    ([{"id": 1, "feedId": None, "roundId": 7}], "unit_unverified"),          # no game in the mapping
    ([{"id": 1, "feedId": 1, "roundId": 8}], "wrong_round"),
    ([{"id": 1, "feedId": 1, "roundId": 7}, {"id": 1, "feedId": 5, "roundId": 7}], "unit_unverified"),   # ambiguous
    (None, "unit_unverified"),                                               # no captured units.json at all
])
def test_a_batter_match_is_not_game_qualified_without_a_bound_unit(monkeypatch, tmp_path, units, state):
    """Producer review r1 C1: the legacy verifier says match on the batter alone; the receipt must not."""
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    if units is not None:
        units_snapshot(tmp_path, units)
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert r["verifier"]["reason"] == "match" and r["qualification"]["all_confirmed"] is False
    assert r["qualification"]["slots"][0]["state"] == state
    if units is None:
        assert r["observation"]["sources"]["units"] is None


def test_a_missing_double_down_leg_and_another_round_are_not_confirmation(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[H._pending(100), H._pending(500, round_id=99)], crosswalk={100: 1, 500: 5})
    units_snapshot(tmp_path, [{"id": 1, "feedId": 1, "roundId": 7}])
    picks = picks_dir(tmp_path, batter_id=1, dd_batter_id=5)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert r["verifier"]["reason"] == "mismatch" and r["verifier"]["required_mlb_ids"] == [1, 5]
    assert [(s["role"], s["state"]) for s in r["qualification"]["slots"]] == [("pick", "confirmed"),
                                                                              ("double_down", "missing")]
    assert r["observation"]["rows"]["pending"] == [H._pending(100)]           # round 99 is another date


def test_an_unresolved_player_is_unverified_not_missing(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[H._pending(777)], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    assert only(tmp_path)["qualification"]["slots"][0]["state"] == "player_unverified"


def test_an_observed_absence_is_a_successful_observation(monkeypatch, tmp_path):
    dms = H._setup(monkeypatch, pending=[], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    assert H._run(picks, IN_WINDOW).exit_code != 0 and len(dms) == 1             # behaviour unchanged
    r = only(tmp_path)
    assert r["outcome"] == "observed" and r["verifier"]["ok"] is False and r["verifier"]["reason"] == "no_pick"
    assert r["qualification"]["slots"][0]["state"] == "missing"
    assert r["marker_status"] == "alerted" and r["exit_code"] == 1


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


# ---- C2: the selection hashes are the bytes the run parsed --------------------------------------------------------

def test_the_selection_binds_the_pick_bytes_parsed_not_a_later_file(monkeypatch, tmp_path):
    import bts.picks as picks_mod
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    original = (picks / f"{DATE}.json").read_bytes()
    real = picks_mod.load_pick_bytes

    def load_then_replace(date, d):
        out = real(date, d)
        other = tmp_path / "other"
        other.mkdir(exist_ok=True)
        H._save_pick(other, DATE, PITCH, batter_id=9)
        (picks / f"{DATE}.json").write_bytes((other / f"{DATE}.json").read_bytes())
        return out
    monkeypatch.setattr(picks_mod, "load_pick_bytes", load_then_replace)
    H._run(picks, IN_WINDOW)
    sel = only(tmp_path)["selection"]
    assert sel["slots"][0]["batter_id"] == 1
    assert sel["pick_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert sel["pick_file_sha256"] != hashlib.sha256((picks / f"{DATE}.json").read_bytes()).hexdigest()


def test_the_commit_binds_the_decision_bytes_gated_on(monkeypatch, tmp_path):
    import bts.daily_decision as dd
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1, delivered=False)
    H._write_decision(picks, DATE, action="single", scoreable=True, delivery_status="private_locked")
    path = dd.decision_path(DATE, picks)
    original = path.read_bytes()
    real = dd.load_decision_bytes

    def load_then_replace(date, d):
        out = real(date, d)
        H._write_decision(picks, DATE, action="skip", scoreable=False, delivery_status="not_applicable")
        return out
    monkeypatch.setattr(dd, "load_decision_bytes", load_then_replace)
    H._run(picks, IN_WINDOW)
    commit = only(tmp_path)["selection"]["commit"]
    assert commit["decision_sha256"] == hashlib.sha256(original).hexdigest() != hashlib.sha256(path.read_bytes()).hexdigest()
    assert (commit["scoreable"], commit["delivery_status"]) == (True, "private_locked")


# ---- C3: no secret reaches a receipt ------------------------------------------------------------------------------

SECRETS = ("SYNTHETIC_SESSION_SECRET", "SYNTHETIC_TOKEN", "SYNTHETIC_COOKIE")


def test_no_secret_reaches_a_receipt_from_rows_or_errors(monkeypatch, tmp_path):
    row = {**H._pending(100), "xsid": SECRETS[0], "token": SECRETS[1], "meta": {"cookie": SECRETS[2]}}
    prof = [{"roundId": 7, "token": SECRETS[1],
             "roundPredictions": [{"number": 1, "unitId": 1, "playerId": 100, "result": None, "xsid": SECRETS[0]}]}]
    H._setup(monkeypatch, pending=[row], profile_preds=prof, crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    text = json.dumps(only(tmp_path))
    assert not any(s in text for s in SECRETS) and "x_1" not in text and "oktaid" not in text


def test_no_secret_reaches_a_receipt_from_an_exception_message(monkeypatch, tmp_path):
    import bts.contest_fetch as cf
    H._setup(monkeypatch)
    req = httpx.Request("GET", f"https://example.invalid/p?xsid={SECRETS[0]}")
    err = httpx.HTTPStatusError(f"401 for {req.url} token={SECRETS[1]}", request=req,
                                response=httpx.Response(401, request=req))
    monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: (_ for _ in ()).throw(err))
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert r["error"] == {"type": "HTTPStatusError", "http_status": 401}
    assert not any(s in json.dumps(r) for s in SECRETS)


def test_mistyped_row_fields_are_dropped_and_named():
    assert er.clean_row({"roundId": 7, "unitId": "1", "playerId": True, "number": 1, "result": 3}) == \
        {"roundId": 7, "unitId": None, "playerId": None, "number": 1, "result": None,
         "mistyped": ["playerId", "result", "unitId"]}


# ---- C4: the response time is the last fetch's return, on fixed clocks ----------------------------------------------

def scripted(monkeypatch, *, response_s, verify_s=0.0):
    """Fix the monotonic clock: the fetch starts at 1000 s; the last lookup returns at 1000 + response_s seconds;
    the verifier then takes verify_s seconds."""
    import time as _time
    import bts.cli as climod
    import bts.contest_fetch as cf
    t = {"now": 1000.0}
    monkeypatch.setattr(_time, "monotonic", lambda: t["now"])
    real_xw, real_verify = climod._fetch_bts_to_mlb, cf.pick_entry_status

    def last_lookup(*a, **k):
        t["now"] = 1000.0 + response_s
        return real_xw(*a, **k)

    def slow_verify(*a, **k):
        out = real_verify(*a, **k)
        t["now"] += verify_s
        return out
    monkeypatch.setattr(climod, "_fetch_bts_to_mlb", last_lookup)
    monkeypatch.setattr(cf, "pick_entry_status", slow_verify)


@pytest.mark.parametrize("response_s, verify_s, done, before", [
    (34 * 60 + 59, 2.0, "2026-06-12T19:04:59-04:00", True),     # responses before the cutoff, verified after it
    (35 * 60, 0.0, "2026-06-12T19:05:00-04:00", False),          # exactly at the cutoff
    (36 * 60, 0.0, "2026-06-12T19:06:00-04:00", False),          # genuinely late
])
def test_the_response_time_is_captured_before_verification(monkeypatch, tmp_path, response_s, verify_s, done, before):
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    scripted(monkeypatch, response_s=response_s, verify_s=verify_s)
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    obs = only(tmp_path)["observation"]
    assert aware(obs["response_completed_at"]) == aware(done) and obs["before_cutoff"] is before


def test_an_earlier_double_down_leg_sets_the_cutoff(monkeypatch, tmp_path):
    from bts.picks import DailyPick, Pick, save_pick
    H._setup(monkeypatch, pending=[H._pending(100), H._pending(500)], crosswalk={100: 1, 500: 5})
    scripted(monkeypatch, response_s=60)
    picks = tmp_path / "picks"
    picks.mkdir()

    def mk(bid, gpk, t):
        return Pick(batter_name=f"B{bid}", batter_id=bid, team="NYY", lineup_position=1, pitcher_name="P", pitcher_id=2,
                    p_game_hit=0.8, flags=[], projected_lineup=False, game_pk=gpk, game_time=t)
    save_pick(DailyPick(date=DATE, run_time="x", pick=mk(1, 1, PITCH), double_down=mk(5, 2, "2026-06-12T22:40:00+00:00"),
                        runner_up=None, notification_sent=True, notification_channel="bluesky_dm",
                        notification_id="dm_x"), picks)
    H._run(picks, "2026-06-12T18:00:00")                         # 40 min before the DD leg (18:40 ET)
    r = only(tmp_path)
    assert aware(r["cutoff_at"]) == aware("2026-06-12T18:35:00-04:00")
    assert aware(r["observation"]["response_completed_at"]) == aware("2026-06-12T18:01:00-04:00")
    assert r["observation"]["before_cutoff"] is True


def test_a_late_observation_keeps_its_actual_completion_time(monkeypatch, tmp_path):
    """The existing 40-minute fake clock: the run's own behaviour holds and the observation is late."""
    import time as _time
    H._setup(monkeypatch, pending=[], crosswalk={100: 1})
    calls = {"n": 0}

    def _mono():
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 1000.0 + 40 * 60
    monkeypatch.setattr(_time, "monotonic", _mono)
    picks = picks_dir(tmp_path, batter_id=1)
    res = H._run(picks, IN_WINDOW)
    assert "cutoff passed during fetch" in res.output
    r = only(tmp_path)
    assert r["observation"]["before_cutoff"] is False
    assert aware(r["observation"]["response_completed_at"]) == aware("2026-06-12T19:10:00-04:00")


# ---- C5: receipt work never changes the run ------------------------------------------------------------------------

def _observable(monkeypatch, tmp_path, inject=None):
    dms = H._setup(monkeypatch, pending=[], crosswalk={100: 1})
    if inject:
        inject()
    picks = picks_dir(tmp_path, batter_id=1)
    res = H._run(picks, IN_WINDOW)
    status = H._status(tmp_path)
    return (len(dms), status["status"], status["reason"], status.get("escalations"), res.exit_code), res


@pytest.mark.parametrize("target", ["payload_sha256", "target_rows", "qualify", "load_units", "_sha"])
def test_a_failure_in_receipt_computation_changes_nothing(monkeypatch, tmp_path, target):
    base, _ = _observable(monkeypatch, tmp_path / "base")
    got, res = _observable(monkeypatch, tmp_path / "inj",
                           lambda: monkeypatch.setattr(er, target, lambda *a, **k: (_ for _ in ()).throw(MemoryError())))
    assert got == base and base[0] == 1 and base[1] == "alerted"
    assert "entry receipt unavailable" in res.output and receipts(tmp_path / "inj") == []


def test_a_failing_hook_is_contained_and_recorded(monkeypatch, tmp_path):
    base, _ = _observable(monkeypatch, tmp_path / "base")
    got, _ = _observable(monkeypatch, tmp_path / "inj", lambda: monkeypatch.setattr(
        er.EntryReceipt, "responses_done", er._guarded(lambda self, *a: 1 / 0)))
    assert got == base
    assert only(tmp_path / "inj")["degraded"] == ["<lambda>:ZeroDivisionError"]


def test_a_serialization_failure_changes_nothing(monkeypatch, tmp_path):
    base, _ = _observable(monkeypatch, tmp_path / "base")
    got, res = _observable(monkeypatch, tmp_path / "inj", lambda: monkeypatch.setattr(
        er, "_serialize", lambda *a, **k: (_ for _ in ()).throw(TypeError("x"))))
    assert got == base and receipts(tmp_path / "inj") == []


# ---- C6: a failed publication is never discoverable ----------------------------------------------------------------

@pytest.mark.parametrize("step", ["write", "file_fsync", "replace", "dir_fsync", "dir_fsync_and_cleanup"])
def test_a_publication_failure_at_any_step_leaves_no_discoverable_receipt(monkeypatch, tmp_path, step):
    import os
    import bts.receipt_io as rio
    base, _ = _observable(monkeypatch, tmp_path / "base")
    real_fsync, real_replace, real_dirsync, real_unlink = os.fsync, os.replace, rio._fsync_dir, type(tmp_path).unlink
    state = {"replaced": False}

    def inject():
        if step == "write":
            real_open = open

            def bad_open(p, mode="r", *a, **k):
                if str(p).endswith(".tmp") and "pick_entry_receipts" in str(p):
                    raise OSError("write")
                return real_open(p, mode, *a, **k)
            monkeypatch.setattr("builtins.open", bad_open)
        if step == "file_fsync":
            monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync"))
                                if state.get("arm") else real_fsync(fd))
            monkeypatch.setattr(rio, "ensure_dir", lambda p: (rio.Path(p).mkdir(parents=True, exist_ok=True),
                                                             state.__setitem__("arm", True)))
        if step == "replace":
            monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("replace"))
                                if "pick_entry_receipts" in str(b) else real_replace(a, b))
        if step in ("dir_fsync", "dir_fsync_and_cleanup"):
            def replace_then_arm(a, b):
                real_replace(a, b)
                if "pick_entry_receipts" in str(b):
                    state["replaced"] = True
            monkeypatch.setattr(os, "replace", replace_then_arm)
            monkeypatch.setattr(rio, "_fsync_dir", lambda p: (_ for _ in ()).throw(OSError("dirsync"))
                                if state["replaced"] else real_dirsync(p))
        if step == "dir_fsync_and_cleanup":
            monkeypatch.setattr(type(tmp_path), "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(
                OSError("unlink")) if self.suffix == ".json" and "pick_entry_receipts" in str(self)
                else real_unlink(self, missing_ok=missing_ok))
    got, res = _observable(monkeypatch, tmp_path / "inj", inject)
    assert got == base and "entry receipt unavailable" in res.output
    assert receipts(tmp_path / "inj") == []
    left = list((tmp_path / "inj" / "health_state" / "pick_entry_receipts").rglob("*"))
    assert not [p for p in left if p.name.endswith(".tmp")]
    if step == "dir_fsync_and_cleanup":
        (final,) = [p for p in left if p.suffix == ".json"]
        assert final.with_name(final.name + ".failed").exists()             # tombstoned, so never discovered


def test_first_use_directory_creation_syncs_each_new_parent(monkeypatch, tmp_path):
    import bts.receipt_io as rio
    synced = []
    real = rio._fsync_dir
    monkeypatch.setattr(rio, "_fsync_dir", lambda p: (synced.append(rio.Path(p)), real(p))[1])
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    root = tmp_path / "health_state" / "pick_entry_receipts"
    for parent in (tmp_path / "health_state", root, root / DATE):
        assert parent in synced, parent


def test_each_run_is_a_distinct_record(monkeypatch, tmp_path):
    H._setup(monkeypatch)
    picks = picks_dir(tmp_path, batter_id=1)
    for t in ("2026-06-12T11:00:00", "2026-06-12T11:15:00"):
        H._run(picks, t)
    rs = receipts(tmp_path)
    assert len(rs) == 2 and len({r["attempt_id"] for r in rs}) == 2



# ---- producer review r2: C1, C3, C7, C8 ---------------------------------------------------------------------------

def test_c1_a_capture_history_that_contradicts_itself_is_not_confirmation(monkeypatch, tmp_path):
    """Two captures bind unit 1 / round 7 first to game 9 and then to game 1: ambiguous, never the latest."""
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    units_snapshot(tmp_path, [{"id": 1, "feedId": 9, "roundId": 7}], name="20260610T120000Z")
    units_snapshot(tmp_path, [{"id": 1, "feedId": 1, "roundId": 7}], name="20260612T120000Z")
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert r["qualification"]["slots"][0]["state"] == "unit_unverified" and not r["qualification"]["all_confirmed"]
    assert len(r["observation"]["sources"]["units"]["files"]) == 2


def test_c1_other_seasons_are_not_consumed_and_a_null_feed_is_no_contradiction(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    units_snapshot(tmp_path, [{"id": 1, "feedId": 5, "roundId": 7}], name="20250610T120000Z")     # another season
    units_snapshot(tmp_path, [{"id": 1, "feedId": None, "roundId": 7}], name="20260601T120000Z")  # not yet bound
    units_snapshot(tmp_path, [{"id": 1, "feedId": 1, "roundId": 7}], name="20260612T120000Z")
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert r["qualification"]["all_confirmed"] is True
    assert [f["captured_utc"] for f in r["observation"]["sources"]["units"]["files"]] == ["20260601T120000Z",
                                                                                        "20260612T120000Z"]


def test_c3_a_result_outside_the_contest_domain_is_nulled_and_named(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[{**H._pending(100), "result": "SYNTHETIC_TOKEN_IN_RESULT"}], crosswalk={100: 1})
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert "SYNTHETIC_TOKEN_IN_RESULT" not in json.dumps(r)
    assert r["observation"]["rows"]["pending"][0]["mistyped"] == ["result"]
    assert er.clean_row({**H._pending(100), "result": "not_hit"})["result"] == "not_hit"


@pytest.mark.parametrize("raw", [b'{\r\n"x":\r\n}', b'{\r"x":\r}', b'{"date": "2026-06-12"\r\n'])
def test_c7_held_bytes_decode_exactly_as_read_text(tmp_path, raw):
    import bts.picks as picks_mod
    f = tmp_path / "2026-06-12.json"
    f.write_bytes(raw)
    def err(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__, str(exc)
    assert err(lambda: picks_mod.load_pick("2026-06-12", tmp_path)) == \
        err(lambda: picks_mod.load_pick_bytes("2026-06-12", tmp_path))


def test_c8_one_row_cannot_confirm_a_duplicated_selection(monkeypatch, tmp_path):
    from bts.picks import DailyPick, Pick, save_pick
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    units_snapshot(tmp_path, [{"id": 1, "feedId": 1, "roundId": 7}])
    picks = tmp_path / "picks"
    picks.mkdir()
    mk = lambda: Pick(batter_name="B", batter_id=1, team="NYY", lineup_position=1, pitcher_name="P", pitcher_id=2,  # noqa: E731
                      p_game_hit=0.8, flags=[], projected_lineup=False, game_pk=1, game_time=PITCH)
    save_pick(DailyPick(date=DATE, run_time="x", pick=mk(), double_down=mk(), runner_up=None, notification_sent=True,
                        notification_channel="bluesky_dm", notification_id="dm_x"), picks)
    H._run(picks, IN_WINDOW)
    q = only(tmp_path)["qualification"]
    assert q["all_confirmed"] is False
    assert {x["state"] for x in q["slots"]} == {"unsupported_duplicate_selection"}
