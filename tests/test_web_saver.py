"""Dashboard Streak Saver controls — pure helpers (web tests use helpers, not the handler)."""
import json
from datetime import datetime, timezone

from bts.web import saver_dashboard_context, saver_transition_response
from bts.saver_state import transition_saver_state

NOW = datetime(2026, 6, 18, 16, 0, tzinfo=timezone.utc)


def _setup(tmp_path, state, best=10, active=10):
    d = tmp_path / "account_state"; d.mkdir(parents=True, exist_ok=True)
    (d / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": active,
        "best_streak": best, "source": "mlb_bts_profile", "source_date": "2026-06-18"}))
    (d / "saver_state.json").unlink(missing_ok=True)        # reset the flag each setup
    if state != "uninitialized":
        transition_saver_state(tmp_path, expected_prior="uninitialized", new_state=state,
                               season=2026, source="t")


def test_button_visibility(tmp_path):
    _setup(tmp_path, "active")
    assert saver_dashboard_context(tmp_path, now=NOW).button == "mark_used"
    _setup(tmp_path, "used")
    assert saver_dashboard_context(tmp_path, now=NOW).button == "undo"
    _setup(tmp_path, "not_earned")
    assert saver_dashboard_context(tmp_path, now=NOW).button is None
    _setup(tmp_path, "uninitialized")
    assert saver_dashboard_context(tmp_path, now=NOW).button is None


def test_warning_when_active_past_15(tmp_path):
    _setup(tmp_path, "active", best=16, active=16)
    assert saver_dashboard_context(tmp_path, now=NOW).warning is True
    _setup(tmp_path, "active", best=12, active=12)
    assert saver_dashboard_context(tmp_path, now=NOW).warning is False


def test_nudge_on_likely_save_in_ledger(tmp_path):
    _setup(tmp_path, "active")
    # two identical snapshots so the held not_hit at 12 is stable
    snap = {"recorded_at": "2026-06-18T17:00:00Z", "active_streak": 12, "predictions": [
        {"roundId": 1, "result": "hit", "streak": 12, "roundPredictions": [{"playerId": 1}]},
        {"roundId": 2, "result": "not_hit", "streak": 12, "roundPredictions": [{"playerId": 2}]},  # held at 12 (save)
    ]}
    (tmp_path / "account_state" / "contest_ledger.jsonl").write_text(
        "\n".join(json.dumps(snap) for _ in range(2)))
    assert saver_dashboard_context(tmp_path, now=NOW).nudge is True


def test_transition_rejects_wrong_expected_prior(tmp_path):
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(tmp_path, expected_prior="not_earned",
                                        new_state="used", same_origin=True, now=NOW)
    assert code == 409


def test_transition_rejects_cross_origin(tmp_path):
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(tmp_path, expected_prior="active",
                                        new_state="used", same_origin=False, now=NOW)
    assert code == 403


def test_transition_marks_used(tmp_path):
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(tmp_path, expected_prior="active",
                                        new_state="used", same_origin=True, now=NOW)
    assert code == 200
    assert saver_dashboard_context(tmp_path, now=NOW).state == "used"


def test_dashboard_rejects_non_ui_transitions(tmp_path):
    _setup(tmp_path, "uninitialized")
    # init transitions are NOT exposed via the dashboard POST (only mark-used / undo)
    code, _ = saver_transition_response(tmp_path, expected_prior="uninitialized",
                                        new_state="active", same_origin=True, now=NOW)
    assert code == 409


def test_dashboard_rejects_bogus_new_state(tmp_path):
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(tmp_path, expected_prior="active",
                                        new_state="bogus", same_origin=True, now=NOW)
    assert code == 409   # rejected, never raises


def test_same_origin_helper():
    from bts.web import _same_origin
    assert _same_origin({}, "host") is True                            # absent -> allow (non-browser)
    assert _same_origin({"Origin": "http://host"}, "host") is True     # match
    assert _same_origin({"Origin": "http://evil"}, "host") is False    # mismatch
    assert _same_origin({"Origin": "http://[::1"}, "host") is False    # malformed -> reject (no raise)


# --- do_POST /saver/transition socket wiring (open item from 2026-06-18: the
# --- handler had only ad-hoc verification; this exercises the real HTTP path
# --- against an isolated PICKS_DIR).

def _serve(tmp_path, monkeypatch):
    import threading
    from http.server import HTTPServer
    import bts.web as web
    monkeypatch.setattr(web, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(web.Handler, "log_message", lambda *a, **k: None)
    srv = HTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(port, body, origin=None, path="/saver/transition"):
    from http.client import HTTPConnection
    from urllib.parse import urlencode
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if origin is not None:
        headers["Origin"] = origin
    conn.request("POST", path, urlencode(body), headers)
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp


def _saver_state(tmp_path):
    return saver_dashboard_context(tmp_path, now=NOW).state


def test_do_post_saver_transition_socket_wiring(tmp_path, monkeypatch):
    _setup(tmp_path, "active")
    srv = _serve(tmp_path, monkeypatch)
    try:
        port = srv.server_address[1]

        r = _post(port, {"expected_prior": "active", "new_state": "used"})
        assert r.status == 303 and r.getheader("Location") == "/"
        assert _saver_state(tmp_path) == "used"

        r = _post(port, {"expected_prior": "used", "new_state": "active"})
        assert r.status == 303
        assert _saver_state(tmp_path) == "active"

        r = _post(port, {"expected_prior": "active", "new_state": "used"},
                  origin="http://evil.example")
        assert r.status == 403
        assert _saver_state(tmp_path) == "active"     # unchanged

        r = _post(port, {"expected_prior": "used", "new_state": "active"})
        assert r.status == 409                        # guard mismatch: state is active
        assert _saver_state(tmp_path) == "active"

        r = _post(port, {"expected_prior": "active", "new_state": "used"}, path="/nope")
        assert r.status == 404
        assert _saver_state(tmp_path) == "active"
    finally:
        srv.shutdown()
        srv.server_close()


def test_dashboard_transition_records_peer_ip(tmp_path):
    # F7: the web handler threads the requesting peer into the audit trail.
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(
        tmp_path, expected_prior="active", new_state="used",
        same_origin=True, now=NOW, peer_ip="100.64.7.7",
    )
    assert code == 200
    log = tmp_path / "account_state" / "saver_transitions.jsonl"
    last = json.loads(log.read_text().splitlines()[-1])
    assert last["peer"] == "100.64.7.7"
    assert last["source"] == "dashboard"
    assert last["outcome"] == "written"
