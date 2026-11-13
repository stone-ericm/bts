"""The end-of-day outcome check (roadmap thread C, 2026-11-13): every day must end in a KNOWN state.

2026's worst days were days nobody was told about: 7/16 (a preview never delivered, no decision record), 8/13 (a
silent pass), 8/30 (a late pick). The check reads the day's decision record and the entry checker's receipts and
names the end state; anything but a clean state is a problem to DM. It never writes pick, decision or streak state.
"""
from __future__ import annotations

import pytest

from bts.day_outcome import classify_day

DATE = "2027-04-02"
SHA = "d" * 64
PICK = {"batter_id": 1, "batter_name": "Kwan", "team": "CLE", "game_pk": 900, "p_game_hit": 0.78}


def _decision(action="single", source="mdp", *, scoreable=True, delivery_status="delivered", double=None):
    return {"schema_version": "bts_daily_decision_v3", "date": DATE, "action": action, "source": source,
            "primary": PICK if action != "skip" else None, "double_down": double,
            "delivery_status": delivery_status, "scoreable": scoreable}


def _receipt(outcome="observed", *, confirmed=True, before_cutoff=True, reason="match", sha=SHA,
             username="stonehengee", date=DATE, degraded=(), completed="2027-04-02T17:31:00-04:00"):
    rec = {"schema": "bts_pick_entry_receipt_v1", "outcome": outcome, "et_date": date, "season": int(date[:4]),
           "selection": {"commit": {"decision_sha256": sha}}, "degraded": list(degraded)}
    if outcome == "observed":
        rec |= {"account": {"username": username},
                "observation": {"before_cutoff": before_cutoff, "response_completed_at": completed},
                "verifier": {"reason": reason}, "qualification": {"all_confirmed": confirmed}}
    return rec


def _classify(**kw):
    base = {"date": DATE, "games": 12, "decision": _decision(), "decision_sha256": SHA, "receipts": [_receipt()],
            "intent": "enter", "expected_username": "stonehengee"}
    base.update(kw)
    return classify_day(**base)


# ---------------------------------------------------------------- clean end states

def test_a_delivered_pick_with_a_confirmed_entry_is_ok():
    out = _classify()
    assert out.ok and out.state == "entered"


def test_a_deliberate_policy_skip_is_ok():
    out = _classify(decision=_decision("skip", "mdp", scoreable=False, delivery_status=None), receipts=[])
    assert out.ok and out.state == "deliberate_skip"


def test_a_day_with_no_games_and_no_decision_is_ok():
    out = _classify(games=0, decision=None, decision_sha256=None, receipts=[])
    assert out.ok and out.state == "no_games"


def test_research_mode_expects_a_private_commit_and_no_entry():
    out = _classify(intent="research", decision=_decision(delivery_status="private_locked"), receipts=[])
    assert out.ok and out.state == "research_commit"


def test_one_confirming_receipt_is_enough_among_unverifiable_ones():
    receipts = [_receipt("fetch_failed"), _receipt("outside_window"), _receipt(), _receipt("already_confirmed")]
    assert _classify(receipts=receipts).state == "entered"


# ---------------------------------------------------------------- the unknown day (7/16, 8/13)

def test_games_were_played_and_there_is_no_decision_record():
    out = _classify(decision=None, decision_sha256=None, receipts=[])
    assert not out.ok and out.state == "no_decision"
    assert "12 games" in out.detail


def test_a_commit_that_was_never_delivered_is_a_problem():
    out = _classify(decision=_decision(scoreable=False, delivery_status="failed"), receipts=[])
    assert not out.ok and out.state == "commit_not_scoreable"


def test_a_skip_the_policy_did_not_decide_is_a_problem():
    out = _classify(decision=_decision("skip", "crash_guard", scoreable=False, delivery_status=None), receipts=[])
    assert not out.ok and out.state == "skip_not_deliberate"
    assert "crash_guard" in out.detail


def test_an_unknown_schedule_is_unverifiable_never_a_quiet_no_games():
    out = _classify(games=None, decision=None, decision_sha256=None, receipts=[])
    assert not out.ok and out.state == "schedule_unverifiable"


def test_a_commit_on_a_day_with_no_games_is_a_contradiction():
    out = _classify(games=0)
    assert not out.ok and out.state == "contradiction"


# ---------------------------------------------------------------- entry

def test_an_observed_absence_is_not_entered():
    out = _classify(receipts=[_receipt(confirmed=False, reason="no_pick")])
    assert not out.ok and out.state == "not_entered"


def test_only_failed_or_unattempted_receipts_are_unverifiable_never_not_entered():
    out = _classify(receipts=[_receipt("fetch_failed"), _receipt("outside_window"), _receipt("incomplete")])
    assert not out.ok and out.state == "entry_unverifiable"
    assert _classify(receipts=[]).state == "entry_unverifiable"


def test_already_confirmed_alone_is_not_a_confirmation():
    """It makes no observation of its own; without the confirming receipt it proves nothing."""
    assert _classify(receipts=[_receipt("already_confirmed")]).state == "entry_unverifiable"
    dressed = _receipt() | {"outcome": "already_confirmed"}      # every observed field present, but no observation
    assert _classify(receipts=[dressed]).state == "entry_unverifiable"


def test_a_confirmation_seen_only_after_the_cutoff_is_late():
    out = _classify(receipts=[_receipt(before_cutoff=False)])
    assert not out.ok and out.state == "entered_late"
    assert "17:31" in out.detail


@pytest.mark.parametrize("change", [
    {"sha": "e" * 64},                       # a receipt for another selection (the pick changed afterwards)
    {"username": "someone_else"},            # another account
    {"date": "2027-04-01"},                  # another contest date
    {"degraded": ["hook_x"]},                # a receipt with failed hooks
])
def test_a_confirmation_that_does_not_bind_to_this_day_does_not_count(change):
    assert _classify(receipts=[_receipt(**change)]).state == "entry_unverifiable"


def test_the_legacy_match_without_game_qualification_does_not_count():
    assert _classify(receipts=[_receipt(confirmed=False, reason="match")]).state == "entry_unverifiable"


def test_a_delivered_pick_in_research_mode_is_a_problem():
    out = _classify(intent="research", receipts=[])
    assert not out.ok and out.state == "delivered_under_research"


def test_a_missing_intent_is_a_problem_not_a_default():
    out = _classify(intent=None)
    assert not out.ok and out.state == "intent_unknown"


def test_the_message_names_the_date_the_state_and_the_pick():
    out = _classify(receipts=[_receipt(confirmed=False, reason="no_pick")])
    text = out.message()
    assert DATE in text and "Kwan" in text and "not entered" in text.lower()


# ---------------------------------------------------------------- the command, on real files

def _box(tmp_path, *, intent="enter", delivery="dm"):
    picks = tmp_path / "data" / "picks"
    picks.mkdir(parents=True)
    cfg = tmp_path / "orch.toml"
    cfg.write_text(f'[orchestrator]\npicks_dir = "{picks}"\n[scheduler]\nentry_intent = "{intent}"\n'
                   f'pick_delivery = "{delivery}"\n')
    return picks, cfg


def _write_decision(picks, **kw):
    from bts.daily_decision import write_decision
    base = dict(action="single", source="mdp", primary=PICK, delivery_status="delivered", scoreable=True)
    base.update(kw)
    assert write_decision(DATE, picks, **base) is not None


def _publish_receipt(picks, rec, name="20270402T173100000000-a.json"):
    import json
    from bts import entry_receipt, receipt_io
    receipt_io.publish(entry_receipt.receipts_root(picks) / DATE / name, json.dumps(rec).encode())


def _run(monkeypatch, tmp_path, picks, cfg, *, games=12, dms=None, extra=()):
    import bts.scheduler as sch
    from click.testing import CliRunner
    from bts.cli import cli
    if isinstance(games, Exception):
        def fetch(date):
            raise games
    else:
        def fetch(date):
            return [{}] * games
    monkeypatch.setattr(sch, "fetch_schedule", fetch)
    if dms is not None:
        monkeypatch.setattr("bts.dm.send_dm", lambda who, text: dms.append((who, text)) or "id")
    return CliRunner().invoke(cli, ["day-outcome", "--date", DATE, "--config", str(cfg), "--picks-dir", str(picks),
                                    "--expected-username", "stonehengee", *extra])


def _bound_receipt(picks, **kw):
    """A receipt bound to the decision file actually on disk."""
    from bts import entry_receipt
    from bts.daily_decision import load_decision_bytes
    return _receipt(sha=entry_receipt._sha(load_decision_bytes(DATE, picks)[1]), **kw)


def test_command_reports_entered_from_a_sealed_receipt_bound_to_the_decision_on_disk(monkeypatch, tmp_path):
    import json
    picks, cfg = _box(tmp_path)
    _write_decision(picks)
    _publish_receipt(picks, _bound_receipt(picks))
    dms = []
    res = _run(monkeypatch, tmp_path, picks, cfg, dms=dms, extra=["--dm-recipient", "me.bsky.social"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["state"] == "entered" and dms == []


def test_command_dms_and_exits_1_when_games_were_played_and_nothing_was_decided(monkeypatch, tmp_path):
    picks, cfg = _box(tmp_path)
    dms = []
    res = _run(monkeypatch, tmp_path, picks, cfg, dms=dms, extra=["--dm-recipient", "me.bsky.social"])
    assert res.exit_code == 1
    assert len(dms) == 1 and dms[0][0] == "me.bsky.social" and "NO DECISION" in dms[0][1] and DATE in dms[0][1]


def test_command_ignores_an_unsealed_receipt(monkeypatch, tmp_path):
    """A receipt file without its seal is unavailable evidence (receipt consumer rule), never a confirmation."""
    import json
    from bts import entry_receipt
    picks, cfg = _box(tmp_path)
    _write_decision(picks)
    d = entry_receipt.receipts_root(picks) / DATE
    d.mkdir(parents=True)
    (d / "20270402T173100000000-a.json").write_text(json.dumps(_bound_receipt(picks)))
    res = _run(monkeypatch, tmp_path, picks, cfg)
    assert res.exit_code == 1 and json.loads(res.output)["state"] == "entry_unverifiable"


def test_command_does_not_count_a_receipt_for_an_earlier_version_of_the_decision(monkeypatch, tmp_path):
    import json
    picks, cfg = _box(tmp_path)
    _write_decision(picks)
    stale = _bound_receipt(picks)
    _write_decision(picks, primary={**PICK, "batter_name": "Ramirez", "batter_id": 2})   # the pick changed afterwards
    _publish_receipt(picks, stale)
    res = _run(monkeypatch, tmp_path, picks, cfg)
    assert res.exit_code == 1 and json.loads(res.output)["state"] == "entry_unverifiable"


def test_command_reports_a_schedule_failure_as_unverifiable(monkeypatch, tmp_path):
    import json
    picks, cfg = _box(tmp_path)
    res = _run(monkeypatch, tmp_path, picks, cfg, games=OSError("network down"))
    assert res.exit_code == 1 and json.loads(res.output)["state"] == "schedule_unverifiable"


def test_command_in_research_mode_accepts_a_private_commit(monkeypatch, tmp_path):
    import json
    picks, cfg = _box(tmp_path, intent="research", delivery="private")
    _write_decision(picks, delivery_status="private_locked")
    res = _run(monkeypatch, tmp_path, picks, cfg)
    assert res.exit_code == 0 and json.loads(res.output)["state"] == "research_commit"


def test_command_treats_an_intent_that_disagrees_with_delivery_as_unknown(monkeypatch, tmp_path):
    import json
    picks, cfg = _box(tmp_path, intent="enter", delivery="private")
    _write_decision(picks)
    res = _run(monkeypatch, tmp_path, picks, cfg)
    assert res.exit_code == 1 and json.loads(res.output)["state"] == "intent_unknown"


def test_command_exits_2_and_dms_when_the_config_cannot_be_read(monkeypatch, tmp_path):
    picks, cfg = _box(tmp_path)
    cfg.write_text("this is [not toml")
    dms = []
    res = _run(monkeypatch, tmp_path, picks, cfg, dms=dms, extra=["--dm-recipient", "me.bsky.social"])
    assert res.exit_code == 2 and len(dms) == 1 and "COULD NOT RUN" in dms[0][1]


def test_command_writes_nothing(monkeypatch, tmp_path):
    picks, cfg = _box(tmp_path)
    _write_decision(picks)
    _publish_receipt(picks, _bound_receipt(picks))
    before = sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in tmp_path.rglob("*"))
    _run(monkeypatch, tmp_path, picks, cfg)
    assert sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in tmp_path.rglob("*")) == before
