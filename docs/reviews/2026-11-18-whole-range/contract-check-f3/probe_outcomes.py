import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from click.testing import CliRunner
from bts import cli, daily_decision, day_outcome as trial, dm, scheduler
from tests.test_day_outcome import DATE, SHA, _decision, _receipt

spec = importlib.util.spec_from_file_location("base_day_outcome", Path.cwd() / "src/bts/day_outcome.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

SKIP = _decision("skip", scoreable=False) | {
    "streak": 30, "saver_available": False, "state_source": "contest",
}
REQUEST = {"effective_date": DATE, "active_streak": 1, "saver_state": "not_earned"}
SWITCH = {"status": "revoked", "switched_at": DATE + "T14:00:00Z",
          "revoked_at": "2027-04-03T03:50:00Z"}


def raw(out):
    return json.dumps(out.as_dict()), out.message()


def classify_pair(**changes):
    values = dict(date=DATE, games=1, void_games=0, decision=_decision(),
                  decision_sha256=SHA, receipts=[_receipt()], intent="enter",
                  expected_username="stonehengee") | changes
    return base.classify_day(**values), trial.classify_day(**values, request=REQUEST, switch=SWITCH)


def compare_states():
    batter = _receipt(confirmed=False)
    batter["qualification"]["slots"] = [{"state": "unit_unverified"}]
    cases = {
        "entered": {},
        "entered_game_unverified": {"receipts": [batter]},
        "entered_late": {"receipts": [_receipt(before_cutoff=False)]},
        "not_entered": {"receipts": [_receipt(confirmed=False, reason="no_pick")]},
        "entry_unverifiable": {"receipts": []},
        "no_decision": {"decision": None},
        "no_games": {"games": 0, "decision": SKIP},
        "no_games_all_void": {"void_games": 1, "decision": SKIP},
        "commit_not_scoreable": {"decision": _decision(scoreable=False)},
        "research_commit": {"intent": "research", "decision": _decision(delivery_status="private_locked")},
        "delivered_under_research": {"intent": "research"},
        "contradiction": {"games": 0},
        "intent_unknown": {"intent": None, "decision": SKIP},
        "schedule_unverifiable": {"games": None, "decision": SKIP},
        "skip_not_deliberate": {"decision": SKIP | {"source": "other"}},
    }
    for label, values in cases.items():
        before, after = classify_pair(**values)
        assert raw(before) == raw(after) and before.state == label.replace("_all_void", "")
        print(label, "identical JSON and message")
    before, after = classify_pair(intent="research", decision=SKIP)
    assert (before.state, after.state) == ("deliberate_skip", "skip_state_changed")
    print("research MDP skip", before.state, after.state)

    fake = SKIP | {"state_source": "automatic"}
    _, answer = classify_pair(decision=fake)
    assert "revoked at 23:50:00 ET" in answer.detail and "enabled at 10:00:00 ET" in answer.detail
    print("Hypothetical action record carrying automatic:", answer.message())
    before, proxy = classify_pair(decision=SKIP)
    proxy = trial.classify_day(date=DATE, games=1, decision=fake, decision_sha256=SHA,
                               receipts=[], intent="enter", expected_username=None,
                               switch={"status": "automatic", "switched_at": DATE + "T14:00:00Z",
                                       "revoked_at": None})
    assert before.ok and not proxy.ok
    print("Proxy for contest == automatic hypothesis, with a same-morning enable before final skip:", proxy.message())


def evidence_and_exit(root):
    picks = root / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True)
    request_path = account / "contest_manual_requests.jsonl"
    switch_path = account / "season_contract.json"
    assert daily_decision.write_decision(
        DATE, picks, action="skip", source="mdp", streak=30, saver_available=False,
        state_source="contest", state_status="fresh", contest_source_date=DATE,
        delivery_status="not_applicable", scoreable=False,
    )
    config = {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}
    cfg_file = root / "orch.toml"
    cfg_file.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')

    def run():
        return trial.gather(DATE, picks, config, expected_username=None, fetch_schedule=lambda _: [{}])

    case_writers = {
        "directory": lambda p: p.mkdir(),
        "FIFO": lambda p: os.mkfifo(p),
        "undecodable": lambda p: p.write_bytes(b"\xff"),
        "JSON list": lambda p: p.write_text("[]\n"),
        "other date": lambda p: p.write_text(json.dumps(REQUEST | {"effective_date": "2027-04-03"})),
        "malformed": lambda p: p.write_text("{bad JSON"),
        "long integer": lambda p: p.write_text('{"effective_date":"' + DATE + '","active_streak":' + '1' * 5000 + '}'),
        "deep nesting": lambda p: p.write_text("[" * 2000 + "0" + "]" * 2000),
    }
    assert run().state == "deliberate_skip"
    for label, writer in case_writers.items():
        for path in (request_path, switch_path):
            writer(path)
            assert run().state == "deliberate_skip", (label, path)
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        print("Both evidence files, ignored:", label)
    with patch.object(trial, "read_regular_file_text", side_effect=PermissionError("unreadable")):
        assert run().state == "deliberate_skip"
    print("Both evidence files, simulated EACCES")

    request_path.write_text(json.dumps(REQUEST) + "\n{torn\n")
    assert run().state == "skip_state_changed"
    request_path.write_text(json.dumps(REQUEST) + "\n" + json.dumps({"effective_date": DATE}) + "\n")
    assert run().state == "deliberate_skip"
    print("Valid request survives broken JSON, then gets masked by a last matching object with no values")

    request_path.write_text(json.dumps(REQUEST | {"active_streak": {"bad": "shape"}}))
    assert run().state == "deliberate_skip"
    request_path.write_text(json.dumps(REQUEST))
    switch_path.write_text("[corrupt]")
    assert run().state == "skip_state_changed"
    print("Valid request survives malformed switch")

    args = ["day-outcome", "--date", DATE, "--picks-dir", str(picks), "--config", str(cfg_file),
            "--dm-recipient", "offline.invalid"]
    with patch.object(scheduler, "fetch_schedule", return_value=[{}]), \
         patch.object(dm, "send_dm", side_effect=OSError("send failure")):
        result = CliRunner().invoke(cli.cli, args)
        assert result.exit_code == 1 and "DM failed: OSError" in result.output
    cfg_file.write_text("[invalid TOML!")
    with patch.object(dm, "send_dm", return_value=None) as mocked:
        result = CliRunner().invoke(cli.cli, args)
    assert result.exit_code == 2 and "THE CHECK COULD NOT RUN" in mocked.call_args.args[1]
    print("Failed send stays 1; invalid config stays 2:", mocked.call_args.args[1])


compare_states()
with tempfile.TemporaryDirectory() as temp:
    evidence_and_exit(Path(temp))
