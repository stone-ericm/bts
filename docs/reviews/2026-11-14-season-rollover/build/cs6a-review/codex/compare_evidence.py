"""Check the recorded base/branch behavior and name base-green controls."""

import local_guard
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bts import entry_intent, scheduler


def load(stem, variant):
    return json.loads((HERE / f"{stem}-{variant}.json").read_text())


def chosen_files(data, *, posted):
    day = "2027-07-15"
    name = f"{day}.json"
    evolution = f"lineup_evolution_{day}.jsonl"
    if data:
        assert set(data) == {name, evolution}, data
    chosen = json.loads(bytes.fromhex(data[name]))
    assert chosen["bluesky_posted"] is posted
    assert chosen["bluesky_uri"] == ("at://capture/pick" if posted else None)
    rows = bytes.fromhex(data[evolution]).splitlines()
    assert rows and json.loads(rows[0])["primary"]["batter_name"] == chosen["pick"]["batter_name"]


def pick_comparison():
    base = load("pick", "base")
    head = load("pick", "head")
    assert set(base) == set(head)
    accepted = refusals = 0
    for case, actual in head.items():
        old = base[case]
        path, state = case.split("/")
        if state in {"dated", "covered"} or path in {
            "empty", "pipeline-error", "unexpected-pipeline-error", "dry-run"
        }:
            event_trace = [["read", False] if r[0] == "read" else r for r in actual["events"]]
            assert all(row[1] is True for row in actual["events"] if row[0] == "read")
            assert old == actual | {"events": event_trace}, (case, old, actual)
            accepted += 1
            continue
        assert actual["exit"] == 1 and actual["exception"] == "SystemExit", (case, actual)
        assert actual["stderr"].startswith("bts run: no pick: Live decision refused: "), (case, actual)
        assert old["stdout"].startswith(actual["stdout"]), (case, actual)
        event_names = [e[0] for e in actual["events"]]
        if state == "mid-run-revoke":
            old_names = [e[0] for e in old["events"]]
            assert event_names == old_names[:len(event_names)]
            assert event_names[-1] == "read" and event_names.count("read") == 2
            if path == "locked":
                assert actual["stdout"].splitlines()[-1] == "Pick locked: Jacob Wilson (game started)"
                assert not actual["files"]
            else:
                chosen_files(actual["files"], posted=False)
                chosen_files(old["files"], posted=True)
        else:
            assert not actual["files"]
            assert event_names == ["prediction", "read"]
        refusals += 1
    return dict(parity=accepted, refused=refusals)


def startup_comparison():
    base, head = (load("startup", key) for key in ("base", "head"))
    new = head["refusal"]
    assert new["worker"] == new["date"] == 0 and base["refusal"]["worker"] == base["refusal"]["date"] == 1
    assert new["loader"] == len(new["opens"]) == 1 and new["unchanged"]
    assert new["opens"][0][0] == "<root>/resource-access/config.toml"
    assert new["events"] == []
    assert base["schedule_identity"] == head["schedule_identity"] and head["schedule_identity"]["identity"]
    for kind in ("dry_game", "dry_off"):
        early, separate, rest = head[kind]["result"]["stderr"].partition("\n")
        assert early.startswith("bts schedule: WARNING (a real run would refuse to start): ")
        assert separate == "\n"
        expected = head[kind] | {"result": head[kind]["result"] | {"stderr": rest}}
        assert base[kind] == expected, (base[kind], expected)
    assert base["errors"] == head["errors"], (base["errors"], head["errors"])
    assert head["snapshot"]["identity"] and head["snapshot"]["loads"] == 1
    assert [r[0] for r in head["snapshot"]["opens"]] == ["r", "w"]
    assert [r[0] for r in base["snapshot"]["opens"]] == ["w", "r"]
    assert head["snapshot"]["original_pick"] and not head["snapshot"]["moved_pick"]
    assert base["snapshot"]["moved_pick"] and not base["snapshot"]["original_pick"]
    for key, prior in base["representatives"].items():
        current = head["representatives"][key]
        if key in ("dm-optional", "orch-private-research"):
            assert current["result"]["exit"] == 2 and not current["events"] and not current["pick"]
            assert prior["pick"] and any(e[0] in ("post", "dm") for e in prior["events"])
        else:
            assert prior == current, (key, prior, current)
    return head["exhaustive"]


def diagnostics_comparison():
    base, head = (load("diagnostics", key) for key in ("base", "head"))
    assert base["day_outcome"] == head["day_outcome"]
    for case, actual in head["cron"].items():
        old = base["cron"][case]
        if case not in ("show/neither", "install/neither"):
            assert actual == old
            continue
        first, last = actual["stderr"].splitlines()
        assert first.startswith("entry-intent: scheduler delivery 'dm' and scheduler.entry_intent = 'enter'")
        assert last.startswith(f"ERROR: refusing to {case.split('/')[0]}: ")
        assert actual["exit"] == 1 and actual["unchanged"] and actual["calls"] == [] and actual["stdout"] == ""
        assert old["exit"] == 0 and old["calls"]
    assert base["example"] == dict(required=False, expected=False, intent=None, other_values_identical=True)
    assert head["example"] == dict(required=True, expected=True, intent="enter", other_values_identical=True)
    day_states = Counter(value["direct"]["state"] for value in head["day_outcome"].values())
    return dict(cron=len(head["cron"]), day_outcome=len(head["day_outcome"]), classifications=dict(day_states))


def matrix_comparison():
    differences = Counter()
    with (HERE / "matrix-base.jsonl").open() as base, (HERE / "matrix-head.jsonl").open() as head:
        for left, right in zip(base, head, strict=True):
            case, base_outcomes = json.loads(left)
            same_case, head_outcomes = json.loads(right)
            assert same_case == case
            for name, previous, current in zip(
                ("schedule", "dry-run", "orchestrate", "entry-intent"), base_outcomes, head_outcomes, strict=True
            ):
                differences[name] += previous != current
    assert differences == dict(schedule=20657, **{"dry-run": 20657, "orchestrate": 31283, "entry-intent": 20631})
    return dict(differences)


def predicate_purity():
    blocker = Mock(side_effect=AssertionError("gate attempted resource or clock access"))
    with ExitStack() as stack:
        for target in ("builtins.open", "pathlib.Path.stat", "pathlib.Path.open", "time.time", "time.monotonic",
                       "datetime.datetime", "datetime.date", "bts.scheduler._now_et"):
            stack.enter_context(patch(target, blocker))
        assert entry_intent.live_gate_problem({"scheduler": {"pick_delivery": "private"}}) is None
        assert entry_intent.live_gate_problem({"scheduler": {"pick_delivery": None}}).startswith(
            "scheduler delivery 'public' requires")
        assert entry_intent.live_gate_problem({"scheduler": {"pick_delivery": "private"}},
                                             delivery_override="public").startswith("delivery 'public' requires")
        assert entry_intent.live_gate_problem({"health_checks": None}) == "[health_checks] is not a table"
    assert not blocker.called


def base_green_controls():
    lines = (HERE / "pytest-base.log").read_text().splitlines()
    greens = [line.removeprefix("PASSED ") for line in lines if line.startswith("PASSED ") and (
        "test_live_gate_cs6a.py::" in line or "test_removing_cron_still_works" in line or
        "test_live_cron_with_required_state" in line
    )]
    assert len(greens) == 12
    (HERE / "base-green-controls.txt").write_text("\n".join(greens) + "\n")
    return len(greens)


if __name__ == "__main__":
    print("CLI matrix changes", matrix_comparison())
    print("bts run", pick_comparison())
    print("Startup", startup_comparison())
    print("Diagnostics", diagnostics_comparison())
    predicate_purity()
    print("Named new base-green controls", base_green_controls())
    assert not local_guard.network
