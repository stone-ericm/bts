"""Take comparable CLI and shared-reader snapshots on the base and F4."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import bts
from bts.util import read_regular_file_text
from report_harness import Chat, _box, _write_decision, observed, report, snapshot
from tests.test_day_outcome import _bound_receipt, _publish_receipt
from tests.test_wr_f3_skip_state_review import evidence

STATES = {
    "entered": (1, {}, "success", "enter"),
    "entered_game_unverified": (1, {}, "batter-only", "enter"),
    "deliberate_skip": (1, {"action": "skip", "scoreable": False}, None, "enter"),
    "no_games": (0, None, None, "enter"),
    "research_commit": (1, {"delivery_status": "private_locked"}, None, "research"),
    "delivered_under_research": (1, {}, None, "research"),
    "intent_unknown": (1, None, None, "invalid"),
    "schedule_unverifiable": (OSError("offline"), None, None, "enter"),
    "no_decision": (2, None, None, "enter"),
    "skip_not_deliberate": (1, {"action": "skip", "source": "boom", "scoreable": False}, None, "enter"),
    "commit_not_scoreable": (1, {"scoreable": False, "delivery_status": "failed"}, None, "enter"),
    "contradiction": (0, {}, None, "enter"),
    "entered_late": (1, {}, "late", "enter"),
    "not_entered": (1, {}, "absence", "enter"),
    "entry_unverifiable": (1, {}, None, "enter"),
    "skip_state_review": (1, {"action": "skip", "scoreable": False, "streak": 19, "saver_available": True},
                          None, "enter"),
    "all_void": ([{"status": {"detailedState": "Postponed"}}], None, None, "enter"),
}
CONTENTS = {
    "absent": None, "empty-object": b"{}", "zero": b'{"streak":0}',
    "legacy": b'{"streak":16,"applied_dates":["2027-04-02",null]}',
    "prior": b'{"season":2026,"streak":4,"applied_dates":"ignored"}',
    "unknown": b'{"season":{},"streak":500,"saver_available":[],"applied_dates":12}',
    "at-limit": b" " * (1024 * 1024 - 2) + b"{}",
}


def collect_reports(root, candidate):
    data = {}
    for state, (games, decision, receipt, intent) in STATES.items():
        fixture = root / state
        picks, config = _box(fixture, intent=intent, delivery="private" if intent == "research" else "dm")
        if decision is not None:
            _write_decision(picks, **decision)
        if receipt == "batter-only":
            bound = _bound_receipt(picks, confirmed=False)
            bound["qualification"]["slots"] = [{"state": "unit_unverified"}]
            _publish_receipt(picks, bound)
        elif receipt == "late":
            observed(picks, before_cutoff=False)
        elif receipt == "absence":
            observed(picks, confirmed=False, reason="no_pick")
        elif receipt == "success":
            observed(picks)
        if state == "skip_state_review":
            account = picks / "account_state"
            account.mkdir()
            for name, kind in [("contest_streak.manual.json", "dated"), ("season_contract.json", "switch")]:
                (account / name).write_text(json.dumps(evidence(kind)))
        for kind, content in CONTENTS.items():
            if content is not None:
                (picks / "streak.json").write_bytes(content)
            before = snapshot(fixture)
            result, chat = report(picks, config, games)
            payload = json.loads(result.stdout)
            assert payload["state"] == ("no_games" if state == "all_void" else state), payload
            rendered = result.stdout
            if candidate:
                assert payload["streak_problem"] is None
                suffix = ', "streak_problem": null}\n'
                assert rendered.endswith(suffix), rendered
                rendered = rendered.removesuffix(suffix) + "}\n"
            assert snapshot(fixture) == before
            data[f"{state}/{kind}"] = [result.exit_code, rendered, result.stderr, chat.attempted, chat.accepted]

    for kind in ("timeout", "no-id", "missing-config"):
        picks, config = _box(root / kind)
        if kind == "missing-config":
            config.unlink()
            (picks / "streak.json").write_bytes(b"[]")
        result, chat = report(picks, config, 1, chat=Chat(failure=kind))
        rendered = result.stdout
        if candidate and rendered:
            assert rendered.endswith(', "streak_problem": null}\n')
            rendered = rendered.removesuffix(', "streak_problem": null}\n') + "}\n"
        data[f"outside-grid/{kind}"] = [result.exit_code, rendered, result.stderr.replace(str(root), "<root>"),
                                       chat.attempted, chat.accepted]
    return data


def default_reader(root):
    root.mkdir()
    inputs = {
        "newlines": b"a\r\nb\rc\n", "utf8": "Curaçao 東京".encode(), "bad-utf8": b"\xff",
        "over-limit": b"x" * (1024 * 1024 + 1), "mode-zero": b"{}",
    }
    for name, value in inputs.items():
        (root / name).write_bytes(value)
    (root / "mode-zero").chmod(0)
    os.mkfifo(root / "fifo")
    (root / "directory").mkdir()
    for name, target in [("symlink", "newlines"), ("dangling", "absent"), ("loop", "loop")]:
        (root / name).symlink_to(target)

    outputs = {}
    for name in [*inputs, "fifo", "directory", "symlink", "dangling", "loop", "absent"]:
        try:
            value = read_regular_file_text(root / name)
        except (OSError, UnicodeDecodeError) as exc:
            outputs[name] = [type(exc).__name__, str(exc).replace(str(root), "<root>")]
        else:
            if name == "over-limit":
                assert value == "x" * (1024 * 1024 + 1)
                value = str(len(value))
            outputs[name] = ["text", value]
    return outputs


mode, *paths = sys.argv[1:]
if mode == "compare":
    base, candidate = (json.loads(Path(path).read_text()) for path in paths)
    for family in base:
        assert base[family].keys() == candidate[family].keys()
        for case, value in base[family].items():
            assert value == candidate[family][case], (family, case, value, candidate[family][case])
        print(f"{family}: {len(base[family])} base/F4 comparisons agree")
else:
    assert mode in {"base", "candidate"}
    candidate = mode == "candidate"
    path = Path(bts.__file__).resolve()
    print(f"{mode} bts.__file__ = {path}")
    assert (str(path).startswith("/private/tmp/wr-f4-r1.")) == (not candidate), path
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = {"reports": collect_reports(root, candidate), "default reader": default_reader(root / "reader")}
    Path(paths[0]).write_text(json.dumps(data, indent=2) + "\n")
