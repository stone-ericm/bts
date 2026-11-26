"""Compare the validator, loaders and live judge to a wire-format oracle."""

import builtins
import datetime
import itertools
import json
import re
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from offline import forbid_network

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
sys.path[:0] = [str(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "src"), str(ROOT)]
forbid_network()

from bts import contest_state, picks
from tests.test_pick_chosen_on_cs5a import DAY, NOW, account_file, daily_pick, set_automatic, set_dated

KINDS = ["dated", "automatic", "model", "legacy_manual"]
SAVERS = ["not_earned", "active", "used"]
OMIT = object()


def wire(kind, streak=7, best=9, saver="used", required=True, effective=DAY):
    identity = dict(effective_date=effective, active_streak=streak, best_streak=best, saver_state=saver)
    return dict(schema="bts_pick_chosen_on_v1", kind=kind, required=required, streak=streak,
                saver_available=saver == "active" or (saver == "not_earned" and streak < 10),
                dated=identity if kind == "dated" else None)


def cases():
    yield "missing", OMIT, False
    for malformed in (None, 7, "dated", [], False, {}):
        yield repr(malformed), malformed, False
    for kind, required in itertools.product(KINDS, [True, False]):
        yield f"{kind}/{required}", wire(kind, required=required), True
    for streak, saver, flag in itertools.product([0, 9, 10, 15, 16, 57], SAVERS, [True, False]):
        candidate = wire("dated", streak=streak, best=max(streak, 17), saver=saver)
        available = candidate["saver_available"]
        candidate["saver_available"] = flag
        yield f"dated saver {streak}/{saver}/{flag}", candidate, flag == available
    for kind, count in itertools.product(KINDS, [-1, 0, 57, 58, 900]):
        candidate = wire(kind, streak=count, best=max(count, 9))
        valid = count >= 0 and (kind in {"model", "legacy_manual"} or count <= 57)
        yield f"{kind} count {count}", candidate, valid
    baseline = wire("dated")
    for depth, template in enumerate([baseline, baseline["dated"]]):
        for key in [*template, "extra"]:
            candidate = deepcopy(baseline)
            target = candidate["dated"] if depth else candidate
            if key == "extra":
                target[key] = "value"
            else:
                del target[key]
            yield f"keyset {depth}/{key}", candidate, False
    for field in ["schema", "required", "saver_available", "kind", "streak", "effective_date",
                  "active_streak", "best_streak", "saver_state"]:
        invalid = [[], {}, None, 1.0]
        if field in {"required", "saver_available"}:
            invalid += [1, 0, "false"]
        elif field in {"streak", "active_streak", "best_streak"}:
            invalid += [True, False, "7"]
        else:
            invalid += [True, 1, "wrong", ""]
        for value in invalid:
            candidate = deepcopy(baseline)
            target = candidate if field in baseline else candidate["dated"]
            target[field] = value
            yield f"type {field}/{value!r}", candidate, False
    for raw, valid in [("2027-7-15", False), ("2027-02-29", False), ("2028-02-29", True),
                       ("20270715", False), ("2027-W28-4", False), ("2027-07-15\x00", False),
                       ("0000-07-15", False), ("9999-12-31", True), ("2027-07-16", True)]:
        yield f"date {raw!r}", wire("dated", effective=raw), valid
    for key, value in [("best_streak", 6), ("active_streak", 6), ("active_streak", -1), ("best_streak", 58)]:
        candidate = deepcopy(baseline)
        candidate["dated"][key] = value
        yield f"cross-field {key}/{value}", candidate, False
    for kind in KINDS:
        candidate = wire(kind)
        candidate["dated"] = None if kind == "dated" else baseline["dated"]
        yield f"identity on {kind}", candidate, False
    for count in (False, True):
        yield f"automatic boolean {count}", dict(wire("automatic"), streak=count), False
        # Even with an agreeing integer inner count, an outer bool is invalid.
        yield f"dated boolean {count}", dict(wire("dated", streak=int(count)), streak=count), False
        candidate = wire("dated", streak=int(count), best=int(count))
        candidate["dated"]["active_streak"] = count
        yield f"dated active boolean {count}", candidate, False
        candidate = wire("dated", streak=0, best=int(count))
        candidate["dated"]["best_streak"] = count
        yield f"dated best boolean {count}", candidate, False
    for best in (8, 10):
        yield f"dated best {best}", wire("dated", best=best), True
    yield "dated saver correction", wire("dated", saver="active"), True
    yield "dated active correction", wire("dated", streak=8), True


def live_states(root):
    directory = root / "live"
    set_dated(directory)
    dated = contest_state.load_decision_streak_state(directory, require_contest_state=True, now=NOW)
    yield dated
    manual = directory / "account_state" / "contest_streak.manual.json"
    raw = json.loads(manual.read_text())
    raw.update(recorded_at="2027-07-15T10:01:00-04:00", reason="corrected spelling")
    manual.write_text(json.dumps(raw))
    unchanged = contest_state.load_decision_streak_state(directory, require_contest_state=True, now=NOW)
    assert dated.dated_identity == unchanged.dated_identity
    yield unchanged
    manual.unlink()
    set_automatic(directory, streak=30)
    yield contest_state.load_decision_streak_state(directory, require_contest_state=True, now=NOW)
    for kind in ["model", "legacy_manual"]:
        pdir = root / kind
        if kind == "legacy_manual":
            account_file(pdir, "contest_streak.manual.json", {"streak": 7, "source": "mlb_bts_profile"})
        reading = contest_state.load_decision_streak_state(pdir, now=NOW)
        assert reading.observation_kind == kind
        yield replace(reading, streak=7)


def compare(root):
    live = list(live_states(root))
    daily = daily_pick()
    body = asdict(daily)
    del body["chosen_on"]
    pick_path = root / f"{DAY}.json"
    shadow_path = root / f"{DAY}.shadow.json"
    checks = 0
    for label, value, valid in cases():
        decoded = dict(body)
        if value is not OMIT:
            decoded["chosen_on"] = value
        raw = json.dumps(decoded).encode()
        pick_path.write_bytes(raw)
        shadow_path.write_bytes(raw)
        present = value if valid else None
        loaded, stored = picks.load_pick_bytes(DAY, root)
        assert stored == raw and loaded.chosen_on == present, label
        assert picks.load_pick(DAY, root).chosen_on == present, label
        assert picks.load_shadow_pick(DAY, root).chosen_on == present, label
        daily.chosen_on = None if value is OMIT else value
        forbidden = Mock(side_effect=AssertionError("judge accessed file or clock"))
        with patch.object(builtins, "open", forbidden), patch.object(Path, "open", forbidden), \
                patch.object(picks, "datetime", forbidden), patch.object(datetime, "datetime", forbidden):
            for state in live:
                matching = valid and value["required"] and value["kind"] == state.observation_kind and (
                    state.observation_kind == "automatic"
                    or state.observation_kind == "dated" and value["dated"] == state.dated_identity
                )
                refusal = picks.saved_pick_refusal(daily, state)
                assert (refusal is None) == matching, (label, state, refusal)
                if refusal is not None:
                    assert re.search(r"Saved pick for .* was not delivered.*The live state is ", refusal)
                    assert refusal.endswith("A later scheduled check or fallback refresh may choose a replacement, if one remains.")
                checks += 1
            assert not forbidden.called
    print(f"{checks} matcher checks across {len(live)} real snapshots and {checks // len(live)} wire inputs")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=REVIEW) as temp:
        compare(Path(temp))
