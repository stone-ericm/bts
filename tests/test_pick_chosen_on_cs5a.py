"""Chosen-on persistence and matching on pinned contest observations."""

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from bts import contest_state, picks


DAY = "2027-07-15"
NOW = datetime.fromisoformat(f"{DAY}T18:35:00-04:00")
SAVERS = {"not_earned", "active", "used"}
KINDS = ("dated", "automatic", "model", "legacy_manual")


def daily_pick():
    return picks.DailyPick(
        DAY, NOW.isoformat(),
        picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                   0.82, [], False, 101, f"{DAY}T23:10:00Z"),
        None, None,
    )


def account_file(path, filename, body):
    target = path / "account_state" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body))


def set_dated(path, now=NOW, streak=7, best=9, saver="used"):
    day = now.astimezone(ZoneInfo("America/New_York")).date()
    account_file(path, "contest_streak.manual.json", {
        "schema_version": "bts_contest_streak_manual_v3", "season": day.year,
        "effective_date": day.isoformat(), "active_streak": streak,
        "best_streak": best, "saver_state": saver, "recorded_at": now.isoformat(),
        "expires_at": f"{day}T23:59:59.999999-04:00", "source": "manual_cli",
    })


def set_automatic(path, now=NOW, streak=7, *, covered=True):
    year = now.year if covered else now.year - 1
    account_file(path, "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v2", "season": year,
        "activation_id": "a" * 32, "active_streak": streak, "best_streak": streak,
        "source": "manual_cli", "source_date": now.date().replace(year=year).isoformat(),
        "recorded_at": now.replace(year=year).isoformat(),
    })
    if covered:
        account_file(path, "season_contract.json", {
            "schema_version": "bts_season_contract_v1", "season": year,
            "status": "automatic", "activation_id": "a" * 32,
            "switched_at": now.isoformat(), "revoked_at": None, "note": None,
        })


def record(kind="dated", *, required=True):
    return {
        "schema": "bts_pick_chosen_on_v1", "required": required, "kind": kind,
        "streak": 7, "saver_available": False,
        "dated": {"effective_date": DAY, "active_streak": 7, "best_streak": 9,
                  "saver_state": "used"} if kind == "dated" else None,
    }


# Set one value (or omit it) per case so each invalid input is attributable.
INVALID = [
    (("schema",), "v0"), (("schema",), []), (("required",), 1),
    (("required",), "true"), (("kind",), {}), (("kind",), "unknown"),
    (("streak",), True), (("streak",), 7.0), (("streak",), "7"),
    (("streak",), -1), (("streak",), 58), (("streak",), 8),
    (("saver_available",), 0), (("saver_available",), True),
    (("dated",), None), (("dated",), []), (("extra",), 0),
    (("dated", "extra"), 0), (("dated", "effective_date"), "2027-7-15"),
    (("dated", "effective_date"), "2027-02-30"),
    (("dated", "effective_date"), 20270715),
    (("dated", "effective_date"), "20270715"),
    (("dated", "active_streak"), True), (("dated", "active_streak"), 8),
    (("dated", "active_streak"), -1), (("dated", "best_streak"), 6),
    (("dated", "best_streak"), 58), (("dated", "best_streak"), None),
    (("dated", "best_streak"), False), (("dated", "saver_state"), []),
    (("dated", "saver_state"), "inactive"),
]


def invalid_records():
    yield "non-object", []
    for path, value in INVALID:
        src = record()
        target = src if len(path) == 1 else src["dated"]
        target[path[-1]] = value
        yield f"{'.'.join(path)}={value!r}", src
    for path in [(key,) for key in record()] + [
        ("dated", key) for key in record()["dated"]
    ]:
        src = record()
        target = src if len(path) == 1 else src["dated"]
        target.pop(path[-1])
        yield f"omit {'.'.join(path)}", src
    yield "non-dated with identity", dict(record(), kind="automatic")
    yield "automatic at 58", dict(record("automatic"), streak=58)


@pytest.mark.parametrize("kind", KINDS)
def test_real_decision_state_identity_and_builder(tmp_path, kind):
    if kind == "dated":
        set_dated(tmp_path)
    elif kind == "automatic":
        set_automatic(tmp_path)
    elif kind == "legacy_manual":
        account_file(tmp_path, "contest_streak.manual.json", {
            "active_streak": 58, "source": "mlb_bts_profile",
        })
    required = kind in {"dated", "automatic"}
    state = contest_state.load_decision_streak_state(
        tmp_path, now=NOW, require_contest_state=required)
    expected = dict(record(kind, required=required), streak=state.streak,
                    saver_available=state.saver_available)
    assert state.observation_kind == kind
    assert state.dated_identity == expected["dated"]
    actual = picks.chosen_on_record(state, required=required)
    assert actual == expected
    if state.dated_identity is not None:
        state.dated_identity["best_streak"] += 1
        assert actual == expected
    with pytest.raises(TypeError):
        picks.chosen_on_record(state, required=int(required))
    if not required:
        with pytest.raises(contest_state.ContestStateError):
            contest_state.load_decision_streak_state(
                tmp_path, now=NOW, require_contest_state=True)


@pytest.mark.parametrize("kind", KINDS)
def test_round_trip_and_exact_prior_encoding(tmp_path, kind):
    daily = daily_pick()
    original = json.dumps({key: val for key, val in asdict(daily).items()
                           if key != "chosen_on"}, indent=2)
    chosen = record(kind)
    if kind in {"model", "legacy_manual"}:
        chosen["streak"] = 58
    daily.chosen_on = chosen
    path = picks.save_pick(daily, tmp_path)
    raw = path.read_bytes()
    parsed = json.loads(raw)
    assert list(parsed)[-1] == "chosen_on"
    assert parsed.pop("chosen_on") == chosen
    assert json.dumps(parsed, indent=2) == original
    assert raw == (original[:-2] + ",\n" + json.dumps({"chosen_on": chosen}, indent=2)[2:]).encode()
    assert picks.load_pick(DAY, tmp_path).chosen_on == chosen
    loaded, held = picks.load_pick_bytes(DAY, tmp_path)
    assert held == raw and loaded.chosen_on == chosen
    shadow = picks.save_shadow_pick(daily, tmp_path)
    shadow_body = json.loads(shadow.read_bytes())
    del shadow_body["double_down"], shadow_body["runner_up"]
    shadow.write_text(json.dumps(shadow_body))
    assert picks.load_shadow_pick(DAY, tmp_path).chosen_on == chosen
    assert "chosen_on" not in (tmp_path / f"lineup_evolution_{DAY}.jsonl").read_text()


@pytest.mark.parametrize("value", [None] + [value for _, value in invalid_records()],
                         ids=["null"] + [name for name, _ in invalid_records()])
def test_invalid_record_normalized_in_all_loaders(tmp_path, value):
    body = {**asdict(daily_pick()), "chosen_on": value}
    raw = json.dumps(body).encode()
    (tmp_path / f"{DAY}.json").write_bytes(raw)
    (tmp_path / f"{DAY}.shadow.json").write_bytes(raw)
    assert getattr(picks.load_pick(DAY, tmp_path), "chosen_on", None) is None
    loaded, held = picks.load_pick_bytes(DAY, tmp_path)
    assert held == raw
    assert getattr(loaded, "chosen_on", None) is None
    assert getattr(picks.load_shadow_pick(DAY, tmp_path), "chosen_on", None) is None


def test_legacy_read_rewrite_and_projection(tmp_path):
    from scripts import live_forward_capture_once as capture
    from scripts import live_forward_provenance_inventory as inventory

    daily = daily_pick()
    old = {key: value for key, value in asdict(daily).items() if key != "chosen_on"}
    path = tmp_path / f"{DAY}.json"
    path.write_text(json.dumps(old, indent=2))
    loaded, raw = picks.load_pick_bytes(DAY, tmp_path)
    assert getattr(loaded, "chosen_on", None) is None
    picks.save_pick(loaded, tmp_path)
    new = json.loads(path.read_bytes())
    for snapshot in (capture.decision_snapshot, inventory._decision_snapshot):
        assert snapshot(old) == snapshot(dict(old, chosen_on=None))
        assert snapshot(old) != snapshot(dict(old, chosen_on=record()))
        assert snapshot(dict(old, chosen_on=record())) != snapshot(
            dict(old, chosen_on=record(required=False)))
    assert path.read_bytes() != raw
    assert new.pop("chosen_on") is None and new == old


def matcher_cases():
    yield "missing", None, None
    yield "null", None, None
    yield "optional", record(required=False), None
    for kind in KINDS:
        yield kind, record(kind), kind if kind in {"automatic", "dated"} else None
    for name, change in (
        ("date", {"effective_date": "2027-07-14"}),
        ("active", {"active_streak": 8}),
        ("best", {"best_streak": 10}),
        ("saver", {"saver_state": "not_earned"}),
    ):
        chosen = record()
        chosen["dated"].update(change)
        chosen["streak"] = chosen["dated"]["active_streak"]
        if name == "saver":
            chosen["saver_available"] = True
        yield name, chosen, None
    for name, chosen in invalid_records():
        yield name, chosen, None


@pytest.mark.parametrize("live_kind", ["dated", "automatic"])
@pytest.mark.parametrize("name,chosen,matches_kind", list(matcher_cases()),
                         ids=[name for name, _, _ in matcher_cases()])
def test_pure_matcher(tmp_path, monkeypatch, live_kind, name, chosen, matches_kind):
    from unittest.mock import Mock

    (set_dated if live_kind == "dated" else set_automatic)(tmp_path)
    state = contest_state.load_decision_streak_state(
        tmp_path, now=NOW, require_contest_state=True)
    daily = daily_pick()
    daily.chosen_on = deepcopy(chosen)
    forbidden = Mock(side_effect=AssertionError("matcher read a file or clock"))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(picks, "datetime", forbidden)
        message = picks.saved_pick_refusal(daily, state)
        assert not forbidden.called
    if live_kind == matches_kind:
        assert message is None
        return
    assert isinstance(message, str)
    absent = name in {"missing", "null"} or name in dict(invalid_records())
    chosen_text = ("no valid record" if absent else
                   "chosen without the live check (for example by the preview)"
                   if name == "optional" else chosen["kind"])
    final = ("A later scheduled check or fallback refresh may choose a replacement, "
             "if one remains.")
    assert message.endswith(final)
    assert message.index(DAY) < message.index(chosen_text) < message.index("live state")
    live_text = message.split("live state", 1)[1]
    assert live_kind in live_text and "not delivered" in message
    if live_kind == "dated":
        for value in state.dated_identity.values():
            assert str(value) in live_text


def test_raw_hashes_and_historical_projections(tmp_path):
    from hashlib import sha256

    from bts.entry_receipt import EntryReceipt
    from bts.state.export import _pick_to_historical
    from bts.state.regenerate import _hist_to_pick_file
    from scripts.audit.season_ledger.sources.pick_files import parse_pick_file

    daily = daily_pick()
    old = asdict(daily)
    old.pop("chosen_on", None)
    path = tmp_path / f"{DAY}.json"
    path.write_text(json.dumps(old, indent=2))
    _, raw_old = picks.load_pick_bytes(DAY, tmp_path)
    daily.chosen_on = record()
    picks.save_pick(daily, tmp_path)
    summary = []
    for raw in (raw_old, path.read_bytes()):
        path.write_bytes(raw)
        loaded, held = picks.load_pick_bytes(DAY, tmp_path)
        receipt = EntryReceipt(et_date=DAY, started_at=NOW, clock=lambda: NOW,
                               expected_username=None, picks_dir=tmp_path)
        receipt.selection(loaded, held, None, None)
        selection = receipt.record()["selection"]
        ledger, = parse_pick_file(path.name, raw).rows
        assert selection["pick_file_sha256"] == ledger["content_sha256"] == sha256(raw).hexdigest()
        assert json.loads(ledger["record_raw_json"]) == json.loads(raw)
        hist = _pick_to_historical(json.loads(raw))
        assert "chosen_on" not in hist and "chosen_on" not in _hist_to_pick_file(hist)
        summary.append((selection, ledger, hist))
    first, second = summary
    assert first[0]["slots"] == second[0]["slots"]
    assert first[0]["pick_file_sha256"] != second[0]["pick_file_sha256"]
    assert first[1]["obs_id"] != second[1]["obs_id"] and first[2] == second[2]
