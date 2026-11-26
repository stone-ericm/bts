"""Drive the real decision loader and dispatcher on the original and scratch source."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.audit.season_walk.walk import ET, _invoke, _set_clock, offline
from bts import contest_state as cs, picks as pick_mod, scheduler
from bts.day_outcome import DM_MAX_CODE_POINTS
from bts.health import alert
from bts.season_contract import AUTO_SCHEMA


try:
    import regex
except ImportError:
    regex = None

MODE = sys.argv[1]
BOX = "/home/bts/projects/bts/data/picks"
AT = datetime(2027, 3, 24, 16, 1, tzinfo=ET)
DAY = AT.date().isoformat()
logging.disable(logging.CRITICAL)


def dump(path, data):
    path.write_text(json.dumps(data))


def invoke(*args):
    code, output = _invoke(*args)
    assert code == 0, output
    return output


def base_auto():
    return {"active_streak": 12, "best_streak": 15, "source_date": "2027-03-23"}


def dated():
    return {"schema_version": cs.DATED_MANUAL_SCHEMA, "effective_date": DAY, "season": 2027,
            "expires_at": f"{DAY}T23:59:59.999999-04:00", "active_streak": 12, "best_streak": 15,
            "saver_state": "used", "recorded_at": f"{DAY}T13:00:00-04:00", "source": "MLB"}


def measure(body):
    return len(body), len(regex.findall(r"\X", body)) if regex else None


def capture(cfg, status, error, prior_count=None, day=DAY):
    if prior_count is not None:
        dump(status, {"status": "failed", "critical_count": prior_count})
    elif status.exists():
        status.unlink()
    attempts = []

    def send(recipient, body):
        attempts.append(body)
        if len(body) > DM_MAX_CODE_POINTS:
            raise RuntimeError("length rejected")
        return "wire"

    kwargs = {"slate_date": day} if MODE == "scratch" else {}
    with patch.object(alert, "send_dm", send):
        scheduler._alert_contest_state_failure(cfg, error, **kwargs)
    assert len(attempts) == 1
    return (*measure(attempts[0]), json.loads(status.read_text())["status"]), attempts[0]


def fixtures(case, variant, root, account):
    switch_path = account / "season_contract.json"
    automatic = account / "contest_streak.json"
    manual = account / "contest_streak.manual.json"
    if case == "absent":
        return
    if case == "other_season":
        _set_clock(AT.replace(year=2026))
        invoke("season-contract", "enable", "--season", 2026, "--picks-dir", root)
        _set_clock(AT)
        return
    invoke("season-contract", "enable", "--season", 2027, "--picks-dir", root)
    sw = json.loads(switch_path.read_text())
    if case == "revoked":
        invoke("season-contract", "revoke", "--season", 2027, "--reason", "restore", "--picks-dir", root)
    if case == "switch_corrupt":
        field, value = variant
        if field is None:
            switch_path.write_text("{")
        else:
            sw[field] = value
            dump(switch_path, sw)
    if case == "legacy":
        dump(manual, {**base_auto(), "override_expires_at": f"{DAY}T23:59:00-04:00"})
    if case == "authority":
        dump(automatic, {**base_auto(), "schema_version": AUTO_SCHEMA, "season": 2027,
                         "activation_id": sw["activation_id"], **variant})
    if case.startswith(("auto_", "manual_")):
        dest = automatic if case.startswith("auto_") else manual
        if variant == "bad JSON":
            dest.write_text("{")
        elif variant == "bad UTF8":
            dest.write_bytes(b"\xff")
        elif variant == "directory":
            dest.mkdir()
        else:
            dump(dest, variant)
    if case == "dated":
        field, value = variant
        dump(manual, {**dated(), field: value})
    if case == "model_json":
        (root / "streak.json").write_text("{")
    if case == "model_streak":
        dump(root / "streak.json", {"season": 2027, "streak": False})
    if case == "saved":
        dump(manual, dated())


def refusal(case, variant, root):
    try:
        decision = cs.load_decision_streak_state(
            root, require_contest_state=not case.startswith("model_"), now=AT
        )
    except cs.ContestStateError as error:
        return cs.ContestStateError(str(error).replace(str(root), BOX))
    if case == "saved":
        from types import SimpleNamespace

        if variant == "absent":
            chosen_on = None
        else:
            chosen_on = pick_mod.chosen_on_record(decision, required=variant != "preview")
            if variant == "changed":
                chosen_on["dated"]["active_streak"] = 11
        text = pick_mod.saved_pick_refusal(SimpleNamespace(date=DAY, chosen_on=chosen_on), decision)
        assert text
        return cs.ContestStateError(text)
    raise AssertionError((case, variant))


class BrokenString(Exception):
    def __str__(self):
        raise ValueError("broken")


cases = [(key, None) for key in ("absent", "revoked", "no_fetch")]
if MODE == "scratch":
    cases += [(key, None) for key in ("other_season", "legacy", "model_json", "model_streak")]
    cases += [("switch_corrupt", v) for v in [
        (None, None), ("schema_version", "bad"), ("season", True), ("status", "bad"),
        ("activation_id", "bad"), ("switched_at", None), ("revoked_at", "bad"), ("note", False),
    ]]
    cases += [("authority", v) for v in [
        {"schema_version": None}, {"schema_version": "obsolete"}, {"season": 2026},
        {"activation_id": "0" * 32},
    ]]
    for kind in ("auto", "manual"):
        cases += [(f"{kind}_parse", v) for v in (
            "bad JSON", "bad UTF8", "directory", [], {"active_streak": True}, {"active_streak": -1},
            {"active_streak": 12, "saver_available": "yes"},
        )]
    cases += [("auto_validation", v) for v in (
        {**base_auto(), "active_streak": 401}, {**base_auto(), "best_streak": 11},
        {"active_streak": 12}, {**base_auto(), "schema_version": AUTO_SCHEMA, "season": "2027"},
        {**base_auto(), "schema_version": AUTO_SCHEMA, "season": 2027, "activation_id": "short"},
    )]
    cases += [("dated", (field, value)) for field, value in [
        ("effective_date", False), ("season", 2026), ("expires_at", "not a timestamp"),
        ("expires_at", f"{DAY}T16:00:00-04:00"), ("active_streak", 401), ("best_streak", 11),
        ("saver_state", "maybe"), ("recorded_at", f"{DAY}T16:00:00"), ("source", 1), ("reason", None),
    ]]
    cases += [("saved", v) for v in ("absent", "preview", "changed")]
    cases += [("synthetic", v) for v in ("huge", "empty line", "broken str", "combining")]

for case, variant in cases:
    with TemporaryDirectory() as temp, offline(AT):
        root = Path(temp) / "data/picks"
        account = root / "account_state"
        account.mkdir(parents=True)
        status = root.parent / "health_state/health_dm_delivery_status.json"
        status.parent.mkdir()
        cfg = {"orchestrator": {"picks_dir": str(root)}, "bluesky": {"dm_recipient": "op.invalid"}}
        if case == "synthetic":
            error = {"huge": cs.ContestStateError("x" * 100000),
                     "empty line": cs.ContestStateError("\nsecond"),
                     "broken str": BrokenString(),
                     "combining": cs.ContestStateError("e\u0301" * 500)}[variant]
        else:
            fixtures(case, variant, root, account)
            error = refusal(case, variant, root)
        plain, body = capture(cfg, status, error)
        prior_failed, _ = capture(cfg, status, error, prior_count=1)
        try:
            first = next(iter(str(error).splitlines()), "")[:320]
        except ValueError:
            first = type(error).__name__
        if MODE == "scratch":
            assert plain[0] < 1000 and prior_failed[0] < 1000 and plain[2] == prior_failed[2] == "sent"
            if case == "absent":
                assert "Live decision refused: season 2027 switch is absent." in body
                assert "bts season-contract status" in body
        print(json.dumps({"case": case, "variant": str(variant)[:200], "reason_first_line": first,
                          "plain": plain, "prior_failed": prior_failed}, ensure_ascii=False))
        if variant == "huge":
            giant_banner, _ = capture(cfg, status, error, prior_count=10**700)
            giant_day, _ = capture(cfg, status, error, day="bad" * 500)
            print("BOUND", giant_banner, giant_day)
