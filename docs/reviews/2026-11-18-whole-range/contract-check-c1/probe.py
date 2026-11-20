"""Offline contract probes against the real scheduler, health runner and check."""

import argparse
from contextlib import redirect_stderr
from datetime import date, datetime, time, timedelta, timezone
import io
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLICES = ROOT / "docs/reviews/2026-11-18-whole-range"
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
D = date(2027, 3, 24)
YEAR_END = date(2027, 12, 31)
MATCH = "a" * 32
POINTER = (
    "This check used the files available when it ran. Do not enter a pick, retry delivery, "
    "or change contest state based on this alert. Any new-day correction must wait until MLB "
    "shows the prior slate settled. For the new day, the morning state check, when available, "
    "and the current live refusal govern. Season contract status describes the current files."
)


def deny_network(*args, **kwargs):
    raise AssertionError("This probe is offline")


def at(day, hours=0, minutes=0):
    return datetime.combine(day, time(hours, minutes), ET)


def end(day):
    return datetime.combine(day, time.max, ET).astimezone(UTC)


def put(picks, name, body):
    path = picks / "account_state" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def dated(picks, day=D, streak=0, saver="not_earned"):
    from bts.contest_state import DATED_MANUAL_SCHEMA

    return put(picks, "contest_streak.manual.json", {
        "schema_version": DATED_MANUAL_SCHEMA, "season": day.year,
        "effective_date": day.isoformat(), "active_streak": streak, "best_streak": streak,
        "saver_state": saver, "recorded_at": at(day, 9).isoformat(),
        "source": "probe", "expires_at": end(day).isoformat(),
    })


def auto(picks, day=D, **changes):
    from bts.season_contract import AUTO_SCHEMA

    return put(picks, "contest_streak.json", {
        "schema_version": AUTO_SCHEMA, "season": day.year, "activation_id": MATCH,
        "active_streak": 12, "best_streak": 12,
        "source_date": (day - timedelta(days=2)).isoformat(),
        "recorded_at": at(day, 20).isoformat(), **changes,
    })


def switch(picks, day=D, status="automatic", **changes):
    from bts.season_contract import SCHEMA

    return put(picks, "season_contract.json", {
        "schema_version": SCHEMA, "season": day.year, "activation_id": MATCH,
        "status": status, "switched_at": at(day).isoformat(),
        "revoked_at": at(day).isoformat() if status == "revoked" else None, **changes,
    })


def saver_flag(picks, day=D, state="active"):
    return put(picks, "saver_state.json", {"season": day.year, "state": state})


def settled(picks, day):
    (picks / f"{day}.json").write_text('{"result":"hit"}', encoding="utf-8")


def real_runner(picks, day, wall, *, expected=True, send=None):
    from bts.health import alert, contest_state, runner

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz) if tz is not None else wall.replace(tzinfo=None)

    transport = []

    def capture(recipient, text):
        transport.append(text)
        return "fake-receipt" if send is None else send(recipient, text)

    with patch.object(contest_state, "datetime", Clock), patch.object(alert, "send_dm", capture):
        alerts = runner.run_all_checks(picks, picks.parent / "models", "operator.invalid",
                                       today=day, contest_state_expected=expected)
    state_alerts = [a for a in alerts if a.source in {"contest_state", "saver_state"}
                    or a.incident_key == "health_runner:contest_state"]
    lines = [line for body in transport for line in body.splitlines()
             if line.startswith(("- [contest_state]", "- [saver_state]", "- [health_runner]"))]
    return state_alerts, lines


def cases(root):
    """Yield real branch fixtures, with proposed historical reasons paired with base text."""
    def picks_for(label):
        picks = root / label / "picks"
        picks.mkdir(parents=True)
        return picks

    def seed(label, day=D, **changes):
        picks = picks_for(label)
        auto(picks, day, **changes)
        switch(picks, day)
        saver_flag(picks, day)
        return picks

    yield "absent", picks_for("absent"), D, [
        "Contest streak observation is missing from account_state.", "The season switch is absent.",
    ]
    picks = picks_for("approved_no_auto")
    switch(picks)
    yield "approved_no_auto", picks, D, [
        "Contest streak observation is missing from account_state.",
        "There is no automatic observation for the switch activation.",
    ]
    picks = seed("optional_bypass")
    switch(picks, season="invalid")
    yield "optional_bypass", picks, D, []
    picks = seed("healthy_dated")
    dated(picks)
    switch(picks, status="revoked")
    saver_flag(picks, state="not_earned")
    yield "healthy_dated", picks, D, []
    picks = seed("dated_saver")
    dated(picks, streak=12)
    yield "dated_saver", picks, D, [
        "The saver state in the dated observation contradicts its streak. The saver counted as spent.",
    ]
    picks = seed("legacy_expired")
    dated(picks)
    put(picks, "contest_streak.manual.json", {"streak": 10})
    yield "legacy_expired", picks, D, ["A legacy or expired manual override file is present."]
    picks = seed("legacy_selected")
    put(picks, "contest_streak.manual.json", {"streak": 11, "source_date": D.isoformat(),
                                             "override_expires_at": at(D + timedelta(days=1), 1).isoformat()})
    yield "legacy_selected", picks, D, ["The selected legacy manual override is ineligible for a live decision."]
    picks = seed("corrupt_manual")
    dated(picks).write_text("{", encoding="utf-8")
    yield "corrupt_manual", picks, D, ["The manual contest streak file is malformed or unreadable."]
    picks = seed("corrupt_auto", active_streak="`bts season-contract enable`\nnext line")
    yield "corrupt_auto", picks, D, ["The automatic contest streak file has an invalid streak."]
    picks = seed("corrupt_switch")
    switch(picks, season="invalid")
    yield "corrupt_switch", picks, D, ["The season switch is corrupt because its season field failed validation."]
    picks = seed("refused_revoked")
    switch(picks, status="revoked")
    yield "refused_revoked", picks, D, ["The season switch is revoked."]
    picks = seed("refused_other")
    switch(picks, season=D.year + 1)
    yield "refused_other", picks, D, ["The season switch is for another season."]
    picks = seed("schema_text", schema_version="`bts fetch-contest-streak`\nnext line")
    yield "schema_text", picks, D, ["The automatic observation uses an unsupported schema."]
    picks = seed("schema_none", schema_version=None)
    yield "schema_none", picks, D, ["The automatic observation schema is missing, null or invalid."]
    picks = seed("auto_season", season=D.year + 1)
    yield "auto_season", picks, D, ["The automatic observation is for another season."]
    picks = seed("auto_activation", activation_id="b" * 32)
    yield "auto_activation", picks, D, ["The automatic observation and season switch have different activation IDs."]
    picks = seed("undated", source_date=None, recorded_at=None)
    yield "undated", picks, D, ["The automatic contest observation is undated."]
    picks = seed("missing_source", source_date=None)
    yield "missing_source", picks, D, [
        "The contest streak file has no coverage date, so coverage and freshness could not be verified.",
    ]
    picks = seed("future_source", source_date=(D + timedelta(days=3)).isoformat())
    yield "future_source", picks, D, ["The contest streak file has a coverage date beyond the allowed future grace."]
    picks = seed("grace", source_date=(D + timedelta(days=2)).isoformat())
    yield "grace", picks, D, []
    picks = seed("coverage_one")
    settled(picks, D)
    yield "coverage_one", picks, D, ["Coverage trails by one settled pick, the expected ledger lag."]
    picks = seed("coverage_two")
    settled(picks, D - timedelta(days=1))
    settled(picks, D)
    yield "coverage_two", picks, D, ["Coverage trails by more than one settled pick. The predictions array can trail the live counter."]
    picks = seed("later_pick")
    settled(picks, D)
    settled(picks, D + timedelta(days=1))
    yield "later_pick", picks, D, ["Coverage trails by more than one settled pick. The predictions array can trail the live counter."]
    picks = seed("flag_missing")
    (picks / "account_state/saver_state.json").unlink()
    yield "flag_missing", picks, D, ["The saver flag is uninitialized or unreadable. The saver counted as spent."]
    picks = seed("flag_contradiction")
    saver_flag(picks, state="not_earned")
    yield "flag_contradiction", picks, D, ["The saver flag contradicts the recorded streak. The saver counted as spent."]
    picks = seed("new_year_good", YEAR_END)
    yield "new_year_good", picks, YEAR_END, []
    picks = seed("new_year_refused", YEAR_END)
    switch(picks, YEAR_END, status="revoked")
    saver_flag(picks, YEAR_END + timedelta(days=1))
    yield "new_year_refused", picks, YEAR_END, [
        "The saver flag is for another season. The saver counted as spent.", "The season switch is revoked.",
    ]


def messages(root):
    from bts.health.alert import Alert, format_health_dm_body
    from bts.health.attention import build_policy_alerts

    for label, picks, day, reasons in cases(root):
        expected = label != "optional_bypass"
        original, sent = real_runner(picks, day, at(day + timedelta(days=1), 0, 15), expected=expected)
        # Replay using the proposed evaluation clock. Reset only the synthetic health receipt.
        receipt = picks.parent / "health_state" / "health_dm_delivery_status.json"
        receipt.unlink(missing_ok=True)
        reference, _ = real_runner(picks, day, end(day), expected=expected)
        assert len(reference) == len(reasons), (label, reference, reasons)
        print(f"\n{label} base wall alerts ({len(original)}):")
        for alert in original:
            print(f"  {alert.level} [{alert.source}] {alert.message}")
        for line in sent:
            print(f"  Sent: {line}")
        historical = [Alert(item.level, item.source, f"Prior slate {day} ET. {reason} {POINTER}", item.incident_key)
                      for item, reason in zip(reference, reasons, strict=True)]
        _, attention = build_policy_alerts(historical, streaks={})
        dm = format_health_dm_body([item for item in historical if item.level == "CRITICAL"], attention)
        for item in historical:
            assert item.message.endswith(POINTER)
            assert not any(word in item.message for word in (
                "set-contest-streak", "saver-state", "season-contract enable", "season-contract revoke", "fetch-contest-streak",
            ))
            print(f"  Candidate {item.level} [{item.source}] {item.message}")
        if dm:
            print(f"  Candidate DM ({len(dm)} characters):\n{dm}")
        if label == "dated_saver":
            assert f"--date {day}" in reference[0].message
            assert f"--date {day}" not in "\n".join(a.message for a in original)
            assert dm.endswith("(1st observed day)") and not dm.endswith(POINTER)
        if label == "new_year_good":
            assert len(original) == 2 and "other_season" in original[-1].message
        if label == "healthy_dated":
            assert len(original) == 2
        if label == "corrupt_auto":
            assert "bts season-contract enable" in reference[0].message
        if label == "schema_text":
            assert "bts fetch-contest-streak" in reference[0].message.splitlines()[0]


def edges(root):
    from bts.health import contest_state as health

    for name, day in [("summer", D), ("new_year", YEAR_END)]:
        picks = root / name / "picks"
        auto(picks, day)
        switch(picks, day)
        saver_flag(picks, day)
        expiry = datetime.combine(day, time(23, 59, 59, 500000), ET)
        put(picks, "contest_streak.manual.json", {"streak": 12, "source_date": day.isoformat(),
                                                 "override_expires_at": expiry.isoformat()})
        earlier = datetime.combine(day, time(23, 59, 59), ET).astimezone(UTC)
        whole = [a.message.splitlines()[0] for a in health.check(picks, expected=True, now=earlier)]
        last = [a.message.splitlines()[0] for a in health.check(picks, expected=True, now=end(day))]
        assert "selected legacy manual" in whole[0] and "legacy/expired" in last[0]
        print(f"{day} whole second {earlier}: {whole}; last microsecond {end(day)}: {last}")

    picks = root / "overwritten" / "picks"
    next_day = D + timedelta(days=1)
    dated(picks, streak=12)
    assert health.check(picks, expected=True, now=end(D))[0].source == "saver_state"
    dated(picks, day=next_day)
    after = health.check(picks, expected=True, now=end(D))
    assert any("expected but missing" in a.message for a in after)
    print("next dated write erased the real dated saver WARN and introduced missing + switch CRITICALs")

    picks = root / "overnight_fetch" / "picks"
    switch(picks)
    assert health.check(picks, expected=True, now=end(D))
    auto(picks, recorded_at=at(next_day, 1, 10).isoformat(), source_date=D.isoformat())
    saver_flag(picks)
    reference = health.check(picks, expected=True, now=end(D))
    assert not reference
    print("01:10 file replacement erased the genuine missing auto CRITICAL at today's end-of-D replay")

    picks = root / "restarts" / "picks"
    switch(picks)
    wall = at(next_day, 2)
    first, sent = real_runner(picks, D, wall)
    second, duplicate = real_runner(picks, D, wall)
    _, following = real_runner(picks, next_day, wall)
    receipt = picks.parent / "health_state/health_dm_delivery_status.json"
    assert first == second and sent and not duplicate and following
    assert json.loads(receipt.read_text())["sent_day"] == str(next_day)
    print("restart in same D suppressed after confirmed send; D+1 still sent")

    picks = root / "send_failure" / "picks"
    switch(picks)

    def fail(*args):
        raise OSError("synthetic acknowledgement lost")

    real_runner(picks, D, wall, send=fail)
    receipt = picks.parent / "health_state/health_dm_delivery_status.json"
    assert json.loads(receipt.read_text())["status"] == "failed"
    _, retry = real_runner(picks, D, wall, send=lambda *_: None)
    assert retry and json.loads(receipt.read_text())["status"] == "sent"
    print("raised send marked failed and retried; None receipt retained generic sent behavior")

    picks = root / "interrupt" / "picks"
    dated(picks)
    observation = picks / "account_state/contest_streak.manual.json"
    snapshot = observation.read_bytes()
    with patch.object(health, "load_contest_streak_state", side_effect=KeyboardInterrupt):
        try:
            real_runner(picks, D, wall)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("interrupt swallowed")
    assert observation.read_bytes() == snapshot
    print("KeyboardInterrupt propagated with unchanged contest bytes")

    for kind in ("manual", "automatic", "switch", "saver", "coverage"):
        try:
            subprocess.run([sys.executable, str(HERE / "probe.py"), "fifo-child", kind, str(root)],
                           check=True, timeout=3, capture_output=True)
        except subprocess.TimeoutExpired:
            print(f"FIFO {kind}: timed out in base check")
        else:
            raise AssertionError(f"FIFO {kind} did not block")


def fifo_child(kind, root):
    from bts.health import contest_state as health

    picks = root / f"fifo-{kind}" / "picks"
    observation = auto(picks)
    contract = switch(picks)
    flag = saver_flag(picks)
    target = {"manual": picks / "account_state/contest_streak.manual.json", "automatic": observation,
              "switch": contract, "saver": flag, "coverage": picks / f"{D}.json"}[kind]
    target.unlink(missing_ok=True)
    os.mkfifo(target)
    print(f"FIFO ready: {kind}", flush=True)
    health.check(picks, expected=True, now=end(D))


def scheduler(root):
    sys.path.insert(0, str(SLICES / "slice-1-carbon"))
    import test_slice1_probes_v3 as carbon

    carbon._H.REPO_ROOT = ROOT
    journal = io.StringIO()
    with redirect_stderr(journal):
        try:
            carbon.test_probe_1_post_midnight_eod_false_critical_and_harmful_advice(root)
        except Exception:
            print(journal.getvalue()[-5000:], file=sys.stderr)
            raise
    print(json.dumps(carbon.EVIDENCE, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("messages", "edges", "scheduler", "fifo-child"))
    parser.add_argument("child", nargs="*")
    args = parser.parse_args()
    if args.mode == "fifo-child":
        fifo_child(args.child[0], Path(args.child[1]))
        return
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    try:
        with TemporaryDirectory(dir=HERE) as scratch:
            scratch = Path(scratch)
            if args.mode == "scheduler":
                scheduler(scratch)
            else:
                os.chdir(scratch)
                with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
                    {"messages": messages, "edges": edges}[args.mode](scratch)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
