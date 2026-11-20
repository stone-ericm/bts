"""Item 2 probe: render every historical branch, verify exact H3/H5/L1 rules under hostile inputs, and exercise L1 worst states."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import signal
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
D = date(2027, 3, 24)
DEC31 = date(2027, 12, 31)
MATCH_ID = "a" * 32

EXPECTED_POINTER = (
    "This check used files available at run time. "
    "Do not enter picks, retry delivery, or change contest state from this alert. "
    "Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. "
    "Use the new day's morning state check, when available, and live refusal for current diagnosis. "
    "Season contract status shows current files."
)

FORBIDDEN_SUBSTRINGS = (
    "bts ",
    "set-contest-streak",
    "saver-state",
    "season-contract",
    "fetch-contest-streak",
    "--",
    "/",
    "\\",
    ".json",
    "contest_streak",
    "season_contract",
    "not_earned",
    "oom",
)


def deny_network(*_args, **_kwargs):
    raise AssertionError("Network access is forbidden")


def at_et(day: date, hr: int = 0, mn: int = 0) -> datetime:
    return datetime.combine(day, time(hr, mn), ET)


def end_et(day: date) -> datetime:
    return datetime.combine(day, time.max, ET)


def check_l1_prohibitions(text: str, day: date, *, allow_streak_label: str | None = None) -> None:
    assert "\n" not in text and "\r" not in text, text
    assert text.startswith(f"Prior slate {day.isoformat()} ET. "), text
    assert text.endswith(" " + EXPECTED_POINTER), text
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in text, (bad, text)
    if "account_state" in text:
        expected_missing = f"Prior slate {day.isoformat()} ET. Contest streak observation is missing from account_state. {EXPECTED_POINTER}"
        assert text == expected_missing, text
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    assert dates == [day.isoformat()], (dates, text)
    stripped = text.removeprefix(f"Prior slate {day.isoformat()} ET. ").removesuffix(" " + EXPECTED_POINTER)
    if allow_streak_label is not None:
        assert stripped.endswith(f" ({allow_streak_label})"), stripped
        stripped = stripped.removesuffix(f" ({allow_streak_label})")
    assert re.findall(r"\d+", stripped) == [], (stripped, text)


def run_historical(picks: Path, day: date, wall: datetime, *, expected: bool = True, isolate: bool = True, prior_streaks: dict | None = None):
    from bts.health import alert, contest_state, runner

    state_dir = picks.parent / f".h_{wall.timestamp()}"
    state_dir.mkdir(parents=True, exist_ok=True)
    warn_path = state_dir / "warn_attention_state.json"
    if prior_streaks is not None:
        warn_path.write_text(json.dumps({"schema_version": "bts_warn_attention_state_v1", "sources": prior_streaks}))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz) if tz is not None else wall.replace(tzinfo=None)

    sent_bodies: list[str] = []
    orig_safe = runner._safe_run
    with (
        patch.object(runner, "datetime", Clock),
        patch.object(contest_state, "datetime", Clock),
        patch.object(alert, "send_dm", side_effect=lambda _r, b: (sent_bodies.append(b), "msg-1")[1]),
        patch.object(runner, "_safe_run", (lambda n, fn: orig_safe(n, fn) if n == "contest_state" else []) if isolate else orig_safe),
    ):
        alerts = runner.run_all_checks(
            picks, picks.parent / "models", "operator.invalid",
            today=day, contest_state_expected=expected,
            thresholds_overrides={
                "warn_attention_state": warn_path,
                "health_dm_delivery_status": state_dir / "dm_status.json",
            },
        )
    slot_alerts = [
        a for a in alerts
        if a.source in {"contest_state", "saver_state", "analytics_job_oom"}
        or a.incident_key == "health_runner:contest_state"
    ]
    return slot_alerts, sent_bodies


def verify_all_h3_branches_and_hostile_inputs(root: Path) -> int:
    from bts.health import contest_state

    assert contest_state.HISTORICAL_POINTER == EXPECTED_POINTER

    hostile_dir = root / "oom" / "bts set-contest-streak --date 2099-01-01"

    def mk(name: str) -> Path:
        p = hostile_dir / name / "picks"
        (p / "account_state").mkdir(parents=True, exist_ok=True)
        return p

    def write_json(picks: Path, name: str, obj) -> Path:
        f = picks / "account_state" / name
        f.write_text(json.dumps(obj), encoding="utf-8")
        return f

    def seed_good(name: str, day: date = D, streak: int = 12) -> Path:
        p = mk(name)
        write_json(p, "contest_streak.json", {
            "schema_version": "bts_contest_streak_auto_v2",
            "season": day.year,
            "activation_id": MATCH_ID,
            "active_streak": streak,
            "best_streak": max(streak, 12),
            "source_date": (day - timedelta(days=1)).isoformat(),
            "recorded_at": at_et(day, 20).isoformat(),
            "source": "bts fetch-contest-streak --force\n/bad/path",
        })
        write_json(p, "season_contract.json", {
            "schema_version": "bts_season_contract_v1",
            "season": day.year,
            "activation_id": MATCH_ID,
            "status": "automatic",
            "switched_at": at_et(day, 8).isoformat(),
            "revoked_at": None,
            "note": "bts season-contract revoke --season 2027\n2099-09-09",
        })
        write_json(p, "saver_state.json", {"season": day.year, "state": "active"})
        return p

    def write_dated(p: Path, day: date = D, streak: int = 0, saver: str = "not_earned", **over) -> Path:
        body = {
            "schema_version": "bts_contest_streak_manual_v3",
            "season": day.year,
            "effective_date": day.isoformat(),
            "active_streak": streak,
            "best_streak": streak,
            "saver_state": saver,
            "recorded_at": at_et(day, 9).isoformat(),
            "expires_at": end_et(day).isoformat(),
            "source": "bts set-contest-streak --streak 99\ninjected",
            **over,
        }
        return write_json(p, "contest_streak.manual.json", body)

    branch_cases: list[tuple[str, Path, date, bool, list[str]]] = []

    # 1. Required observation missing + switch absent / switch enabled (kind 4)
    branch_cases.append(("missing_absent", mk("missing_absent"), D, True, [
        "Contest streak observation is missing from account_state.",
        "The season switch is absent.",
    ]))
    p = mk("missing_enabled")
    write_json(p, "season_contract.json", {
        "schema_version": "bts_season_contract_v1", "season": D.year, "activation_id": MATCH_ID,
        "status": "automatic", "switched_at": at_et(D, 8).isoformat(), "revoked_at": None,
    })
    branch_cases.append(("missing_enabled", p, D, True, [
        "Contest streak observation is missing from account_state.",
        "There is no automatic observation for the switch activation.",
    ]))

    # 2. Loader manual / auto malformed, invalid streak, invalid best, invalid saver
    for fname, label in [("contest_streak.manual.json", "manual"), ("contest_streak.json", "automatic")]:
        p = mk(f"{label}_malformed")
        (p / "account_state" / fname).write_text("{bts set-contest-streak\n", encoding="utf-8")
        branch_cases.append((f"{label}_malformed", p, D, True, [f"The {label} contest streak file is malformed or unreadable."]))

        p = mk(f"{label}_non_dict")
        write_json(p, fname, ["bts season-contract enable"])
        branch_cases.append((f"{label}_non_dict", p, D, True, [f"The {label} contest streak file is malformed or unreadable."]))

        p = mk(f"{label}_bad_streak")
        write_json(p, fname, {"streak": "`bts set-contest-streak --date 2027-03-25`\nline2"})
        branch_cases.append((f"{label}_bad_streak", p, D, True, [f"The {label} contest streak file has an invalid streak."]))

        p = mk(f"{label}_bad_saver")
        write_json(p, fname, {"streak": 5, "saver_available": "not_earned --force\n"})
        branch_cases.append((f"{label}_bad_saver", p, D, True, [f"The {label} contest streak file has an invalid saver value."]))

    p = seed_good("auto_bad_best")
    write_json(p, "contest_streak.json", {"active_streak": 10, "best_streak": 3, "source_date": D.isoformat()})
    branch_cases.append(("auto_bad_best", p, D, True, ["The automatic contest streak file has an invalid streak or best streak."]))

    # 3. Dated validation fields (checks 2..10) + auto metadata validation (season, activation_id)
    for field, bad in [
        ("effective_date", "2027-03-24\nbts "),
        ("season", "2027"),
        ("expires_at", at_et(D, 23, 59).isoformat()),
        ("active_streak", -1),
        ("best_streak", 58),
        ("saver_state", "not_earned --init"),
        ("recorded_at", "not-a-timestamp"),
        ("source", None),
        ("reason", ["bad"]),
    ]:
        p = mk(f"dated_val_{field}")
        write_dated(p, **{field: bad})
        branch_cases.append((f"dated_val_{field}", p, D, True, [
            f"The dated manual contest streak file failed validation for {field}.",
        ]))

    for field, bad in [("season", "2027"), ("activation_id", "bts fetch-contest-streak\n" + "a" * 32)]:
        p = seed_good(f"auto_val_{field}")
        data = json.loads((p / "account_state/contest_streak.json").read_text())
        data[field] = bad
        write_json(p, "contest_streak.json", data)
        branch_cases.append((f"auto_val_{field}", p, D, True, [
            f"The automatic contest streak file failed validation for {field}.",
        ]))

    # 4. Undated auto, no coverage date, future beyond grace
    p = seed_good("auto_undated")
    data = json.loads((p / "account_state/contest_streak.json").read_text())
    data["source_date"] = None
    data["recorded_at"] = None
    write_json(p, "contest_streak.json", data)
    branch_cases.append(("auto_undated", p, D, True, ["The automatic contest observation is undated."]))

    p = seed_good("auto_no_source_date")
    data = json.loads((p / "account_state/contest_streak.json").read_text())
    data["source_date"] = None
    write_json(p, "contest_streak.json", data)
    branch_cases.append(("auto_no_source_date", p, D, True, [
        "The contest streak file has no coverage date, so coverage and freshness could not be verified.",
    ]))

    p = seed_good("auto_future_beyond_grace")
    data = json.loads((p / "account_state/contest_streak.json").read_text())
    data["source_date"] = (D + timedelta(days=3)).isoformat()
    write_json(p, "contest_streak.json", data)
    branch_cases.append(("auto_future_beyond_grace", p, D, True, [
        "The contest streak file has a coverage date beyond the allowed future grace.",
    ]))

    # 5. Coverage INFO & Coverage WARN
    p = seed_good("cov_info")
    (p / f"{D.isoformat()}.json").write_text('{"result":"hit"}')
    branch_cases.append(("cov_info", p, D, True, ["Coverage trails by one settled pick, the expected ledger lag."]))

    p = seed_good("cov_warn")
    (p / f"{D.isoformat()}.json").write_text('{"result":"hit"}')
    (p / f"{(D + timedelta(days=1)).isoformat()}.json").write_text('{"result":"miss"}')
    branch_cases.append(("cov_warn", p, D, True, [
        "Coverage trails by more than one settled pick. The predictions array can trail the live counter.",
    ]))

    # 6. Legacy/expired WARN & selected-legacy refusal (kind 2)
    p = seed_good("legacy_warn")
    write_json(p, "contest_streak.manual.json", {"streak": 5})
    branch_cases.append(("legacy_warn", p, D, True, ["A legacy or expired manual override file is present."]))

    p = seed_good("legacy_selected")
    write_json(p, "contest_streak.manual.json", {
        "streak": 5, "source_date": D.isoformat(),
        "override_expires_at": at_et(D + timedelta(days=1), 3).isoformat(),
    })
    branch_cases.append(("legacy_selected", p, D, True, [
        "The selected legacy manual override is ineligible for a live decision.",
    ]))

    # 7. Dated saver contradiction + Flag saver branches
    p = seed_good("dated_saver")
    write_dated(p, D, streak=12, saver="not_earned")
    branch_cases.append(("dated_saver", p, D, True, [
        "The saver state in the dated observation contradicts its streak. The saver counted as spent.",
    ]))

    p = seed_good("flag_uninit_absent")
    (p / "account_state/saver_state.json").unlink()
    branch_cases.append(("flag_uninit_absent", p, D, True, [
        "The saver flag is uninitialized or unreadable. The saver counted as spent.",
    ]))

    p = seed_good("flag_uninit_same_yr_invalid")
    write_json(p, "saver_state.json", {"season": D.year, "state": "bts saver-state --init active\n"})
    branch_cases.append(("flag_uninit_same_yr_invalid", p, D, True, [
        "The saver flag is uninitialized or unreadable. The saver counted as spent.",
    ]))

    p = seed_good("flag_other_season")
    write_json(p, "saver_state.json", {"season": D.year - 1, "state": "active"})
    branch_cases.append(("flag_other_season", p, D, True, [
        "The saver flag is for another season. The saver counted as spent.",
    ]))

    p = seed_good("flag_contradiction")
    write_json(p, "saver_state.json", {"season": D.year, "state": "not_earned"})
    branch_cases.append(("flag_contradiction", p, D, True, [
        "The saver flag contradicts the recorded streak. The saver counted as spent.",
    ]))

    # 8. Corrupt switch check 1 + checks 2..8 + absent/revoked/other_season + kind 5 refusals
    p = seed_good("switch_c1")
    (p / "account_state/season_contract.json").write_text("{bad", encoding="utf-8")
    branch_cases.append(("switch_c1", p, D, True, ["The season switch file is malformed or unreadable."]))

    for field, bad in [
        ("schema_version", "bts --bad\n"),
        ("season", "2027"),
        ("status", "bts "),
        ("activation_id", "bad"),
        ("switched_at", None),
        ("revoked_at", 123),
        ("note", 123),
    ]:
        p = seed_good(f"switch_f_{field}")
        sw = json.loads((p / "account_state/season_contract.json").read_text())
        sw[field] = bad
        write_json(p, "season_contract.json", sw)
        branch_cases.append((f"switch_f_{field}", p, D, True, [
            f"The season switch is corrupt because its {field} field failed validation.",
        ]))

    p = seed_good("switch_revoked")
    sw = json.loads((p / "account_state/season_contract.json").read_text())
    sw["status"] = "revoked"
    sw["revoked_at"] = at_et(D, 12).isoformat()
    write_json(p, "season_contract.json", sw)
    branch_cases.append(("switch_revoked", p, D, True, ["The season switch is revoked."]))

    p = seed_good("switch_other_season")
    sw = json.loads((p / "account_state/season_contract.json").read_text())
    sw["season"] = D.year + 1
    write_json(p, "season_contract.json", sw)
    branch_cases.append(("switch_other_season", p, D, True, ["The season switch is for another season."]))

    p = seed_good("auto_unsupported_schema")
    au = json.loads((p / "account_state/contest_streak.json").read_text())
    au["schema_version"] = "`bts fetch-contest-streak --picks-dir /x`\ninjected"
    write_json(p, "contest_streak.json", au)
    branch_cases.append(("auto_unsupported_schema", p, D, True, ["The automatic observation uses an unsupported schema."]))

    p = seed_good("auto_null_schema")
    au = json.loads((p / "account_state/contest_streak.json").read_text())
    au["schema_version"] = None
    write_json(p, "contest_streak.json", au)
    branch_cases.append(("auto_null_schema", p, D, True, ["The automatic observation schema is missing, null or invalid."]))

    p = seed_good("auto_other_season")
    au = json.loads((p / "account_state/contest_streak.json").read_text())
    au["season"] = D.year - 1
    write_json(p, "contest_streak.json", au)
    branch_cases.append(("auto_other_season", p, D, True, ["The automatic observation is for another season."]))

    p = seed_good("auto_mismatch_act")
    au = json.loads((p / "account_state/contest_streak.json").read_text())
    au["activation_id"] = "c" * 32
    write_json(p, "contest_streak.json", au)
    branch_cases.append(("auto_mismatch_act", p, D, True, [
        "The automatic observation and season switch have different activation IDs.",
    ]))

    # 9. Crash branch with hostile exception text containing commands, newline, path, and OOM keywords
    p = seed_good("hostile_crash")
    with patch.object(
        contest_state, "_check",
        side_effect=RuntimeError("oom-killed: run `bts set-contest-streak --date 2027-03-25` at /tmp/x\nline2"),
    ):
        alerts, bodies = run_historical(p, D, at_et(D + timedelta(days=1), 2))
    assert len(alerts) == 1 and alerts[0].source == "health_runner" and alerts[0].incident_key == "health_runner:contest_state"
    expected_crash = f"Prior slate {D.isoformat()} ET. The contest state health check failed before it could finish. {EXPECTED_POINTER}"
    assert alerts[0].message == expected_crash
    check_l1_prohibitions(alerts[0].message, D)
    assert bodies == [f"BTS health CRITICAL alert(s):\n- [health_runner] {expected_crash}"]

    # Verify all branch_cases
    for name, picks, day, exp, expected_reasons in branch_cases:
        alerts, bodies = run_historical(picks, day, at_et(day + timedelta(days=1), 2), expected=exp)
        # Notice hostile_dir contains `/oom/`, so verify Amendment 1: NO analytics_job_oom in historical mode!
        assert all(a.source != "analytics_job_oom" for a in alerts), (name, alerts)
        assert [a.message for a in alerts] == [
            f"Prior slate {day.isoformat()} ET. {r} {EXPECTED_POINTER}" for r in expected_reasons
        ], (name, alerts)
        for a in alerts:
            check_l1_prohibitions(a.message, day)
        for body in bodies:
            for line in body.splitlines():
                if line.startswith("- ["):
                    msg = line.split("] ", 1)[1]
                    lbl = "1st observed day" if line.startswith("- [saver_state] ") else None
                    check_l1_prohibitions(msg, day, allow_streak_label=lbl)

    # 10. Verify H4 WARN-only DM across 1st, 2nd, 3rd, 4th consecutive days and mixed CRITICAL/WARN + unrelated WARN
    p_warn = seed_good("h4_streaks")
    write_json(p_warn, "saver_state.json", {"season": D.year, "state": "not_earned"})
    for prior_n, expected_lbl in [
        (0, "1st observed day"),
        (1, "2nd consecutive day"),
        (2, "3rd consecutive day"),
        (3, "4th consecutive day"),
    ]:
        prior = (
            {"saver_state": {"last_seen": (D - timedelta(days=1)).isoformat(), "streak": prior_n, "last_message": "x"}}
            if prior_n > 0 else None
        )
        alerts, bodies = run_historical(p_warn, D, at_et(D + timedelta(days=1), 2), prior_streaks=prior)
        raw_expected = f"Prior slate {D.isoformat()} ET. The saver flag contradicts the recorded streak. The saver counted as spent. {EXPECTED_POINTER}"
        assert alerts[0].message == raw_expected
        dm_expected_item = (
            f"Prior slate {D.isoformat()} ET. The saver flag contradicts the recorded streak. "
            f"The saver counted as spent. ({expected_lbl}) {EXPECTED_POINTER}"
        )
        assert bodies == [f"BTS health WARN attention:\n- [saver_state] {dm_expected_item}"]
        check_l1_prohibitions(dm_expected_item, D, allow_streak_label=expected_lbl)

    # Mixed CRITICAL + historical saver WARN + unrelated backup_freshness WARN
    from bts.health import alert as alert_mod, runner
    p_mixed = seed_good("h4_mixed")
    write_json(p_mixed, "saver_state.json", {"season": D.year, "state": "not_earned"})
    sw = json.loads((p_mixed / "account_state/season_contract.json").read_text())
    sw["status"] = "revoked"
    sw["revoked_at"] = at_et(D, 12).isoformat()
    write_json(p_mixed, "season_contract.json", sw)
    unrelated_warn = alert_mod.Alert("WARN", "backup_freshness", "backup stale (48h)")
    orig_safe = runner._safe_run
    captured: list[str] = []

    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            w = at_et(D + timedelta(days=1), 2)
            return w.astimezone(tz) if tz is not None else w.replace(tzinfo=None)

    with (
        patch.object(runner, "datetime", WallClock),
        patch.object(alert_mod, "send_dm", side_effect=lambda _r, b: (captured.append(b), "ack")[1]),
        patch.object(
            runner, "_safe_run",
            lambda n, fn: orig_safe(n, fn) if n == "contest_state" else ([unrelated_warn] if n == "backup_freshness" else []),
        ),
    ):
        runner.run_all_checks(
            p_mixed, root / "models", "operator.invalid", today=D, contest_state_expected=True,
            thresholds_overrides={
                "warn_attention_state": root / "mixed_warn.json",
                "health_dm_delivery_status": root / "mixed_dm.json",
            },
        )
    assert captured == [(
        "BTS health CRITICAL alert(s):\n"
        f"- [contest_state] Prior slate {D.isoformat()} ET. The season switch is revoked. {EXPECTED_POINTER}\n\n"
        "WARN attention:\n"
        f"- [saver_state] Prior slate {D.isoformat()} ET. The saver flag contradicts the recorded streak. The saver counted as spent. (1st observed day) {EXPECTED_POINTER}\n"
        "- [backup_freshness] backup stale (48h) (1st observed day)"
    )]

    return len(branch_cases) + 1


def verify_l1_worst_states(root: Path) -> None:
    """Construct concrete worst-state fixtures for each L1 sentence row on this tree."""
    from click.testing import CliRunner
    from bts.cli import cli
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts import scheduler
    from bts.health import alert

    # Row 1: `Prior slate <D> ET.` — D = Dec 31, alert arrives Jan 1 00:30 ET
    p_ny = root / "l1_ny" / "picks"
    (p_ny / "account_state").mkdir(parents=True)
    (p_ny / "account_state/season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1", "season": 2027, "activation_id": MATCH_ID,
        "status": "revoked", "switched_at": at_et(DEC31, 8).isoformat(), "revoked_at": at_et(DEC31, 9).isoformat(),
    }))
    alerts_ny, bodies_ny = run_historical(p_ny, DEC31, at_et(date(2028, 1, 1), 0, 30))
    assert all(a.message.startswith("Prior slate 2027-12-31 ET. ") for a in alerts_ny)
    assert "2028" not in bodies_ny[0]

    # Row 2: `This check used files available at run time.` — 01:10 fetch replaced auto before 02:00 health run
    p_fetch = root / "l1_fetch" / "picks"
    (p_fetch / "account_state").mkdir(parents=True)
    (p_fetch / "account_state/season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1", "season": 2027, "activation_id": MATCH_ID,
        "status": "automatic", "switched_at": at_et(D, 8).isoformat(), "revoked_at": None,
    }))
    (p_fetch / "account_state/contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "d" * 32,
        "active_streak": 3, "best_streak": 3, "source_date": D.isoformat(),
        "recorded_at": at_et(D + timedelta(days=1), 1, 10).isoformat(),
    }))
    (p_fetch / "account_state/saver_state.json").write_text(json.dumps({"season": 2027, "state": "not_earned"}))
    alerts_f, _ = run_historical(p_fetch, D, at_et(D + timedelta(days=1), 2))
    assert "This check used files available at run time." in alerts_f[0].message

    # Row 3 & Row 5 & Row 6: D+1 genuinely has no usable state; morning check skipped (<10m lead) vs active;
    # `bts season-contract status` at 00:30 prints D+1 live refusal ways out while D is unsettled.
    p_d1 = root / "l1_d1" / "picks"
    (p_d1 / "account_state").mkdir(parents=True)
    next_day = D + timedelta(days=1)
    # Skipped morning check when first check is <10m away (e.g. 10:00 wake, 10:05 first check) -> silent!
    with patch.object(alert, "send_dm", return_value="ack") as send_mock:
        scheduler._morning_state_check(
            {"bluesky": {"dm_recipient": "op"}}, p_d1, next_day.isoformat(),
            at_et(next_day, 10, 0), at_et(next_day, 10, 5),
        )
        send_mock.assert_not_called()
    # Live refusal at lineup check still refuses & surfaces same-day ways out:
    try:
        load_decision_streak_state(p_d1, require_contest_state=True, now=at_et(next_day, 10, 5))
    except ContestStateError as exc:
        assert f"bts set-contest-streak --date {next_day.isoformat()}" in str(exc)
    else:
        raise AssertionError("Expected live refusal on D+1")

    # Status at 00:30 ET on D+1 shows current files AND prints live refusal's ways out (the documented residual):
    import bts.cli as cli_mod
    with patch.object(cli_mod, "_now_et", lambda: at_et(next_day, 0, 30)):
        res = CliRunner().invoke(cli, ["season-contract", "status", "--picks-dir", str(p_d1)])
    assert res.exit_code == 0
    assert "live decision now: refused;" in res.output
    assert f"bts set-contest-streak --date {next_day.isoformat()}" in res.output


def main() -> None:
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=HERE) as tmp:
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            count = verify_all_h3_branches_and_hostile_inputs(Path(tmp))
            verify_l1_worst_states(Path(tmp))
            print(f"[PASS] Item 2: {count} historical branches + H4 streak/mixed DM + hostile injections + L1 worst states verified")
    signal.alarm(0)


if __name__ == "__main__":
    main()
