"""Offline fixtures for the F2 round-two probes."""

import datetime as dt
import json
import logging
import os
import signal
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas  # Import before pinning datetime for the CLI and scheduler.
from click.testing import CliRunner

from bts import cli, contest_state, saver_state, season_contract
from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
DATETIME = dt.datetime
DAY = DATETIME(2027, 7, 15, 22, 15, tzinfo=ET)
DECEMBER = DATETIME(2027, 12, 31, 23, 48, tzinfo=ET)
NEXT = (DAY + dt.timedelta(days=1)).replace(hour=0, minute=12)
INCIDENT = "saver_state:dated_used_mismatch"
DIGEST = "health_dm_delivery_status.json"
SUFFIX = ".dated_used_mismatch.json"
MISMATCH = DIGEST + SUFFIX
RECIPIENT = "test.invalid"
ACTIVATION = "a" * 32

logging.disable(logging.CRITICAL)


@contextmanager
def clock(now):
    class Pinned(DATETIME):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    with ExitStack() as stack:
        for module in (dt, cli, contest_state, saver_state, health, runner, alert):
            stack.enter_context(patch.object(module, "datetime", Pinned))
        yield


@contextmanager
def deadline(seconds=2):
    class ReadDeadline(BaseException):
        pass

    def expired(*_):
        raise ReadDeadline("read exceeded its deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def cli_call(picks, now, *args):
    with clock(now):
        result = CliRunner().invoke(cli.cli, [*args, "--picks-dir", str(picks)], catch_exceptions=False)
    assert result.exit_code == 0, (args, result.output)
    return result.output


def observe(picks, now=DAY, *, saver="used", streak=14, best=None):
    return write(picks / "account_state/contest_streak.manual.json", {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": str(now.date()),
        "season": now.year,
        "expires_at": DATETIME.combine(now.date(), dt.time.max, ET).isoformat(),
        "active_streak": streak,
        "best_streak": max(streak, 16) if best is None else best,
        "saver_state": saver,
        "recorded_at": now.astimezone(dt.timezone.utc).isoformat(),
        "source": "cli_dated",
    })


def flag(picks, state, year=DAY.year):
    return write(picks / "account_state/saver_state.json", {"season": year, "state": state})


def automatic(picks, when, *, streak=14, coverage=None):
    account = picks / "account_state"
    write(account / "season_contract.json", {
        "schema_version": "bts_season_contract_v1", "season": when.year,
        "status": "automatic", "activation_id": ACTIVATION,
        "switched_at": (when - dt.timedelta(hours=2)).isoformat(),
        "revoked_at": None, "note": None,
    })
    return write(account / "contest_streak.json", {
        "schema_version": season_contract.AUTO_SCHEMA, "season": when.year,
        "activation_id": ACTIVATION, "active_streak": streak, "best_streak": max(streak, 16),
        "source": "mlb_bts_profile", "source_date": str(coverage or when.date()),
        "recorded_at": when.isoformat(),
    })


def receipt(picks, name=DIGEST):
    return picks.parent / "health_state" / name


def files(root):
    return {str(path.relative_to(root)): path.read_bytes().hex()
            for path in sorted(root.rglob("*")) if path.is_file()}


def select(alerts):
    return [row for row in alerts if row.incident_key == INCIDENT]


def eod(picks, day=DAY, *, wall=None, recipient=RECIPIENT, existing=(),
        transport=None, overrides=None, expected=True):
    """Keep _safe_run, attention and dispatch real; replace irrelevant check inputs."""
    sends = []
    def capture(recipient, body):
        sends.append((recipient, body))
        return transport(recipient, body) if transport is not None else "ack"

    inactive_checks = (
        "blend_training", "pick_entry", "post_failure", "fallback_defer", "late_delivery",
        "postponed_pick", "result_resolution", "predicted_vs_realized", "realized_calibration",
        "mdp_policy_alignment", "tail_policy", "same_team_corr", "pitcher_sparsity", "projected_lineup",
        "park_drag_freshness", "disk_fill", "backup_freshness", "unit_drift",
        "scheduler_state_integrity", "streak_validation", "probable_pitcher_cache",
    )
    with ExitStack() as stack:
        stack.enter_context(clock(wall or day))
        stack.enter_context(patch.object(alert, "send_dm", capture))
        stack.enter_context(patch.object(runner.calibration, "check", return_value=list(existing)))
        stack.enter_context(patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None))
        for check in inactive_checks:
            stack.enter_context(patch.object(getattr(runner, check), "check", return_value=[]))
        with deadline():
            alerts = runner.run_all_checks(
                picks, picks.parent / "models", recipient, today=day.date(),
                thresholds_overrides=overrides, contest_state_expected=expected,
            )
    return alerts, sends
