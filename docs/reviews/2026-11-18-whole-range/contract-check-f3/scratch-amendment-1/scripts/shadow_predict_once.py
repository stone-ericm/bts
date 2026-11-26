#!/usr/bin/env python3
"""Run one BTS shadow prediction outside the scheduler process."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = Path("/home/bts/.bts-orchestrator.toml")
DEFAULT_UNIT = "bts-shadow-prediction.service"


def today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=today_et())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    return parser


def run_once(config_path: Path, date: str, unit: str | None) -> int:
    from bts.orchestrator import load_config
    from bts.picks import load_pick, pick_was_delivered
    from bts.scheduler import (
        _run_shadow_prediction,
        _update_analytics_job_status,
        load_state,
    )

    config = load_config(config_path)
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    daily = load_pick(date, picks_dir)
    if daily is None:
        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "failed",
            reason="production_pick_missing",
            unit=unit,
        )
        print(f"[SHADOW MODEL] No production pick found for {date}.", file=sys.stderr)
        return 1

    state = load_state(date, picks_dir)
    if not pick_was_delivered(daily) and not (state and state.pick_locked):
        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "failed",
            reason="production_pick_not_locked",
            unit=unit,
        )
        print(f"[SHADOW MODEL] Production pick is not locked for {date}.",
              file=sys.stderr)
        return 1

    _run_shadow_prediction(
        config,
        date,
        daily.pick.batter_name,
        allow_prior_dispatched=True,
        attempt_reason="shadow_unit_attempt",
        unit=unit,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_once(args.config, args.date, args.unit or None)
    except Exception as exc:
        print(f"[SHADOW MODEL] Unhandled failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
