"""Tier 3: streak.json schema validation.

Catches state corruption in the streak file. Per memory, BTS DD scoring
requires both picks to hit; if either fails the streak resets to 0. The
streak.json file is updated end-of-day after result polling. Corruption
patterns:
  - File missing
  - Malformed JSON
  - streak field not int
  - streak field negative
  - saver_available not bool
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "streak_validation"


def check(picks_dir: Path) -> list[Alert]:
    """Returns CRITICAL alert if streak.json is missing, malformed, or schema-invalid."""
    streak_path = picks_dir / "streak.json"
    if not streak_path.exists():
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=f"streak.json missing at {streak_path}",
        )]
    try:
        data = json.loads(streak_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=f"streak.json malformed: {e}",
        )]

    issues = []
    streak = data.get("streak")
    if not isinstance(streak, int):
        issues.append(f"streak field is {type(streak).__name__}, expected int")
    elif streak < 0:
        issues.append(f"streak field is negative: {streak}")

    saver = data.get("saver_available")
    # saver_available is allowed to be missing (not all configs use it),
    # but if present it must be a bool.
    if saver is not None and not isinstance(saver, bool):
        issues.append(f"saver_available is {type(saver).__name__}, expected bool")

    if issues:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message="streak.json schema invalid: " + "; ".join(issues),
        )]
    return []
