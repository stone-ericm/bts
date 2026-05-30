"""Contest-account state validation for live recommendation safety."""

from __future__ import annotations

from pathlib import Path

from bts.contest_state import ContestStateError, load_contest_streak_state
from bts.health.alert import Alert

SOURCE = "contest_state"


def check(picks_dir: Path, *, expected: bool = False) -> list[Alert]:
    """Return CRITICAL when required contest-account state is missing or invalid."""
    try:
        state = load_contest_streak_state(picks_dir)
    except ContestStateError as exc:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=str(exc),
        )]

    if state is None and expected:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                "contest-account streak state expected but missing at "
                f"{picks_dir / 'account_state'}"
            ),
        )]

    return []
