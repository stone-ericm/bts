"""Export current BTS state to a committable snapshot file.

Used once, at the moment of cloud migration cutover, to freeze the
pre-migration history into a git-tracked file. After export, the
regenerate command uses this file as the source of truth for dates
before the cutoff and uses Bluesky + MLB API for dates after.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

EXPORT_VERSION = 1
DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


class UnresolvedPickError(RuntimeError):
    """Raised when export is attempted with unresolved picks present."""


def export_initial_state(picks_dir: Path, output_path: Path) -> dict:
    """Export the full BTS state as a committable snapshot.

    Enforces the invariant that no pick may be in an unresolved state.
    Raises UnresolvedPickError with the list of offending files if any
    pick has `result is None`.

    Returns the exported dict (also written to output_path).
    """
    pick_files = _collect_pick_files(picks_dir)

    unresolved = []
    historical: list[dict] = []
    for pf in pick_files:
        data = json.loads(pf.read_text())
        if data.get("result") is None:
            unresolved.append(pf.name)
            continue
        historical.append(_pick_to_historical(data))

    if unresolved:
        raise UnresolvedPickError(
            f"Refusing to export: {len(unresolved)} pick(s) still unresolved: "
            f"{', '.join(sorted(unresolved)[:5])}"
            f"{'...' if len(unresolved) > 5 else ''}. "
            f"Wait for results to finalize and try again."
        )

    streak_file = picks_dir / "streak.json"
    if not streak_file.exists():
        raise RuntimeError(
            f"streak.json not found in {picks_dir}. "
            f"Export requires a streak file to determine the starting point."
        )
    streak_data = json.loads(streak_file.read_text())

    historical.sort(key=lambda p: p["date"])
    cutoff_date = historical[-1]["date"] if historical else "none"

    snapshot = {
        "version": EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": cutoff_date,
        "streak_at_cutoff": streak_data.get("streak", 0),
        "saver_available": streak_data.get("saver_available", True),
        "historical_picks": historical,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))
    return snapshot


def _collect_pick_files(picks_dir: Path) -> list[Path]:
    """Return only files whose name matches YYYY-MM-DD.json."""
    return [
        p for p in picks_dir.iterdir()
        if p.is_file() and DATE_PATTERN.match(p.name)
    ]


def _pick_to_historical(pick_data: dict) -> dict:
    """Project a full pick file into its committable historical form."""
    return {
        "date": pick_data["date"],
        "pick": _project_pick(pick_data.get("pick")),
        "double_down": _project_pick(pick_data.get("double_down")) if pick_data.get("double_down") else None,
        "result": pick_data.get("result"),
        "bluesky_posted": pick_data.get("bluesky_posted", False),
        "bluesky_uri": pick_data.get("bluesky_uri"),
    }


def _project_pick(pick: dict | None) -> dict | None:
    """Extract all Pick fields so the result is loadable by load_pick()."""
    if pick is None:
        return None
    return {
        "batter_name": pick["batter_name"],
        "batter_id": pick["batter_id"],
        "team": pick["team"],
        "lineup_position": pick.get("lineup_position", 0),
        "pitcher_name": pick["pitcher_name"],
        "pitcher_id": pick.get("pitcher_id"),
        "p_game_hit": pick["p_game_hit"],
        "flags": pick.get("flags", []),
        "projected_lineup": pick.get("projected_lineup", False),
        "game_pk": pick["game_pk"],
        "game_time": pick["game_time"],
        "pitcher_team": pick.get("pitcher_team"),
    }
