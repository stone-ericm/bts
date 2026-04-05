"""Live scorecard data extraction from MLB game feed."""

from bts.data.schema import HIT_EVENTS

_RESULT_MAP = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "walk": "BB",
    "hit_by_pitch": "HBP",
    "sac_fly": "SF",
    "sac_bunt": "SAC",
    "double_play": "DP",
    "grounded_into_double_play": "GDP",
    "force_out": "FC",
    "intent_walk": "IBB",
    "catcher_interf": "CI",
}

_TRAJECTORY_PREFIX = {
    "fly_ball": "F",
    "ground_ball": "G",
    "line_drive": "L",
    "popup": "P",
}


def format_result_code(
    event: str,
    event_type: str,
    last_pitch_code: str | None,
    trajectory: str | None,
    fielder_position: int | None,
) -> str:
    """Convert MLB API event data to traditional scorecard shorthand."""
    if event_type == "strikeout":
        return "\u042f" if last_pitch_code == "C" else "K"

    if event_type == "field_out":
        prefix = _TRAJECTORY_PREFIX.get(trajectory, "F")
        return f"{prefix}{fielder_position}" if fielder_position else f"{prefix}?"

    if event_type == "field_error":
        return f"E{fielder_position}" if fielder_position else "E"

    return _RESULT_MAP.get(event_type, event_type.upper()[:3] if event_type else "?")
