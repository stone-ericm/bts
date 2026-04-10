#!/usr/bin/env python3
"""Phase 2: Diff Fly shadow output against Pi5 real pick output.

Usage:
    python3 scripts/cutover/phase2-shadow-diff.py --date 2026-04-15

Exit codes:
    0 = match
    1 = mismatch (details in output)
    2 = error (couldn't load one of the sources)
"""
import argparse
import json
import subprocess
import sys

FLOAT_TOLERANCE = 1e-6


def load_fly_shadow(date: str) -> dict | None:
    """Download Fly shadow output via flyctl ssh."""
    result = subprocess.run(
        ["flyctl", "ssh", "console", "-a", "bts-mlb", "-C",
         f"cat /data/shadow/{date}/fly.json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Could not read Fly shadow for {date}: {result.stderr}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Fly shadow for {date} is not valid JSON", file=sys.stderr)
        return None


def load_pi5_real(date: str) -> dict | None:
    """Read Pi5 real pick state via SSH."""
    result = subprocess.run(
        ["ssh", "stonehengee@pi5.local",
         f"cat ~/projects/bts/data/picks/{date}.json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Could not read Pi5 pick for {date}: {result.stderr}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Pi5 pick for {date} is not valid JSON", file=sys.stderr)
        return None


def _get_nested(obj: dict, path: list[str]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def compare_picks(fly: dict, pi5: dict) -> list[str]:
    """Return a list of mismatch descriptions. Empty list = strict match."""
    issues: list[str] = []

    for field in ["batter_id", "pitcher_id", "game_pk", "team", "batter_name", "pitcher_name"]:
        fly_val = _get_nested(fly, ["pick", field])
        pi5_val = _get_nested(pi5, ["pick", field])
        if fly_val != pi5_val:
            issues.append(f"pick.{field}: fly={fly_val!r} pi5={pi5_val!r}")

    for field in ["p_game_hit"]:
        fly_val = _get_nested(fly, ["pick", field])
        pi5_val = _get_nested(pi5, ["pick", field])
        if fly_val is None and pi5_val is None:
            continue
        if fly_val is None or pi5_val is None:
            issues.append(f"pick.{field}: fly={fly_val} pi5={pi5_val}")
            continue
        if abs(float(fly_val) - float(pi5_val)) > FLOAT_TOLERANCE:
            issues.append(f"pick.{field}: |delta|={abs(fly_val - pi5_val):.2e} > tolerance")

    fly_has_dd = fly.get("double_down") is not None
    pi5_has_dd = pi5.get("double_down") is not None
    if fly_has_dd != pi5_has_dd:
        issues.append(f"double_down presence: fly={fly_has_dd} pi5={pi5_has_dd}")
    elif fly_has_dd:
        for field in ["batter_id", "game_pk"]:
            fly_val = _get_nested(fly, ["double_down", field])
            pi5_val = _get_nested(pi5, ["double_down", field])
            if fly_val != pi5_val:
                issues.append(f"double_down.{field}: fly={fly_val!r} pi5={pi5_val!r}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Compare Fly shadow to Pi5 real pick")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    args = parser.parse_args()

    fly = load_fly_shadow(args.date)
    pi5 = load_pi5_real(args.date)

    if fly is None or pi5 is None:
        print(f"ERROR: could not load one or both sides for {args.date}", file=sys.stderr)
        sys.exit(2)

    issues = compare_picks(fly, pi5)
    if not issues:
        print(f"MATCH: Fly == Pi5 for {args.date}")
        sys.exit(0)

    print(f"MISMATCH: {args.date}")
    for issue in issues:
        print(f"  - {issue}")
    sys.exit(1)


if __name__ == "__main__":
    main()
