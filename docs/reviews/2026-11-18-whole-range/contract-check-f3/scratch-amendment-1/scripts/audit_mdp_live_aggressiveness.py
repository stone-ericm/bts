#!/usr/bin/env python3
"""Audit recent production picks against the saved MDP action surface.

This is an evidence-only diagnostic. It reads pick JSON files plus a saved MDP
policy artifact and reports how current live probabilities map into policy bins,
what action the policy would take at specified streak states, and which
double-down decisions would be flagged by a simple conservative pair floor.

It does not write policy artifacts, mutate pick files, or claim a deploy path.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Sequence

from bts.simulate.mdp import ACTIONS, load_policy, lookup_action
from bts.strategy import SEASON_END_DATE


DEFAULT_STREAK_STATES = (0, 4, 8, 10, 15)
DEFAULT_Q0_DOUBLE_FLOOR = 0.55
_DATE_JSON_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


@dataclass(frozen=True)
class LivePickRow:
    path: str
    date: str
    primary_name: str
    primary_p: float
    primary_projected: bool
    double_name: str | None
    double_p: float | None
    double_projected: bool | None
    p_both: float | None
    result: str | None
    slot_results: dict | None


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _classify(p_game_hit: float, boundaries: Sequence[float]) -> int:
    q = 0
    for boundary in boundaries:
        if p_game_hit >= boundary:
            q += 1
    return q


def _pick_paths(picks_dir: Path) -> list[Path]:
    """Return root-level production pick files only, excluding shadow/archive JSON."""
    return [
        p for p in sorted(picks_dir.glob("*.json"))
        if _DATE_JSON_RE.match(p.name)
    ]


def load_live_pick_rows(
    picks_dir: Path,
    *,
    today: date | None = None,
) -> list[LivePickRow]:
    rows: list[LivePickRow] = []
    for path in _pick_paths(picks_dir):
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pick_date = _parse_date(body.get("date") or path.stem)
        if today is not None and pick_date > today:
            continue

        primary = body.get("pick") or {}
        dd = body.get("double_down")
        primary_p = primary.get("p_game_hit")
        if primary_p is None:
            continue

        dd_p = None
        p_both = None
        if dd and dd.get("p_game_hit") is not None:
            dd_p = float(dd["p_game_hit"])
            p_both = float(primary_p) * dd_p

        rows.append(LivePickRow(
            path=str(path),
            date=pick_date.isoformat(),
            primary_name=str(primary.get("batter_name") or ""),
            primary_p=float(primary_p),
            primary_projected=bool(primary.get("projected_lineup", False)),
            double_name=str(dd.get("batter_name")) if dd else None,
            double_p=dd_p,
            double_projected=bool(dd.get("projected_lineup", False)) if dd else None,
            p_both=p_both,
            result=body.get("result"),
            slot_results=body.get("slot_results"),
        ))
    return rows


def _days_remaining(pick_date: str, season_length: int) -> int:
    end = _parse_date(SEASON_END_DATE)
    current = _parse_date(pick_date)
    return max(0, min((end - current).days, season_length))


def _action_for_row(
    row: LivePickRow,
    *,
    policy_table,
    boundaries: Sequence[float],
    season_length: int,
    streak: int,
    saver_available: bool,
) -> str:
    return lookup_action(
        policy_table,
        list(boundaries),
        streak,
        _days_remaining(row.date, season_length),
        saver_available,
        row.primary_p,
        season_length,
    )


def run_audit(
    *,
    picks_dir: Path,
    policy_path: Path,
    today: date | None = None,
    recent: int | None = None,
    streak_states: Sequence[int] = DEFAULT_STREAK_STATES,
    saver_available: bool = True,
    q0_double_floor: float = DEFAULT_Q0_DOUBLE_FLOOR,
) -> dict:
    policy_table, boundaries, season_length = load_policy(policy_path)
    rows = load_live_pick_rows(picks_dir, today=today)
    if recent is not None:
        rows = rows[-recent:]

    row_dicts = []
    qbin_counts: Counter[int] = Counter()
    action_counts_by_streak: dict[str, Counter[str]] = {
        str(streak): Counter() for streak in streak_states
    }
    guardrail_candidates: list[dict] = []

    for row in rows:
        qbin = _classify(row.primary_p, boundaries)
        qbin_counts[qbin] += 1
        actions = {}
        for streak in streak_states:
            action = _action_for_row(
                row,
                policy_table=policy_table,
                boundaries=boundaries,
                season_length=season_length,
                streak=streak,
                saver_available=saver_available,
            )
            actions[str(streak)] = action
            action_counts_by_streak[str(streak)][action] += 1

        flagged_streaks = [
            str(streak)
            for streak in streak_states
            if (
                qbin == 0
                and row.p_both is not None
                and row.p_both < q0_double_floor
                and actions[str(streak)] == "double"
            )
        ]
        entry = {
            **asdict(row),
            "quality_bin": qbin,
            "actions_by_streak": actions,
            "q0_double_floor_flagged_streaks": flagged_streaks,
        }
        row_dicts.append(entry)
        if flagged_streaks:
            guardrail_candidates.append(entry)

    primary_values = [row.primary_p for row in rows]
    dd_values = [row.double_p for row in rows if row.double_p is not None]
    p_both_values = [row.p_both for row in rows if row.p_both is not None]

    return {
        "production_deploy_claim": False,
        "writes_policy_artifact": False,
        "methodology": {
            "picks_dir": str(picks_dir),
            "policy_path": str(policy_path),
            "today": today.isoformat() if today else None,
            "recent": recent,
            "streak_states": list(streak_states),
            "saver_available": saver_available,
            "q0_double_floor": q0_double_floor,
            "caveat": (
                "Pick JSON does not currently persist the exact pre-decision "
                "streak/saver/action state, so action lookup is evaluated over "
                "explicit supplied streak states rather than asserted as the "
                "historical production state."
            ),
        },
        "policy": {
            "boundaries": [float(x) for x in boundaries],
            "season_length": int(season_length),
        },
        "summary": {
            "n": len(rows),
            "date_min": min((row.date for row in rows), default=None),
            "date_max": max((row.date for row in rows), default=None),
            "primary_mean": mean(primary_values) if primary_values else None,
            "primary_min": min(primary_values) if primary_values else None,
            "primary_max": max(primary_values) if primary_values else None,
            "double_mean": mean(dd_values) if dd_values else None,
            "p_both_mean": mean(p_both_values) if p_both_values else None,
            "qbin_counts": {str(k): int(v) for k, v in sorted(qbin_counts.items())},
            "action_counts_by_streak": {
                streak: dict(counts)
                for streak, counts in action_counts_by_streak.items()
            },
            "q0_double_floor_candidate_count": len(guardrail_candidates),
        },
        "guardrail_candidates": guardrail_candidates,
        "rows": row_dicts,
    }


def _parse_streaks(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            values.append(int(stripped))
    if not values:
        raise argparse.ArgumentTypeError("at least one streak state is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--picks-dir", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, default=Path("data/models/mdp_policy.npz"))
    parser.add_argument("--today", type=_parse_date)
    parser.add_argument("--recent", type=int)
    parser.add_argument("--streak-states", type=_parse_streaks, default=list(DEFAULT_STREAK_STATES))
    parser.add_argument("--no-saver", action="store_true")
    parser.add_argument("--q0-double-floor", type=float, default=DEFAULT_Q0_DOUBLE_FLOOR)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = run_audit(
        picks_dir=args.picks_dir,
        policy_path=args.policy_path,
        today=args.today,
        recent=args.recent,
        streak_states=args.streak_states,
        saver_available=not args.no_saver,
        q0_double_floor=args.q0_double_floor,
    )
    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
