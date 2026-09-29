"""Contest slot history, the streak chain, lookups and game matching (spec §6)."""
from __future__ import annotations

from collections import Counter

SLOT_VALUE_KEYS = ("slot_result", "slot_result_state", "hits", "hits_state", "at_bats", "at_bats_state",
                   "slot_number", "round_result", "round_streak", "round_streak_increase")


def _order(row: dict) -> tuple:
    return (row["recorded_at"], row["line_no"])     # fixed-precision UTC: string order is time order


def slot_history(contest_rows: list[dict]) -> list[dict]:
    """Per slot identity (round_id, unit_id, player_id): first/last seen, the last qualified values, whether
    values changed, and `dropped_later` when a later qualified line no longer shows it. A newer omission
    never erases the older positive observation; every earlier value stays in the occurrence table."""
    lines = sorted({_order(r) for r in contest_rows if r["row_level"] == "line"})
    groups: dict[tuple, list[dict]] = {}
    for r in sorted((r for r in contest_rows if r["row_level"] == "slot"), key=_order):
        groups.setdefault((r["round_id"], r["unit_id"], r["player_id"]), []).append(r)
    out = []
    for (round_id, unit_id, player_id), obs in sorted(groups.items()):
        last = obs[-1]
        out.append({"round_id": round_id, "unit_id": unit_id, "player_id": player_id,
                    "first_seen": obs[0]["recorded_at"], "last_seen": last["recorded_at"],
                    "last_line_no": last["line_no"], "n_observations": len(obs),
                    "changed": len({tuple(o[k] for k in SLOT_VALUE_KEYS) for o in obs}) > 1,
                    "dropped_later": any(line > _order(last) for line in lines),
                    "last_obs_id": last["obs_id"], **{k: last[k] for k in SLOT_VALUE_KEYS}})
    return out


def line_round_streaks(contest_rows: list[dict]) -> dict[int, dict[int, int | None]]:
    """line_no → {round_id: reported post-round streak} for every round present in that line."""
    out: dict[int, dict[int, int | None]] = {}
    for r in contest_rows:
        if r["row_level"] in ("round", "slot"):
            out.setdefault(r["line_no"], {})[r["round_id"]] = r["round_streak"]
    return out


def entered_rounds(contest_rows: list[dict]) -> list[int]:
    """Every roundId any qualified line reports — the account's entered rounds — sorted."""
    return sorted({r["round_id"] for r in contest_rows if r["row_level"] in ("round", "slot")})


def streak_before(line_streaks: dict[int, int | None], round_id: int, entered: list[int]) -> int | None:
    """Interpretation I2: the previous entered round is the greatest roundId below `round_id` that any
    qualified line reports; its streak counts only when that round is present in this same line."""
    earlier = [r for r in entered if r < round_id]
    return line_streaks.get(earlier[-1]) if earlier else None
