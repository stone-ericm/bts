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


def rounds_lookup(round_rows: list[dict]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for r in round_rows:
        out.setdefault(r["round_id"], set()).add(r["round_date"])
    return out


def players_lookup(player_rows: list[dict]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for p in player_rows:
        if p["feed_id"] is not None:
            out.setdefault(p["player_id"], set()).add(p["feed_id"])
    return out


def units_lookup(unit_rows: list[dict]) -> dict[int, dict]:
    """unit_id → every feedId / roundId any capture recorded; conflicts are kept, never collapsed."""
    out: dict[int, dict] = {}
    for u in unit_rows:
        entry = out.setdefault(u["unit_id"], {"feed_ids": set(), "round_ids": set()})
        if u["feed_id"] is not None:
            entry["feed_ids"].add(u["feed_id"])
        if u["round_id"] is not None:
            entry["round_ids"].add(u["round_id"])
    return out


def unit_status_history(unit_rows: list[dict]) -> dict[int, list[tuple]]:
    """unit_id → [(capture time, status)] across captures, oldest first (Interpretation I6)."""
    out: dict[int, list[tuple]] = {}
    for u in unit_rows:
        out.setdefault(u["unit_id"], []).append((u["captured_at"], u["status"]))
    return {k: sorted(v, key=lambda x: (x[0] or "", str(x[1]))) for k, v in out.items()}


def team_games(schedule_rows: list[dict]) -> dict[tuple[str, str], set[int]]:
    """(query date, team abbreviation) → every listed gamePk, whatever its status (postponed, cancelled and
    suspended entries included), so 'exactly one game' is never produced by filtering."""
    out: dict[tuple[str, str], set[int]] = {}
    for g in schedule_rows:
        for abbr in (g["away_abbr"], g["home_abbr"]):
            out.setdefault((g["query_date"], abbr), set()).add(g["game_pk"])
    return out
