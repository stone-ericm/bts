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


def _result(slot: dict, *, date=None, batter_id=None, game_pk=None, selection_id=None, match: str, reason: str) -> dict:
    return {"round_id": slot["round_id"], "unit_id": slot["unit_id"], "player_id": slot["player_id"],
            "date": date, "batter_id": batter_id, "game_pk": game_pk, "selection_id": selection_id,
            "match": match, "match_reason": reason}


def match_slot(slot: dict, *, rounds: dict, players: dict, units: dict, team_games: dict,
               schedule_status: dict[str, str], local_selections: list[dict]) -> dict:
    """Spec §6. Never matches on the slot `number`. Only a unit with no capture at all may use the
    pick-time-team inference, and only on a date whose schedule is complete; unit evidence that is not one
    round-consistent feedId is ambiguous and transfers nothing. Only `evidenced` / `inferred` links carry a
    selection_id."""
    dates = rounds.get(slot["round_id"], set())
    if len(dates) != 1:
        return _result(slot, match="unmapped", reason="round_date_unknown" if not dates else "round_date_conflict")
    date = next(iter(dates))
    feeds = players.get(slot["player_id"], set())
    if len(feeds) != 1:
        return _result(slot, date=date, match="unmapped", reason="player_unknown" if not feeds else "player_conflict")
    batter = next(iter(feeds))
    candidates = [s for s in local_selections if s["date"] == date and s["batter_id"] == batter]
    unit = units.get(slot["unit_id"])
    if unit is not None:
        if len(unit["feed_ids"]) != 1:
            reason = "unit_capture_without_feed_id" if not unit["feed_ids"] else "conflicting_unit_evidence"
            return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)
        if unit["round_ids"] and unit["round_ids"] != {slot["round_id"]}:
            return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="unit_round_contradiction")
        game = next(iter(unit["feed_ids"]))
        same = [s for s in candidates if s["game_pk"] == game]
        if len(same) == 1:
            reason = "unit_capture"
        elif same:
            reason = "unit_capture_multiple_local"
        else:
            reason = "unit_capture_other_game" if candidates else "unit_capture_no_local_selection"
        return _result(slot, date=date, batter_id=batter, game_pk=game,
                       selection_id=same[0]["selection_id"] if len(same) == 1 else None,
                       match="evidenced", reason=reason)
    if not candidates:
        return _result(slot, date=date, batter_id=batter, match="unmapped", reason="no_unit_capture_contest_only")
    if len(candidates) > 1:
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="multiple_local_selections")
    sel = candidates[0]
    if sel["game_pk"] is None:
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="selection_game_pk_unrecorded")
    status = schedule_status.get(date)
    if status != "complete":
        reason = "team_schedule_missing" if status is None else "team_schedule_incomplete"
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)
    games = team_games.get((date, sel["team_at_pick"]), set())
    if games == {sel["game_pk"]}:
        return _result(slot, date=date, batter_id=batter, game_pk=sel["game_pk"], selection_id=sel["selection_id"],
                       match="inferred", reason="pick_time_team_single_scheduled_game")
    if not games:
        reason = "team_not_on_schedule"
    else:
        reason = "team_schedule_not_unique" if len(games) > 1 else "team_schedule_other_game"
    return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)


def resolve_duplicate_links(matches: list[dict]) -> list[dict]:
    """Interpretation I8: two contest slot identities linking one selection (an entry changed within a
    round) are both demoted to ambiguous; neither transfers a grade."""
    counts = Counter(m["selection_id"] for m in matches if m["selection_id"])
    return [dict(m, match="ambiguous", match_reason="multiple_contest_slots_for_selection", selection_id=None)
            if m["selection_id"] and counts[m["selection_id"]] > 1 else m for m in matches]
