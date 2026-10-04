"""W2.3 MLB forecast benchmark, pure core: the as-of join of MLB's probabilityStarter to our served slate
(design docs/superpowers/specs/2026-10-04-mlb-forecast-benchmark-design.md rev 2, gate 1)."""
from __future__ import annotations

import pandas as pd


def latest_sheet(captures: list[tuple], round_dates: dict, date: str, cutoff: pd.Timestamp):
    """The latest stored whole sheet at or before ``cutoff``, THEN its rows for the round dated ``date``.

    The sheet is chosen before filtering (design A-E1): a newer sheet that is empty or lists only other rounds gives
    no support, and an older sheet is never consulted instead. A stored newer sheet is a full new observation, so
    players absent from it are absent. Returns ``(stamp, rows)``; ``(None, [])`` when no sheet precedes the cutoff.
    The stamp is the capture run's start time, not a per-feed receipt time (timing uncertain)."""
    eligible = [(pd.Timestamp(stamp), stamp, rows) for stamp, rows in captures if pd.Timestamp(stamp) <= cutoff]
    if not eligible:
        return None, []
    _, stamp, rows = max(eligible, key=lambda x: x[0])
    return stamp, [r for r in rows if round_dates.get(r.get("roundId")) == date]


def forecasts_asof(captures: list[tuple], round_dates: dict, players: dict, date: str, cutoff: pd.Timestamp) -> dict:
    """batter_id → {p, n_sel, captured_at, round_id} from ``latest_sheet`` only."""
    stamp, rows = latest_sheet(captures, round_dates, date, cutoff)
    out: dict = {}
    for r in rows:
        bid = players.get(r.get("playerId"))
        if bid is None or r.get("probabilityStarter") is None:
            continue
        if int(bid) in out:                       # one player listed twice in one sheet: conflicting, excluded
            out[int(bid)] = None
            continue
        out[int(bid)] = {"p": float(r["probabilityStarter"]), "n_sel": r.get("numberSelections"),
                         "captured_at": stamp, "round_id": r.get("roundId"), "player_id": r.get("playerId")}
    return {k: v for k, v in out.items() if v is not None}


def games_by_batter(fc: dict, player_squads: dict, units: list[dict], round_id: int) -> dict:
    """batter_id → the set of game_pks (unit feedId) in round ``round_id`` whose home or away squad is the player's
    squad, both read from MLB's own sheets at the declared observation boundary (design A-E2). Our slate is never
    consulted. ``None`` means multiplicity unknown: a unit of the round (or of no resolved round) with an unresolved
    squad could be anyone's game, and a squad's unit with an unresolved feedId is a possible second game; neither is
    discarded to manufacture a singleton. A player without a squad gets an empty set."""
    unresolved_round = any(u.get("roundId") is None for u in units)
    todays = [u for u in units if u.get("roundId") == round_id]
    unresolved_squad = any(u.get("homeSquadId") is None or u.get("awaySquadId") is None for u in todays)
    out = {}
    for bid, f in fc.items():
        squad = player_squads.get(f.get("player_id"))
        if squad is None:
            out[bid] = set()
            continue
        if unresolved_round or unresolved_squad:
            out[bid] = None
            continue
        mine = [u for u in todays if squad in (u.get("homeSquadId"), u.get("awaySquadId"))]
        out[bid] = None if any(u.get("feedId") is None for u in mine) else {int(u["feedId"]) for u in mine}
    return out


def join_to_slate(slate: pd.DataFrame, fc: dict, batter_games: dict) -> tuple[pd.DataFrame, dict]:
    """Attach mlb_p to slate rows. A forecast row has no unit, so its game is linked only by inference: MLB's sheets
    must give the batter exactly one game that date (``games_by_batter``) and the slate row must be that game.
    Anything else stays unmatched and is counted. The link is labelled ``inferred_unique_game``, never witnessed."""
    joined = slate.copy()
    status, mlb_p, stamps = [], [], []
    for b, g in zip(joined["batter_id"], joined["game_pk"]):
        games = batter_games.get(b, set())
        if b not in fc:
            status.append("not_listed"); mlb_p.append(float("nan")); stamps.append(None)
        elif games is None:
            status.append("multiplicity_unknown"); mlb_p.append(float("nan")); stamps.append(None)
        elif len(games) != 1:
            status.append("multi_or_no_game"); mlb_p.append(float("nan")); stamps.append(None)
        elif int(g) not in games:
            status.append("game_mismatch"); mlb_p.append(float("nan")); stamps.append(None)
        else:
            status.append("inferred_unique_game"); mlb_p.append(fc[b]["p"]); stamps.append(fc[b]["captured_at"])
    joined["link_status"], joined["mlb_p"], joined["mlb_captured_at"] = status, mlb_p, stamps
    cov = {"slate_rows": int(len(slate)), "mlb_listed": len(fc),
           **{k: int(v) for k, v in joined["link_status"].value_counts().items()},
           "mlb_not_in_slate": len(set(fc) - set(slate["batter_id"]))}
    return joined, cov
