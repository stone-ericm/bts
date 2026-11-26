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


def valid_p(v) -> bool:
    import math
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0


def forecasts_counted(captures: list[tuple], round_dates: dict, players: dict, date: str, cutoff: pd.Timestamp):
    """``forecasts_asof`` with its exclusions counted (code review r1 F1): duplicate player rows are resolved on the
    raw sheet BEFORE probability validity (a player listed twice is excluded whatever its values), then invalid
    probabilities, unmapped players and two players mapping to one batter are excluded and counted."""
    stamp, rows = latest_sheet(captures, round_dates, date, cutoff)
    counts = {"rows": len(rows), "duplicate_player": 0, "invalid_probability": 0, "unmapped_player": 0,
              "batter_conflict": 0, "invalid_identity": 0}
    by_pid: dict = {}
    for r in rows:
        if not _int(r.get("playerId")) or not _int(r.get("roundId")):
            counts["invalid_identity"] += 1
            continue
        by_pid.setdefault(r["playerId"], []).append(r)
    out: dict = {}
    for pid, rs in by_pid.items():
        if len(rs) > 1:
            counts["duplicate_player"] += 1
            continue
        r = rs[0]
        if not valid_p(r.get("probabilityStarter")):
            counts["invalid_probability"] += 1
            continue
        bid = players.get(pid)
        if bid is None:
            counts["unmapped_player"] += 1
            continue
        if int(bid) in out:
            out[int(bid)] = None
            counts["batter_conflict"] += 1
            continue
        out[int(bid)] = {"p": float(r["probabilityStarter"]), "n_sel": r.get("numberSelections"),
                         "captured_at": stamp, "round_id": r.get("roundId"), "player_id": pid}
    return {k: v for k, v in out.items() if v is not None}, counts


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
    unresolved_round = any(not _int(u.get("roundId")) for u in units)
    todays = [u for u in units if u.get("roundId") == round_id]
    unresolved_squad = any(not _int(u.get("homeSquadId")) or not _int(u.get("awaySquadId")) for u in todays)
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
        out[bid] = None if any(not _int(u.get("feedId")) for u in mine) else {u["feedId"] for u in mine}
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


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def round_for_date(rounds: list[dict], date: str) -> tuple[int | None, str | None]:
    """The single round id dated ``date`` in a rounds sheet. A round id carrying two different dates anywhere in the
    sheet is a contradiction (code review r1 F1), as are zero or several rounds for the date; none is guessed."""
    dates_by_id: dict = {}
    for r in rounds:
        if _int(r.get("id")) and isinstance(r.get("date"), str):
            dates_by_id.setdefault(r["id"], set()).add(r["date"][:10])
    ids = sorted(i for i, ds in dates_by_id.items() if date in ds)
    if not ids:
        return None, "no_round"
    if any(len(dates_by_id[i]) > 1 for i in ids):
        return None, "conflicting_round"
    if len(ids) > 1:
        return None, "multiple_rounds"
    return ids[0], None


def unit_conflicts(units: list[dict], round_id: int) -> int:
    """Contradictions touching ``round_id`` (code review r1 F1, r2 N1): one unit id carrying two different
    (feedId, roundId, home, away) mappings anywhere in the sheet, when any of them is in the target round, counted
    BEFORE round filtering; and, within the round, one game (feedId) listed with two different squad pairs."""
    by_unit: dict = {}
    for u in units:
        if _int(u.get("id")):
            by_unit.setdefault(u["id"], set()).add((u.get("feedId"), u.get("roundId"), u.get("homeSquadId"),
                                                    u.get("awaySquadId")))
    n = sum(1 for keys in by_unit.values() if len(keys) > 1 and any(k[1] == round_id for k in keys))
    by_game = {}
    for u in units:
        if u.get("roundId") != round_id or not _int(u.get("roundId")):
            continue
        pair = (u.get("homeSquadId"), u.get("awaySquadId"))
        if u.get("feedId") in by_game and by_game[u.get("feedId")] != pair:
            n += 1
        by_game.setdefault(u.get("feedId"), pair)
    return n


def player_lookup(players: list[dict]) -> tuple[dict, dict, list]:
    """playerId → feedId (batter_id) and → squadId from one players sheet. An id listed twice with different
    feedId or squadId is a conflict and excluded from both maps (identical duplicates are harmless)."""
    seen: dict = {}
    conflicts = set()
    for p in players:
        pid = p.get("id")
        if not _int(pid):
            continue
        val = (p.get("feedId") if _int(p.get("feedId")) else None, p.get("squadId") if _int(p.get("squadId")) else None)
        if pid in seen and seen[pid] != val:
            conflicts.add(pid)
        seen.setdefault(pid, val)
    feed = {k: v[0] for k, v in seen.items() if k not in conflicts and v[0] is not None}
    squad = {k: v[1] for k, v in seen.items() if k not in conflicts and v[1] is not None}
    return feed, squad, sorted(conflicts)


def units_complete(units: list[dict], round_id: int, scheduled_game_pks: set) -> bool:
    """True only when every game the MLB schedule lists for the date has a unit in the round. Without schedule
    evidence the unit universe is not established complete, so no unique-game inference is made from it."""
    if not scheduled_game_pks:
        return False
    have = {u["feedId"] for u in units if _int(u.get("roundId")) and u["roundId"] == round_id and _int(u.get("feedId"))}
    return set(scheduled_game_pks) <= have


def unchanged_since(captures: list[tuple], stamp: str, round_id: int, player_id: int) -> str | None:
    """The earliest stored sheet stamp from which this player's probability for the round is identical in every
    stored sheet up to ``stamp``. An observation measure only: a stored sheet with the player absent ends the run,
    and an unstored interval is not assumed unchanged beyond what the stored sheets show."""
    ordered = sorted(captures, key=lambda c: pd.Timestamp(c[0]))
    upto = [c for c in ordered if pd.Timestamp(c[0]) <= pd.Timestamp(stamp)]

    def value(rows):
        vals = [r.get("probabilityStarter") for r in rows
                if r.get("roundId") == round_id and r.get("playerId") == player_id]
        return vals[0] if len(vals) == 1 else None
    if not upto or value(upto[-1][1]) is None:
        return None
    target, since = value(upto[-1][1]), upto[-1][0]
    for st, rows in reversed(upto[:-1]):
        if value(rows) != target:
            break
        since = st
    return since
