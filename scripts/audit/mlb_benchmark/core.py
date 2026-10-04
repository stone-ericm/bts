"""W2.3 MLB forecast benchmark, pure core: the as-of join of MLB's probabilityStarter to our served slate."""
from __future__ import annotations

import pandas as pd


def forecasts_asof(captures: list[tuple], round_dates: dict, players: dict, date: str, cutoff: pd.Timestamp) -> dict:
    """batter_id → {p, n_sel, captured_at, round_id} from the latest capture at or before ``cutoff`` listing a round
    dated ``date`` (design gate i). Captures are content-deduped, so a player absent from a later capture keeps the
    value of the last earlier capture listing them; tomorrow's round never supplies today's forecast."""
    out: dict = {}
    for captured_at, rows in sorted(captures, key=lambda c: pd.Timestamp(c[0])):
        if pd.Timestamp(captured_at) > cutoff:
            break
        for r in rows:
            if round_dates.get(r.get("roundId")) != date:
                continue
            bid = players.get(r.get("playerId"))
            if bid is None or r.get("probabilityStarter") is None:
                continue
            out[int(bid)] = {"p": float(r["probabilityStarter"]), "n_sel": r.get("numberSelections"),
                             "captured_at": captured_at, "round_id": r.get("roundId")}
    return out


def join_to_slate(slate: pd.DataFrame, fc: dict) -> tuple[pd.DataFrame, dict]:
    """Attach mlb_p to the slate rows by batter; a batter with more than one slate row that day (a doubleheader) is
    ambiguous and stays unmatched. Coverage is reported both ways."""
    counts = slate["batter_id"].value_counts()
    dh = set(counts[counts > 1].index)
    joined = slate.copy()
    joined["mlb_p"] = [fc[b]["p"] if (b in fc and b not in dh) else float("nan") for b in joined["batter_id"]]
    joined["mlb_captured_at"] = [fc[b]["captured_at"] if (b in fc and b not in dh) else None for b in joined["batter_id"]]
    slate_ids = set(slate["batter_id"])
    cov = {"slate_rows": int(len(slate)), "mlb_listed": len(fc),
           "matched_unique": int(joined["mlb_p"].notna().sum()),
           "ambiguous_doubleheader": len(dh & set(fc)),
           "mlb_not_in_slate": len(set(fc) - slate_ids),
           "slate_not_listed": len(slate_ids - set(fc))}
    return joined, cov
