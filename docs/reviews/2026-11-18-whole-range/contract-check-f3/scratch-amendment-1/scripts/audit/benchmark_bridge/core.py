"""W1.2 benchmark bridge, pure core (design docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md, rev 2).

Everything here is deterministic and data-free, so it is tested with synthetic inputs. Nothing reads outcomes.
"""
from __future__ import annotations

import json
import math
from typing import Callable

import numpy as np
import pandas as pd

from bts.simulate.backtest_blend import PA_EST_BY_LINEUP

SUBMISSION_CUTOFF = pd.Timedelta(minutes=5)


def serialized_probability(p: float) -> float:
    """The value the slate writer stores for ``p`` (pandas ``to_json`` default precision, bts.slate.save_slate)."""
    return json.loads(pd.DataFrame({"p": [p]}).to_json(orient="records"))[0]["p"]


def _key(row: dict) -> tuple[int, int] | None:
    try:
        return int(row["batter_id"]), int(row["game_pk"])
    except (KeyError, TypeError, ValueError):
        return None


def selection_consistency(slate_rows: list[dict], decision_primary: dict | None,
                          pick_primary: dict | None) -> dict:
    """Decision-first selection consistency of one date's slate (design §2, Codex r1 F1).

    A match means the selected (batter_id, game_pk) is a slate row whose stored probability equals the selected
    probability after the writer's serialization. It establishes consistency only: never that this slate file
    produced the final action. A disagreement between the decision and the pick file is recorded, not resolved.
    """
    chosen, source = (decision_primary, "decision") if decision_primary else (pick_primary, "pick")
    conflict = bool(decision_primary and pick_primary and _key(decision_primary) != _key(pick_primary))
    if chosen is None:
        return {"state": "no_selection", "source": None, "conflict": False}
    key = _key(chosen)
    want = serialized_probability(float(chosen["p_game_hit"])) if chosen.get("p_game_hit") is not None else None
    for r in slate_rows:
        if _key(r) == key and want is not None and r.get("p_game_hit") == want:
            return {"state": "selection_consistent", "source": source, "conflict": conflict}
    return {"state": "inconsistent", "source": source, "conflict": conflict}


def count_normalized(a26: float, n_actual: int, lineup_slot: int | None, n_est_override: float | None = None) -> float:
    """A26-count (design §3, Codex r1 F2): ``1 − (1 − A26) ** (N_est / N_actual)``.

    Holds the geometric per-PA no-hit rate fixed; equals A26 when N_est == N_actual. NaN when N_actual <= 0.
    """
    if not n_actual or n_actual <= 0:
        return float("nan")
    n_est = n_est_override if n_est_override is not None else PA_EST_BY_LINEUP.get(lineup_slot, 4.0)
    return 1.0 - (1.0 - a26) ** (n_est / n_actual)


def outcome_label(n_pa: int, n_hits: int, source_complete: bool) -> str:
    """Baseball-event label (design §4): no_pa only from a complete source; otherwise unknown."""
    if n_pa > 0:
        return "hit" if n_hits > 0 else "no_hit"
    return "no_pa" if source_complete else "unknown"


def eligible_surrogate(scheduled_start: pd.Timestamp | None, written_at: pd.Timestamp) -> bool | None:
    """The scheduled-time surrogate (design §5): first pitch more than 5 minutes after the slate's written_at."""
    if scheduled_start is None or pd.isna(scheduled_start):
        return None
    return bool(scheduled_start - SUBMISSION_CUTOFF > written_at)


def rank1(df: pd.DataFrame, score_col: str, eligible_col: str = "eligible", order_col: str = "row_order"):
    """Index of the eligible argmax; exact ties broken by the archived D row order. None when nothing is eligible.

    Ranking happens before outcomes are joined, so a no_pa/unknown winner is never replaced (design §4)."""
    cand = df[df[eligible_col].fillna(False).astype(bool) & df[score_col].notna()]
    if cand.empty:
        return None
    best = cand[score_col].max()
    return cand[cand[score_col] == best].sort_values(order_col).index[0]


def paired_pool(df: pd.DataFrame, a: str, b: str, eligible_col: str = "eligible") -> tuple[pd.DataFrame, dict]:
    """The identical pool for an arm pair (design §6): finite in both arms and eligible; exclusions counted."""
    fin_a, fin_b = np.isfinite(df[a].astype(float)), np.isfinite(df[b].astype(float))
    elig = df[eligible_col].fillna(False).astype(bool)
    excl = {"missing_a": int((~fin_a).sum()), "missing_b": int((~fin_b).sum()),
            "ineligible": int((fin_a & fin_b & ~elig).sum())}
    return df[fin_a & fin_b & elig], excl


def _rank_auc(pos, neg) -> float | None:
    from bts.health.slate_auc import _rank_auc as auc
    return auc(list(pos), list(neg))


def equal_date_auc(df: pd.DataFrame, score: str, y: str, date_col: str = "date") -> tuple[float, int, int]:
    """Equal-date mean of tie-aware within-date AUCs; dates lacking either class are omitted and counted."""
    vals, omitted = [], 0
    for _, g in df.groupby(date_col, sort=True):
        a = _rank_auc(g.loc[g[y] == 1, score], g.loc[g[y] == 0, score])
        if a is None:
            omitted += 1
        else:
            vals.append(a)
    return (float(np.mean(vals)) if vals else float("nan")), len(vals), omitted


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-15) -> float:
    p, y = np.clip(np.asarray(p, float), eps, 1 - eps), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def date_block_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], n_resamples: int = 10_000,
                         seed: int = 20261004, date_col: str = "date", return_draws: bool = False):
    """Percentile interval of ``stat`` over whole dates resampled with replacement (repeated dates kept)."""
    dates = np.array(sorted(df[date_col].unique()))
    groups = {d: g for d, g in df.groupby(date_col, sort=False)}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.choice(dates, size=len(dates), replace=True)
        draws[i] = stat(pd.concat([groups[d] for d in pick], ignore_index=True))
    if return_draws:
        return draws
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


ACCEPTED_FILES = ("table.parquet", "summary.json", "manifest.json")


def read_accepted_run(run_dir) -> tuple[dict, dict]:
    """The accepted W1.2 run's files as bytes, each read once and verified against the run's ``ACCEPTED.json`` (written
    when the W1.2 memo accepts the run). Consumers parse these bytes, so what is hashed is what is used. Refuses a run
    without an acceptance record or with any mismatch."""
    import hashlib
    from pathlib import Path
    d = Path(run_dir)
    acc_path = d / "ACCEPTED.json"
    if not acc_path.exists():
        raise SystemExit(f"{d} has no ACCEPTED.json: not an accepted W1.2 run")
    acc_bytes = acc_path.read_bytes()
    acc = json.loads(acc_bytes)
    out = {"ACCEPTED.json": acc_bytes}
    for name in ACCEPTED_FILES:
        b = (d / name).read_bytes()
        if hashlib.sha256(b).hexdigest() != acc.get("files", {}).get(name):
            raise SystemExit(f"{name} in {d} does not match ACCEPTED.json")
        out[name] = b
    return out, acc


def capture_files(directory) -> list:
    """Every static capture in ``directory``, plain ``.json`` and gzipped ``.json.gz`` alike, in stamp order. The
    capture writer switched to gzip on 2026-07-10, so a plain-only glob silently drops every later capture."""
    from pathlib import Path
    d = Path(directory)
    return sorted([*d.glob("*.json"), *d.glob("*.json.gz")], key=lambda p: p.name)


def slim_feed(feed: dict | None) -> dict | None:
    """The archived feed reduced to ``gameData`` without its ``players`` map: every field the bridge reads (teams,
    weather, officials, venue, datetime, status) and none of ``liveData``. A full v1.1 feed costs about 4 MB in
    memory; a season of slate games held whole exhausted the run's memory."""
    if feed is None:
        return None
    return {"gameData": {k: v for k, v in feed.get("gameData", {}).items() if k != "players"}}


def reconstruct_slot(row: dict, feed: dict) -> tuple[dict | None, dict]:
    """One ``_fetch_game_slots``-shaped slot for C (design §3, Codex r1 F4).

    Batter-level inputs come from the D slate row: identity, lineup, pitcher, projected state. Game-level fields
    come from the archived final feed (source ``final_feed``: post-game values, not proof of what was live at
    serve time). ``pitcher_hand`` is None, as at pre-game serving, where no plays existed and predict() fell back
    to its lookup. Returns ``(None, {"reason": ...})`` when the row cannot be placed in the feed.
    """
    gd = feed.get("gameData", {})
    teams = gd.get("teams", {})
    side = next((s for s in ("home", "away") if teams.get(s, {}).get("abbreviation") == row.get("team")), None)
    if side is None:
        return None, {"reason": "team_not_in_feed"}
    opp = "away" if side == "home" else "home"
    weather = gd.get("weather", {}) or {}
    wind = weather.get("wind", "") or ""
    temp = weather.get("temp")
    speed = wind.split(" mph")[0] if "mph" in wind else None
    hp = next((o.get("official", {}).get("id") for o in gd.get("officials", [])
               if o.get("officialType") == "Home Plate"), None)
    slot = {
        "batter_id": int(row["batter_id"]), "batter_name": row.get("batter_name"), "team": row["team"],
        "pitcher_team": teams[opp].get("abbreviation"), "opp_team_id": teams[opp].get("id"),
        "lineup": row.get("lineup"), "pitcher_id": row.get("pitcher_id"), "pitcher_name": row.get("pitcher_name"),
        "pitcher_hand": None, "venue_id": gd.get("venue", {}).get("id"),
        "weather_temp": int(temp) if temp else None,
        "weather_wind_dir": wind.split(", ", 1)[1] if ", " in wind else "",
        "weather_wind_speed": float(speed) if speed else 0.0,
        "roof_type": gd.get("venue", {}).get("fieldInfo", {}).get("roofType", ""),
        "hp_umpire_id": hp, "game_pk": int(row["game_pk"]),
    }
    if row.get("projected"):
        slot["projected"] = True
    return slot, {"game_fields": "final_feed", "pitcher_hand": "serving_none"}


def history_and_lookups(date: str, df_feat: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """The pre-date frame, its serving lookups and an opener-check cache, built once per date and shared by every C
    scoring that day. ``_check_opener`` depends only on (pitcher_id, the pre-date frame) and its result is read only,
    so caching it per date changes no score."""
    import bts.model.predict as pm
    dates = df_feat["date"]
    if dates.is_monotonic_increasing:      # a contiguous prefix: a row slice (no copy of the whole feature frame)
        hist = df_feat.iloc[: int(dates.searchsorted(pd.Timestamp(date), side="left"))]
    else:
        hist = df_feat[dates < pd.Timestamp(date)]
    return hist, pm._build_feature_lookups(hist), {}


def score_c(date: str, slots: list[dict], df_feat: pd.DataFrame, artifact: dict, check_openers: bool = True,
            feature_cols: list[str] | None = None, prepared: tuple | None = None) -> pd.DataFrame:
    """Score injected slots with an archived (or frozen) artifact through the real ``predict()`` (design §3).

    History, lookups and the opener check use only rows dated before ``date``, as serving's morning frame did.
    The artifact is copied: ``_model`` is the single model and the rest is the blend, as ``run_pipeline`` unpacks a
    cached blend. ``_fetch_game_slots`` is replaced only for this call, so nothing touches the network.
    """
    from unittest import mock

    import bts.model.predict as pm

    hist, lookups, opener_cache = prepared if prepared is not None else history_and_lookups(date, df_feat)
    blend = dict(artifact)
    model = blend.pop("_model")
    real_opener = pm._check_opener

    def cached_opener(pid, frame):
        if pid not in opener_cache:
            opener_cache[pid] = real_opener(pid, frame)
        return opener_cache[pid]

    with mock.patch.object(pm, "_fetch_game_slots", lambda _d: slots), \
            mock.patch.object(pm, "_check_opener", cached_opener):
        out = pm.predict(date, hist, model, lookups, check_openers=check_openers, blend=blend,
                         feature_cols=feature_cols)
    return out[["batter_id", "game_pk", "p_game_hit"]].copy() if not out.empty else out


def verified_eligibility(status_history: list[tuple], written_at: pd.Timestamp, pregame: set, not_pregame: set):
    """Eligibility from the latest BTS unit capture at or before the slate's written_at (design §5).

    True for a pre-game status, False for a started/final/postponed one, None when no capture precedes written_at or
    the status is not classified (never guessed)."""
    before = [(pd.Timestamp(t), s) for t, s in status_history if t is not None and pd.Timestamp(t) <= written_at]
    if not before:
        return None
    status = max(before, key=lambda x: x[0])[1]
    if status in pregame:
        return True
    if status in not_pregame:
        return False
    return None


def outcome_table(cands: pd.DataFrame, pa: pd.DataFrame, final_games: set) -> pd.DataFrame:
    """Label each (date, batter_id, game_pk) candidate from scoring PA rows (resumed portion already excluded).

    no_pa needs the game in ``final_games`` (a complete, final source); otherwise a row without PA is unknown."""
    agg = (pa.groupby(["date", "game_pk", "batter_id"])["is_hit"].agg(n_pa="size", n_hits="sum").reset_index())
    out = cands.merge(agg, on=["date", "game_pk", "batter_id"], how="left")
    out["n_pa"] = out["n_pa"].fillna(0).astype(int)
    out["n_hits"] = out["n_hits"].fillna(0).astype(int)
    out["outcome"] = [outcome_label(n, h, int(g) in final_games)
                      for n, h, g in zip(out["n_pa"], out["n_hits"], out["game_pk"])]
    return out
