"""Daily producer for the park_drag_delta external table (arming item 2).

Refreshes ``data/external/park_drag/`` on the box:

    producer/pitches_current.csv.gz   four-seam pitch store, current season
    producer/games_meta.csv           per-game weather/venue cache (all seasons)
    producer/game_level_static.csv    frozen game-level Cd for PRIOR seasons
    park_drag_export.csv              consumer artifact (see features/park_drag.py)
    park_drag_manifest.json           freshness metadata
    producer_status.json              last-run outcome for the health source

Flow: fetch the last ``lookback_days`` of four-seam pitches from Baseball
Savant (window-replace into the store — idempotent), fetch game meta for new
game_pks from the MLB Stats API, recompute the current season's game-level
drag (Nathan 9P method, elevation + game-temp adjusted), concat with the
static prior seasons, rebuild the serving-correct export (one row per
venue_id × calendar date from strictly-prior games), write atomically.

House scraper hygiene: ``browser_headers`` identity (Savant 403s default UAs
from datacenter IPs), jittered pacing, and a hard 403/429 ->
``RateLimitedError`` abort (never hammer). Statcast is public/unauthenticated
— this job never touches contest-account cookies.

Season rollover: when the store's newest season is behind today's year the
run fails with a seed instruction (one-time reseed from the analysis pipeline
in ~/projects/juiced-ball-analysis, which remains the historical producer).

Derived from the validated pipeline in ~/projects/juiced-ball-analysis
(compute_game_cd.py / build_feature_table.py); the feature definition is
specced in docs/superpowers/specs/2026-07-07-park-drag-delta-context-feature.md.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bts.leaderboard.endpoints import browser_headers
from bts.leaderboard.scraper import RateLimitedError

# ---------------- physics constants (Nathan 9P method, imperial) ----------------
G = 32.174                                  # ft/s^2
M_SLUG = (5.125 / 16.0) / G                 # 5.125 oz ball -> slugs
R_FT = 9.125 / (2 * math.pi) / 12.0         # 9.125 in circumference -> radius ft
AREA = math.pi * R_FT ** 2
RHO0 = 0.0023769                            # slug/ft^3 at 59F sea level

TEAM_ELEV = {
    "AZ": 1086, "ARI": 1086, "ATL": 1050, "BAL": 40, "BOS": 20, "CHC": 600,
    "CWS": 600, "CHW": 600, "CIN": 490, "CLE": 650, "COL": 5190, "DET": 585,
    "HOU": 45, "KC": 750, "KCR": 750, "LAA": 160, "LAD": 340, "MIA": 7,
    "MIL": 635, "MIN": 830, "NYM": 10, "NYY": 55, "ATH": 25, "OAK": 25,
    "PHI": 30, "PIT": 730, "SD": 15, "SDP": 15, "SEA": 15, "SF": 10, "SFG": 10,
    "STL": 465, "TB": 45, "TBR": 45, "TEX": 550, "TOR": 250, "WSH": 25, "WSN": 25,
}
VENUE_ELEV_OVERRIDE = {   # neutral sites / temporary homes (substring match)
    "Tokyo": 30, "Alfredo Harp": 7349, "Estadio": 7349, "London": 50,
    "Sutter": 25, "Steinbrenner": 10, "Bristol": 1443, "Williamsport": 528,
    "Rickwood": 600, "Muncy": 30, "Las Vegas": 2000, "Sahlen": 600,
    "TD Ballpark": 40, "Field of Dreams": 980, "Gocheok": 30,
}
INDOOR_CONDITIONS = {"Dome", "Roof Closed"}

REQUIRED_PITCH_COLS = ["game_pk", "game_date", "home_team", "release_speed",
                       "vx0", "vy0", "vz0", "ax", "ay", "az"]
META_FIELDS = "gameData,datetime,dateTime,dayNight,weather,condition,temp,wind,venue,id,name"
SHRINK_K = 500.0          # pitches; w = N/(N+K) — must match the analysis builder
ROW_CAP_GUARD = 23000     # Savant csv row cap is ~25k; warn margin


class ProducerError(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _jitter_sleep(lo: float = 1.5, hi: float = 3.5) -> None:
    time.sleep(random.uniform(lo, hi))


def _http_get(url: str, *, accept: str, timeout: float = 180.0):
    import httpx
    return httpx.get(url, headers=browser_headers(accept=accept), timeout=timeout)


# ---------------- fetch ----------------

def fetch_savant_ff(season: int, d0: date, d1: date, http_get=_http_get) -> pd.DataFrame:
    """Four-seam pitch rows for [d0, d1]. 403/429 -> RateLimitedError (abort)."""
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?all=true"
        f"&hfPT=FF%7C&hfGT=R%7C&hfSea={season}%7C"
        f"&game_date_gt={d0}&game_date_lt={d1}"
        "&player_type=pitcher&min_pitches=0&min_results=0"
        "&group_by=name&sort_col=pitches&player_event_sort=api_p_release_speed"
        "&sort_order=desc&type=details"
    )
    last_err: Exception | None = None
    for attempt in range(4):
        if attempt:
            time.sleep(4 * attempt + random.uniform(0, 2))
        try:
            r = http_get(url, accept="text/csv, */*")
            if r.status_code in (403, 429):
                raise RateLimitedError(r.status_code, url)
            r.raise_for_status()
            if r.text.lstrip().startswith("<"):
                raise ProducerError("savant returned HTML, not CSV")
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            if len(df) > ROW_CAP_GUARD:
                raise ProducerError(f"savant window {d0}..{d1} near row cap ({len(df)})")
            missing = [c for c in REQUIRED_PITCH_COLS if c not in df.columns]
            if len(df) and missing:
                raise ProducerError(f"savant csv missing columns {missing}")
            return df[REQUIRED_PITCH_COLS] if len(df) else df
        except RateLimitedError:
            raise  # kill-switch: never retry a 403/429
        except Exception as e:  # noqa: BLE001 — retry transient failures
            last_err = e
    raise ProducerError(f"savant fetch failed for {d0}..{d1}: {last_err}")


def fetch_game_meta(pks: list[int], http_get=_http_get, max_workers: int = 6) -> pd.DataFrame:
    """Weather/venue/start for game_pks via the trimmed statsapi live feed."""
    def one(pk: int) -> dict:
        url = (f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
               f"?fields={META_FIELDS}")
        for attempt in range(3):
            if attempt:
                time.sleep(2 * attempt)
            try:
                r = http_get(url, accept="application/json", timeout=30.0)
                r.raise_for_status()
                gd = r.json().get("gameData", {})
                w = gd.get("weather", {}) or {}
                dt = gd.get("datetime", {}) or {}
                v = gd.get("venue", {}) or {}
                return {"game_pk": pk, "start_utc": dt.get("dateTime"),
                        "day_night": dt.get("dayNight"), "temp_f": w.get("temp"),
                        "condition": w.get("condition"), "wind": w.get("wind"),
                        "venue_id": v.get("id"), "venue": v.get("name")}
            except Exception:  # noqa: BLE001
                continue
        return {"game_pk": pk}

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, pk) for pk in pks]
        for f in as_completed(futs):
            rows.append(f.result())
    return pd.DataFrame(rows)


# ---------------- physics ----------------

def _elevation(home_team: str, venue) -> float:
    if isinstance(venue, str):
        for key, e in VENUE_ELEV_OVERRIDE.items():
            if key in venue:
                return float(e)
    return float(TEAM_ELEV.get(home_team, np.nan))


def air_density(elev_ft, temp_f):
    press_ratio = (1.0 - 6.8756e-6 * elev_ft) ** 5.2559
    return RHO0 * press_ratio * (518.67 / (temp_f + 459.67))


def trimmed_mean(x: np.ndarray, prop: float = 0.05) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    k = int(math.floor(prop * len(x)))
    if len(x) - 2 * k <= 0:
        return float(np.mean(x)) if len(x) else float("nan")
    return float(np.mean(x[k:len(x) - k]))


def compute_game_level(pitches: pd.DataFrame, meta: pd.DataFrame, season: int) -> pd.DataFrame:
    """Per-game env-adjusted trimmed-mean Cd (>=25 QC'd four-seamers)."""
    df = pitches.merge(meta.drop_duplicates("game_pk"), on="game_pk", how="left")
    df = df.dropna(subset=["vx0", "vy0", "vz0", "ax", "ay", "az", "release_speed"])
    df = df[(df.release_speed >= 85) & (df.release_speed <= 104.5)]
    df["temp_f"] = pd.to_numeric(df["temp_f"], errors="coerce").clip(30, 110).fillna(70.0)
    df["elev"] = [
        _elevation(ht, v) for ht, v in zip(df["home_team"], df["venue"])
    ]
    df = df.dropna(subset=["elev", "venue_id"])
    rho = air_density(df["elev"].values, df["temp_f"].values)
    v = np.sqrt(df.vx0 ** 2 + df.vy0 ** 2 + df.vz0 ** 2)
    a_drag = -(df.ax * df.vx0 + df.ay * df.vy0 + (df.az + G) * df.vz0) / v
    df["cd"] = M_SLUG * a_drag / (0.5 * rho * AREA * v ** 2)
    df = df[(df.cd > 0.24) & (df.cd < 0.50)]

    rows = []
    for pk, g in df.groupby("game_pk"):
        if len(g) < 25:
            continue
        first = g.iloc[0]
        rows.append({
            "game_pk": pk, "season": season, "game_date": first["game_date"],
            "venue_id": int(first["venue_id"]), "venue": first["venue"],
            "n_pitch": len(g), "cd_trim": trimmed_mean(g.cd.values),
        })
    return pd.DataFrame(rows)


# ---------------- builder (must stay consistent with the analysis repo) ----------------

def build_export(game_level: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = game_level.dropna(subset=["venue_id", "cd_trim"]).copy()
    g["venue_id"] = g["venue_id"].astype("int64")
    g["game_date"] = pd.to_datetime(g["game_date"])

    vd = (g.assign(wcd=g.cd_trim * g.n_pitch)
            .groupby(["venue_id", "season", "game_date"], as_index=False)
            .agg(wcd=("wcd", "sum"), n_pitch=("n_pitch", "sum")))
    vd["cd_venue_date"] = vd.wcd / vd.n_pitch
    vd = vd.sort_values(["venue_id", "game_date"]).reset_index(drop=True)

    grp = vd.groupby(["venue_id", "season"], group_keys=False)
    vd["roll15_incl"] = grp["cd_venue_date"].apply(
        lambda x: x.rolling(15, min_periods=5).mean())
    vd["npitch15_incl"] = grp["n_pitch"].apply(
        lambda x: x.rolling(15, min_periods=5).sum())
    vd["nwin_incl"] = grp["cd_venue_date"].apply(
        lambda x: x.rolling(15, min_periods=5).count())
    vd["expand_incl"] = grp["cd_venue_date"].apply(
        lambda x: x.expanding(min_periods=10).mean())
    vd["date_idx"] = grp.cumcount()
    anchors = vd.loc[vd.date_idx == 9, ["venue_id", "season", "expand_incl"]].rename(
        columns={"expand_incl": "anchor10"})
    vd = vd.merge(anchors, on=["venue_id", "season"], how="left")
    vd.loc[vd.date_idx < 9, "anchor10"] = np.nan

    rows = []
    for season, gs in vd.groupby("season"):
        d0, d1 = gs.game_date.min(), gs.game_date.max()
        cal = pd.DataFrame({"date": pd.date_range(d0, d1 + pd.Timedelta(days=1))})
        for vid, s in gs.groupby("venue_id"):
            right = s[["game_date", "roll15_incl", "expand_incl", "anchor10",
                       "npitch15_incl", "nwin_incl"]].rename(columns={"game_date": "date"})
            m = pd.merge_asof(cal, right, on="date", allow_exact_matches=False)
            m["venue_id"] = vid
            rows.append(m)
    ex = pd.concat(rows, ignore_index=True)
    w = ex.npitch15_incl / (ex.npitch15_incl + SHRINK_K)
    ex["park_drag_delta"] = (ex.roll15_incl - ex.anchor10) * w
    ex["park_drag_delta_expanding"] = (ex.roll15_incl - ex.expand_incl) * w
    ex["park_drag_n_window"] = ex.nwin_incl
    export = ex[["venue_id", "date", "park_drag_delta",
                 "park_drag_delta_expanding", "park_drag_n_window"]].copy()
    if export.duplicated(["venue_id", "date"]).any():
        raise ProducerError("duplicate (venue_id, date) in built export")

    manifest = {
        "built_from": "park_drag_producer (box daily refresh)",
        "generated_at": _now_iso(),
        "max_source_game_date": str(vd.game_date.max().date()),
        "export_rows": int(len(export)),
        "export_nonnull_share": float(export.park_drag_delta.notna().mean()),
        "shrink_k_pitches": SHRINK_K,
        "primary_column": "park_drag_delta (roll15 - anchor10, shrunk)",
    }
    return export, manifest


# ---------------- refresh orchestration ----------------

def _atomic_write_df(df: pd.DataFrame, path: Path, **kw) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def refresh(root: Path, *, today: date | None = None, lookback_days: int = 3,
            http_get=_http_get) -> dict:
    """One daily refresh cycle. Returns a summary dict; writes producer_status.json."""
    root = Path(root)
    prod = root / "producer"
    status_path = root / "producer_status.json"
    summary: dict = {"ok": False, "started_at": _now_iso()}
    try:
        today = today or date.today()
        store_path = prod / "pitches_current.csv.gz"
        meta_path = prod / "games_meta.csv"
        static_path = prod / "game_level_static.csv"
        for p in (store_path, meta_path, static_path):
            if not p.exists():
                raise ProducerError(
                    f"missing producer input {p} — seed from "
                    f"~/projects/juiced-ball-analysis (see module docstring)")

        store = pd.read_csv(store_path, low_memory=False)
        store["game_date"] = pd.to_datetime(store["game_date"])
        season = int(store["game_date"].dt.year.max())
        if today.year > season and today.month >= 3:
            raise ProducerError(
                f"store season {season} is behind {today.year} — reseed for the new season")

        yesterday = today - timedelta(days=1)
        window_start = max(
            (store["game_date"].max() - pd.Timedelta(days=lookback_days)).date(),
            date(season, 3, 1),
        )
        fetched = pd.DataFrame()
        if window_start <= yesterday:
            fetched = fetch_savant_ff(season, window_start, yesterday, http_get=http_get)
            _jitter_sleep()
        summary["fetched_pitches"] = int(len(fetched))
        summary["window"] = [str(window_start), str(yesterday)]

        if len(fetched):
            fetched["game_date"] = pd.to_datetime(fetched["game_date"])
            keep = store[(store["game_date"] < pd.Timestamp(window_start))]
            store = pd.concat([keep, fetched], ignore_index=True)  # window-replace
            _atomic_write_df(store, store_path, compression="gzip")

        meta = pd.read_csv(meta_path)
        new_pks = sorted(set(store.game_pk.unique()) - set(meta.game_pk.unique()))
        summary["new_meta_games"] = len(new_pks)
        if new_pks:
            got = fetch_game_meta(new_pks, http_get=http_get)
            meta = pd.concat([meta, got], ignore_index=True)
            _atomic_write_df(meta, meta_path)

        current = compute_game_level(store, meta, season)
        static = pd.read_csv(static_path)
        game_level = pd.concat([static, current], ignore_index=True)
        export, manifest = build_export(game_level)

        _atomic_write_df(export, root / "park_drag_export.csv")
        _atomic_write_json(manifest, root / "park_drag_manifest.json")
        summary.update(ok=True, export_rows=manifest["export_rows"],
                       max_source_game_date=manifest["max_source_game_date"],
                       current_season_games=int(len(current)))
    except RateLimitedError as e:
        summary.update(ok=False, error=f"rate_limited: {e}", rate_limited=True)
    except Exception as e:  # noqa: BLE001 — status file is the observability channel
        summary.update(ok=False, error=str(e))
    summary["finished_at"] = _now_iso()
    try:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(summary, status_path)
    except OSError:
        pass
    return summary
