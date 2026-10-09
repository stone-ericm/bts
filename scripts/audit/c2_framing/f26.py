"""C2 side item (e): the catcher-grouped framing measure's 2026 out-of-sample test (design
`docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md`, frozen at `9b344d9` "at its reviewed commit plus the
verbatim r1 diff, confirmed by r2"; register rows C2-framing-2026-test, Eric 2026-10-09 "Option 1", and
C2-framing-2026-design-frozen).

**What it measures.** Variant A (the screen's catcher-grouped `catcher_framing` replacing `pitcher_catcher_framing`)
against the screen-recipe baseline on 2026, in three arms per seed:
- `baseline`: the production feature set, as in the screen;
- `A_posted` (decides): each predicted day's catcher is the final-boxscore starter proxy of the opposing fielding side;
- `A_projected` (stress arm): it is the opposing team's most-used identified starter proxy in its last 10 earlier-dated
  games.
Training rows keep the screen's build-keyed feature; only the predicted day's rows carry the arm's catcher (§4), through
`blend_walk_forward(..., predict_day_transform=...)`.

**The rules here** are the design's §3 (the starter proxy and the lookup entry), §4 (the projection, the as-of value,
the missing-catcher reasons) and §6 (the calendar and the dispositions). The run, its admission and the aggregate are
below them.

Research code: it changes nothing in production.
"""
from __future__ import annotations

import math
import re
from datetime import date as _date

import numpy as np

SEEDS = (2273360, 260991262, 1746737973, 2048, 3629294338, 1277948386, 3219332220, 2207587974, 3170105529,
         2675988121)                               # §5: the literal order; no seed-set file is read
TEST_SEASON = 2026
ARMS = ("baseline", "A_posted", "A_projected")
NEW_COL = "catcher_framing"
MIN_RATES = 5                                       # §4: five nonmissing daily rates
PROJECTION_WINDOW = 10                              # §4: the last 10 earlier-dated games
PRACTICAL_MIN = 0.003                               # §6: +0.3pp
SEEDS_POSITIVE_MIN = 6                              # §6: d > 0 on at least 6 of 10 seeds
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261009
BLOCK_DAYS = 7
SLOTS = frozenset(f"{k}00" for k in range(1, 10))   # §3: battingOrder exactly "100" .. "900"
SIDES = ("away", "home")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class InputRefused(RuntimeError):
    """An input breaks a registered rule: admission refuses rather than repairs (§3)."""


# ---------------------------------------------------------------- §3: the starter proxy and the lookup entry

def _pos_int(v) -> bool:
    return type(v) is int and v > 0                  # no bool, float or string coercion


def _game_fields(feed, game_pk: int, season: int) -> dict:
    try:
        gd = feed["gameData"]
        game, when, teams = gd["game"], gd["datetime"], gd["teams"]
    except (KeyError, TypeError):
        raise InputRefused(f"game {game_pk}: no gameData game, datetime or teams")
    if not (isinstance(game, dict) and _pos_int(game.get("pk")) and game["pk"] == game_pk):
        raise InputRefused(f"game {game_pk}: gameData.game.pk is {game.get('pk') if isinstance(game, dict) else None!r}")
    s = game.get("season")
    if not ((type(s) is int and s == season) or (isinstance(s, str) and s.isdigit() and int(s) == season)):
        raise InputRefused(f"game {game_pk}: season {s!r} is not {season}")
    if game.get("type") != "R":
        raise InputRefused(f"game {game_pk}: type {game.get('type')!r} is not a regular-season game")
    d = when.get("officialDate") if isinstance(when, dict) else None
    try:
        ok = isinstance(d, str) and bool(ISO_DATE.fullmatch(d)) and _date.fromisoformat(d).year == season
    except ValueError:
        ok = False
    if not ok:
        raise InputRefused(f"game {game_pk}: officialDate {d!r} is malformed")
    if "gameNumber" in game:
        gn, fallback = game["gameNumber"], False
        if not _pos_int(gn):
            raise InputRefused(f"game {game_pk}: gameNumber {gn!r} is malformed")
    else:
        gn, fallback = 1, True
    tids = {}
    for side in SIDES:
        t = teams.get(side) if isinstance(teams, dict) else None
        tid = t.get("id") if isinstance(t, dict) else None
        if not _pos_int(tid):
            raise InputRefused(f"game {game_pk}: the {side} team id {tid!r} is malformed")
        tids[side] = tid
    return {"game_pk": game_pk, "season": season, "game_type": "R", "official_date": d, "game_number": gn,
            "game_number_fallback": fallback, "team_ids": tids}


def _side_proxy(feed, side: str) -> tuple[int | None, str]:
    """The side's starter proxy and its identification reason (§3)."""
    try:
        players = feed["liveData"]["boxscore"]["teams"][side]["players"]
    except (KeyError, TypeError):
        return None, "malformed_player"
    if not isinstance(players, dict):
        return None, "malformed_player"
    bad_player = bad_positions = False
    candidates = []
    for p in players.values():
        if not isinstance(p, dict):
            bad_player = True
            continue
        if "battingOrder" not in p:
            continue                                    # not in the batting order: not examined
        bo = p["battingOrder"]
        if not isinstance(bo, str):
            bad_player = True
            continue
        if bo not in SLOTS:
            continue                                    # a substitute's slot ("201") or another string
        person = p.get("person")
        pid = person.get("id") if isinstance(person, dict) else None
        if not _pos_int(pid):
            bad_player = True
            continue
        ap = p.get("allPositions")
        if not (isinstance(ap, list) and ap and all(isinstance(e, dict) and isinstance(e.get("code"), str) for e in ap)):
            bad_positions = True
            continue
        if ap[0]["code"] == "2":                        # the array's given order; no later C entry, no fallback
            candidates.append(pid)
    if bad_player:
        return None, "malformed_player"
    if bad_positions:
        return None, "malformed_positions"
    if len(candidates) == 1:
        return candidates[0], "identified"
    return None, "no_candidate" if not candidates else "multiple_candidates"


def starter_proxy(feed, game_pk: int, season: int) -> list[dict]:
    """Exactly one record per (game_pk, fielding_side), including unidentified sides (§3). The fielding side is the
    side whose catcher it is: its catcher fields while the other side bats. Malformed game fields refuse."""
    g = _game_fields(feed, game_pk, season)
    out = []
    for side in SIDES:
        cid, why = _side_proxy(feed, side)
        out.append({k: g[k] for k in ("game_pk", "season", "game_type", "official_date", "game_number",
                                      "game_number_fallback")}
                   | {"fielding_side": side, "team_id": g["team_ids"][side], "catcher_id": cid, "reason": why})
    return out


def lookup_entry(feed, game_pk: int) -> dict:
    """Production's probable-pitcher lookup entry for one game (`bts.features.compute._build_probable_pitcher_lookup`):
    the probable pitchers from `gameData.probablePitchers` and the team ids from `gameData.teams` (§3)."""
    gd = feed.get("gameData", {}) if isinstance(feed, dict) else {}
    game = gd.get("game", {}) if isinstance(gd, dict) else {}
    if not (isinstance(game, dict) and _pos_int(game.get("pk")) and game["pk"] == game_pk):
        raise InputRefused(f"game {game_pk}: gameData.game.pk does not match")
    teams, pp = gd.get("teams", {}), gd.get("probablePitchers", {})
    return {"away": pp.get("away", {}).get("id"), "home": pp.get("home", {}).get("id"),
            "away_tid": teams.get("away", {}).get("id"), "home_tid": teams.get("home", {}).get("id")}


# ---------------------------------------------------------------- §4: the projection

def _iso(d) -> str:
    import pandas as pd
    return pd.Timestamp(d).date().isoformat()


class TeamHistory:
    """Each team's table games in history order `(officialDate, game_number, game_pk)`, home and away, across seasons."""

    def __init__(self, table):
        self._games: dict[int, list[tuple]] = {}
        for r in table.itertuples(index=False):
            cid = None if r.catcher_id is None or (isinstance(r.catcher_id, float) and math.isnan(r.catcher_id)) \
                else int(r.catcher_id)
            self._games.setdefault(int(r.team_id), []).append(
                ((str(r.official_date), int(r.game_number), int(r.game_pk)), cid))
        for v in self._games.values():
            v.sort(key=lambda e: e[0])
        self._cache: dict[tuple, int | None] = {}

    def project(self, team_id, predicted_date) -> int | None:
        """The most frequent identified catcher among the team's last 10 games dated strictly before the predicted
        date (fewer if fewer exist); the window is taken first and unidentified games are not replaced; a tie goes to
        the tied catcher whose latest start in the window has the greatest history-order tuple; None if no identified
        catcher remains."""
        key = (int(team_id), _iso(predicted_date))
        if key not in self._cache:
            games = [g for g in self._games.get(key[0], []) if g[0][0] < key[1]]
            window = games[-PROJECTION_WINDOW:]
            count: dict[int, int] = {}
            latest: dict[int, tuple] = {}
            for order, cid in window:
                if cid is not None:
                    count[cid] = count.get(cid, 0) + 1
                    latest[cid] = max(latest.get(cid, order), order)
            if not count:
                self._cache[key] = None
            else:
                top = max(count.values())
                self._cache[key] = max((c for c in count if count[c] == top), key=lambda c: latest[c])
        return self._cache[key]


# ---------------------------------------------------------------- §4: the as-of value

class AsOfFraming:
    """The catcher-grouped feature for a named catcher as of a predicted date: the screen's build-keyed daily rates
    (mean `pa_borderline_csr` per `(fielding_catcher_id, date)`, missing PA rates ignored; both games of a doubleheader
    share one date), restricted to dates strictly before the predicted date, with `expanding(min_periods=5).mean()`'s
    last value (NaN for an empty prefix or fewer than five nonmissing daily rates). Pandas' expanding mean at a position
    depends only on that position's prefix, so each catcher's series is computed once."""

    def __init__(self, df):
        import pandas as pd
        daily = df.groupby(["fielding_catcher_id", "date"])["pa_borderline_csr"].mean().sort_index()
        self._by: dict[int, tuple] = {}
        for cid, s in daily.groupby(level=0):
            s = s.droplevel(0)
            dates = pd.to_datetime(s.index).values.astype("datetime64[ns]")
            values = s.expanding(min_periods=MIN_RATES).mean().to_numpy(dtype=float)
            self._by[int(cid)] = (dates, values)

    def value(self, catcher_id, predicted_date) -> float:
        import pandas as pd
        if catcher_id is None or (isinstance(catcher_id, float) and math.isnan(catcher_id)):
            return math.nan
        hit = self._by.get(int(catcher_id))
        if hit is None:
            return math.nan
        dates, values = hit
        k = int(np.searchsorted(dates, pd.Timestamp(predicted_date).to_datetime64().astype("datetime64[ns]"),
                                side="left"))
        return float(values[k - 1]) if k > 0 else math.nan


# ---------------------------------------------------------------- §4: the arm transform

REASONS = ("identified", "no_catcher", "too_few_rates")


class ArmTransform:
    """The predicted day's `catcher_framing` under an A arm; every other column, the index and the order unchanged.
    Batter rows with `is_home` face the away side's catcher, the others the home side's. Counts each mutually exclusive
    missing reason per distinct opposing side-game and per predicted PA row (§4)."""

    def __init__(self, arm: str, table, asof: AsOfFraming):
        if arm not in ("A_posted", "A_projected"):
            raise ValueError(f"not an A arm: {arm!r}")
        self.arm, self.asof = arm, asof
        self._sides = {(int(r.game_pk), str(r.fielding_side)): r for r in table.itertuples(index=False)}
        self._history = TeamHistory(table) if arm == "A_projected" else None
        self.counts = {"side_games": dict.fromkeys(REASONS, 0), "pa_rows": dict.fromkeys(REASONS, 0)}
        self.identified_ids: set[int] = set()

    def _catcher(self, rec, day):
        if self.arm == "A_posted":
            c = rec.catcher_id
            return None if c is None or (isinstance(c, float) and math.isnan(c)) else int(c)
        return self._history.project(rec.team_id, day)

    def __call__(self, day_data, day):
        out = day_data.copy()
        sides = np.where(out["is_home"].astype(bool).to_numpy(), "away", "home")
        keys = list(zip(out["game_pk"].astype(int).tolist(), sides.tolist()))
        values = np.full(len(out), math.nan)
        resolved = {}
        for key in dict.fromkeys(keys):
            rec = self._sides.get(key)
            if rec is None:
                raise InputRefused(f"side-game {key} is not in the starter-proxy table")
            cid = self._catcher(rec, day)
            v = self.asof.value(cid, day) if cid is not None else math.nan
            why = "no_catcher" if cid is None else ("too_few_rates" if math.isnan(v) else "identified")
            if why == "identified":
                self.identified_ids.add(cid)
            self.counts["side_games"][why] += 1
            resolved[key] = (v, why)
        for i, key in enumerate(keys):
            v, why = resolved[key]
            values[i] = v
            self.counts["pa_rows"][why] += 1
        out[NEW_COL] = values
        return out


# ---------------------------------------------------------------- §5–§6: the calendar and the dispositions

def expected_calendar(pa) -> list[str]:
    """The sorted official dates with at least one scoreable (original-portion) batter-game, from the pinned 2026 rows.
    A boolean `is_resumed_portion` with no missing values is required (§5): an absent or malformed flag refuses."""
    import pandas as pd
    if "is_resumed_portion" not in pa.columns:
        raise InputRefused("pa_2026 has no is_resumed_portion column")
    flag = pa["is_resumed_portion"]
    if not pd.api.types.is_bool_dtype(flag) or bool(flag.isna().any()):
        raise InputRefused("pa_2026's is_resumed_portion is not a boolean column without missing values")
    kept = pa.loc[~flag.to_numpy(dtype=bool)]
    return sorted({_iso(d) for d in pd.to_datetime(kept["date"]).unique()})


def _quantile_draws(daily: np.ndarray, idx: np.ndarray) -> tuple[float, bool]:
    means = daily[idx].mean(axis=1)
    return float(np.quantile(means, 0.1, method="linear")), bool(np.ptp(means) == 0)


def dispose(x) -> dict:
    """§6 over x[s, t] = A-posted's rank-1 hit minus the baseline's, seeds in §5's order, dates ascending over the
    complete calendar. Incomplete takes precedence: anything but ten seeds over a nonempty calendar of finite deltas."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(SEEDS) or x.shape[1] == 0 or not bool(np.isfinite(x).all()):
        return {"disposition": "incomplete", "shape": list(x.shape)}
    n = x.shape[1]
    d = x.mean(axis=1)
    m = float(d.mean())
    daily = x.mean(axis=0)
    L, constant = _quantile_draws(daily, np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_DRAWS, n)))
    starts = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_DRAWS, math.ceil(n / BLOCK_DAYS)))
    block = ((starts[:, :, None] + np.arange(BLOCK_DAYS)) % n).reshape(BOOTSTRAP_DRAWS, -1)[:, :n]
    L_block, _ = _quantile_draws(daily, block)
    positive_seeds = int((d > 0).sum())
    if m >= PRACTICAL_MIN and L > 0 and positive_seeds >= SEEDS_POSITIVE_MIN:
        verdict = "positive"
    elif m <= 0:
        verdict = "negative"
    else:
        verdict = "inconclusive"
    return {"disposition": verdict, "m": m, "L": L, "L_block7": L_block, "d": d.tolist(),
            "seeds_positive": positive_seeds, "n_days": n, "bootstrap_constant": constant,
            "daily": {"positive": int((daily > 0).sum()), "negative": int((daily < 0).sum()),
                      "zero": int((daily == 0).sum()), "discordant_seed_days": int((x != 0).sum())}}


# ================================================================ the run, its admission and the aggregate

import argparse                                     # noqa: E402
import io                                           # noqa: E402
import json                                         # noqa: E402
import os                                           # noqa: E402
import subprocess                                   # noqa: E402
import sys                                          # noqa: E402
import time                                         # noqa: E402
from datetime import datetime, timezone             # noqa: E402
from pathlib import Path                            # noqa: E402

from scripts.audit.c1 import ledger                 # noqa: E402
from scripts.audit.c2_framing import screen as S    # noqa: E402

SEASONS_IN = tuple(range(2017, 2027))
PROXY_SEASONS = (2025, 2026)
LOOKUP_NAME = "probable_pitcher_lookup.2017-2026.json"
TABLE_NAME = "starter_proxy.2025-2026.json"
SOURCES_NAME = "raw_sources.2025-2026.json"
FROZEN_PA = "pa_2026.parquet"                       # the frozen copy, read from the inputs directory
INPUT_NAMES = tuple(f"pa_{s}.parquet" for s in SEASONS_IN) + (LOOKUP_NAME, TABLE_NAME, SOURCES_NAME)
OUT_ROOT = Path.home() / "projects" / "bts" / "data" / "hetzner_results" / "c2" / "framing_2026"
ADMISSION_REL = "scripts/audit/c2_framing/admission_2026.json"
REGISTER_REL = "docs/audit/2026-09-22-exposure-register.md"
DESIGN = "docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md"
EXPOSURE_ROW = "X-37"
SCOPE = "catcher framing 2026 test"
INPUTS_ROW = "C2-framing-2026-inputs"
ALLOWANCE_ROW = "C2-framing-2026-allowance"
PREP_ROW = "C2-framing-2026-prep-read"
CLOSURE = ("scripts/__init__.py", "scripts/audit/__init__.py", "scripts/audit/c1", "scripts/audit/c2_framing",
           "src/bts", "pyproject.toml", "uv.lock", DESIGN)
QUIET_MINUTES = (45, 190)                           # no launch from 00:45 to 03:10 America/New_York
MAX_WALL_H = 16
C1_DIR = OUT_ROOT.parents[1] / "c1"
UNIT_RE = re.compile(r"c1-c2-f26-seed(\d+)-\d{8}T\d{6}Z-[0-9a-f]{8}")
UNIT_ORDER = [(a, TEST_SEASON) for a in ARMS]
IDENTITY_KEYS = ("review_report", "review_report_sha256", "reviewed_commit", "exposure_commit", "admission_sha256")
HEX = re.compile(r"[0-9a-f]{64}")
HEAD = re.compile(r"[0-9a-f]{40}")
INPUTS_RE = re.compile(r"^\*\*PINNED (\d{4}-\d{2}-\d{2}): catcher framing 2026 test inputs `([0-9a-f]{64})`\*\*$")
PREP_RE = re.compile(r"^\*\*DECLARED (\d{4}-\d{2}-\d{2}): the catcher framing 2026 test's preparation read\*\*$")
ALLOW_RE = re.compile(r"^\*\*RULED (\d{4}-\d{2}-\d{2}) \(Eric\): ALLOW the catcher framing 2026 test; shared C1/C2 "
                      r"compute cap (\d+(?:\.\d+)?) CPU-hours; declared budget (\d+(?:\.\d+)?) CPU-hours per seed; "
                      r"first walk-forward stop (\d+(?:\.\d+)?) CPU-hours\*\*$")

RunInvalid = S.RunInvalid


# ---------------------------------------------------------------- admission

def pins_shape_problem(pins) -> str | None:
    if not (isinstance(pins, dict) and set(pins) == set(INPUT_NAMES)
            and all(isinstance(v, str) and HEX.fullmatch(v) for v in pins.values())):
        return f"admission input_pins must give a sha256 for exactly {sorted(INPUT_NAMES)}"
    return None


def _row_after_exposure(repo: Path, register_text: str, exposure_commit: str, row_id: str, pattern) -> tuple:
    """(the ruling cell's match, problem): the row must be structured, and absent at the exposure commit, so it was
    recorded after X-37."""
    from scripts.audit.c1 import admission as A
    cells = A.row_cells(register_text, row_id)
    m = pattern.match(cells[2].strip()) if cells and len(cells) >= 4 else None
    if not m:
        return None, f"no structured {row_id} row"
    earlier = A._git(repo, "show", f"{exposure_commit}:{REGISTER_REL}", check=False).stdout
    if A.row_cells(earlier, row_id):
        return None, f"the {row_id} row already existed at the exposure commit: it is recorded after X-37"
    return m, None


def inputs_row_problem(repo: Path, register_text: str, exposure_commit: str, pins: dict) -> str | None:
    """The pins are bound in the register after X-37 (§8 step 3): row INPUTS_ROW, absent at the exposure commit, whose
    ruling cell is exactly "**PINNED <date>: catcher framing 2026 test inputs `<sha256 of the canonical pins>`**"."""
    m, problem = _row_after_exposure(repo, register_text, exposure_commit, INPUTS_ROW, INPUTS_RE)
    if problem:
        return problem if "already existed" in problem else f"{problem} binding the input pins"
    if m.group(2) != S.pins_digest(pins):
        return f"the {INPUTS_ROW} row does not bind these input pins"
    return None


def prep_row_problem(repo: Path, register_text: str, exposure_commit: str) -> str | None:
    """The first 2026 read has its own register row, recorded after X-37 (the manager's gate, 2026-10-09): row
    PREP_ROW whose ruling cell is exactly "**DECLARED <date>: the catcher framing 2026 test's preparation read**"."""
    return _row_after_exposure(repo, register_text, exposure_commit, PREP_ROW, PREP_RE)[1]


def admission_gate(repo: Path | None = None, *, require_inputs: bool = True):
    """X-37 (published, unchanged, binding the reviewed code and its review), an unchanged executable closure, and,
    once the inputs are prepared, the pins bound by the inputs row. Returns (head, admission, identity)."""
    from scripts.audit.c1 import admission as A
    repo = A.REPO if repo is None else repo
    path = repo / ADMISSION_REL
    if not path.is_file():
        raise SystemExit(f"refusing: no admission record at {ADMISSION_REL}")
    raw = path.read_bytes()
    adm = json.loads(raw)
    head, reasons = A.admission_check(repo, adm, closure=CLOSURE, admission_rel=ADMISSION_REL,
                                      register_rel=REGISTER_REL, exposure_row=EXPOSURE_ROW, scope_phrase=SCOPE)
    if require_inputs and not reasons:
        pins = adm.get("input_pins")
        problem = pins_shape_problem(pins) or inputs_row_problem(repo, (repo / REGISTER_REL).read_text(),
                                                                 adm["exposure_commit"], pins)
        if problem:
            reasons.append(problem)
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head, adm, {**A.accepted_identity(repo, adm), "admission_sha256": S._sha(raw)}


def allowance(register_text: str) -> dict | None:
    """Eric's compute allowance (row ALLOWANCE_ROW): the ruling cell must match exactly and the source cell's first token
    must be exactly `Eric`; every number finite and positive, the stop below the per-seed budget."""
    from scripts.audit.c1 import admission as A
    cells = A.row_cells(register_text, ALLOWANCE_ROW)
    if not cells or len(cells) < 4:
        return None
    m = ALLOW_RE.match(cells[2].strip())
    if not m or cells[3].split()[:1] != ["Eric"]:
        return None
    cap, budget, stop = float(m.group(2)), float(m.group(3)), float(m.group(4))
    if not all(math.isfinite(v) and v > 0 for v in (cap, budget, stop)) or stop >= budget:
        return None
    return {"cap": cap, "budget": budget, "first_unit_stop": stop}


def head_admitted(repo: Path, identity: dict, head: str) -> list[str]:
    """`screen.head_admitted` with this test's closure and admission record."""
    from scripts.audit.c1 import admission as A
    rc, xc = identity.get("reviewed_commit"), identity.get("exposure_commit")
    if not all(isinstance(c, str) and HEAD.fullmatch(c) for c in (rc, xc, head)):
        return [f"run HEAD {str(head)[:7]} is not admitted: the head and the admitted commits must be full commit ids"]
    if A._git(repo, "cat-file", "-e", f"{head}^{{commit}}", check=False).returncode != 0:
        return [f"run HEAD {head[:7]} is not admitted: not a commit in this repository"]
    if not A._ancestor(repo, xc, head):
        return [f"run HEAD {head[:7]} is not admitted: the exposure commit is not its ancestor"]
    changed = [f for f in A._git(repo, "diff", "--name-only", rc, head, "--", *CLOSURE).stdout.split()
               if f != ADMISSION_REL]
    return [f"run HEAD {head[:7]} is not admitted: executable files differ from the reviewed commit: {changed[:5]}"] \
        if changed else []


def seed_allowed(seed: int, register_text: str, out_root: Path, *, identity: dict, pins: dict,
                 calendar: list[str], repo: Path | None = None) -> tuple[bool, str, dict | None]:
    """Eric's allowance, whose cap is the launcher's; then seeds one at a time in §5's order: every earlier seed holds
    exactly one complete admitted run. Returns (allowed, reason, allowance)."""
    if seed not in SEEDS:
        return False, f"{seed} is not a registered seed {SEEDS}", None
    allow = allowance(register_text)
    if allow is None:
        return False, f"the test needs Eric's compute allowance (register row {ALLOWANCE_ROW})", None
    if allow["cap"] != ledger.CAP_H:
        return False, f"Eric's cap {allow['cap']:g} is not the launcher's cap {ledger.CAP_H:g}", None
    for s in SEEDS[:SEEDS.index(seed)]:
        root = out_root / f"seed_{s}"
        runs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
        if len(runs) != 1:
            return False, f"seeds run one at a time, in order: earlier seed {s} has {len(runs)} runs, not one", None
        try:
            validate_run(runs[0], s, out_root=out_root, identity=identity, pins=pins, calendar=calendar, repo=repo)
        except RunInvalid as e:
            return False, f"seeds run in order: earlier seed {s}'s run is not a complete admitted run: {e}", None
    return True, "allowed", allow


def launch_window_problem(hour: int, minute: int) -> str | None:
    if QUIET_MINUTES[0] <= hour * 60 + minute < QUIET_MINUTES[1]:
        return "no launch from 00:45 to 03:10 America/New_York (production's 03:00 chain)"
    return None


def refuse_inside_the_window() -> None:
    problem = launch_window_problem(*S.ny_clock())
    if problem:
        raise SystemExit(f"refusing: {problem}")


def guarded_unit_problem(seed: int, budget: float, cgroup_text: str, c1_dir: Path) -> str | None:
    """`screen.guarded_unit_problem` for this test's unit names: the kernel's cgroup record places this process in
    `<unit>.service/payload` of a unit the launcher names for this seed, and that unit's PENDING record declares this
    seed's budget. Fails closed."""
    path = next((x[3:].strip() for x in cgroup_text.splitlines() if x.startswith("0::")), None)
    parts = path.split("/") if path else []
    unit = parts[-2].removesuffix(".service") if len(parts) >= 2 and parts[-2].endswith(".service") else None
    m = UNIT_RE.fullmatch(unit) if unit else None
    if not (parts and parts[-1] == "payload" and m and int(m.group(1)) == SEEDS.index(seed) + 1):
        return f"seeds run only as the guarded payload of their own C1 launcher unit (cgroup {path!r})"
    try:
        rec = json.loads((c1_dir / f"PENDING_{unit}.json").read_text())
    except (OSError, ValueError):
        return f"no launcher record PENDING_{unit}.json for this C1 launcher unit"
    if not (isinstance(rec, dict) and rec.get("unit") == unit and rec.get("declared_cpu_hours") == budget):
        return f"the launcher record for {unit} does not name this unit with this seed's budget {budget:g}"
    return None


# ---------------------------------------------------------------- inputs

def read_pinned_json(path: Path, sha: str):
    from scripts.audit.c1 import admission as A
    raw, _ = A.read_pinned(path, sha)
    return json.loads(raw)


def load_inputs(data_dir: Path, inputs_dir: Path, pins: dict):
    """The ten PA parquets from their pinned bytes: 2017-2025 from the data directory (the screen's pins), 2026 from the
    frozen copy in the inputs directory. Nothing else in either directory is read."""
    import pandas as pd
    from scripts.audit.c1 import admission as A
    frames = []
    for s in SEASONS_IN:
        name = f"pa_{s}.parquet"
        raw, _ = A.read_pinned((inputs_dir if name == FROZEN_PA else data_dir) / name, pins[name])
        frames.append(pd.read_parquet(io.BytesIO(raw)))
    return pd.concat(frames, ignore_index=True)


def table_frame(records: list) -> "pd.DataFrame":
    import pandas as pd
    t = pd.DataFrame(records)
    if t.duplicated(["game_pk", "fielding_side"]).any():
        raise InputRefused("duplicate (game_pk, fielding_side) records in the starter-proxy table")
    return t


def canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def prepare(data_dir: Path, raw_dir: Path, screen_inputs: Path, screen_pins: dict, out_dir: Path) -> dict:
    """The preparation read (§3, §8 step 3), once, after X-37 and its own register row: the frozen copy of `pa_2026`,
    the 2026 lookup entries and the 2025-2026 starter-proxy table from exactly the raw feeds of the game ids in the
    pinned 2025 and 2026 PA files, and the source manifest. Only identity and schedule fields are extracted. Returns
    the pins of all inputs and the counts."""
    import pandas as pd
    from scripts.audit.c1 import admission as A
    out_dir.mkdir(exist_ok=False)
    raw26 = (data_dir / FROZEN_PA).read_bytes()
    A.durable_write(out_dir / FROZEN_PA, raw26)
    pins = {f"pa_{s}.parquet": screen_pins[f"pa_{s}.parquet"] for s in SEASONS_IN if s != TEST_SEASON}
    pins[FROZEN_PA] = S._sha(raw26)
    games = {}
    for s in PROXY_SEASONS:
        raw = raw26 if s == TEST_SEASON else A.read_pinned(data_dir / f"pa_{s}.parquet", pins[f"pa_{s}.parquet"])[0]
        pa = pd.read_parquet(io.BytesIO(raw), columns=["game_pk", "season"])
        if not bool((pa["season"] == s).all()):
            raise InputRefused(f"pa_{s} holds rows of another season")
        games[s] = sorted({int(g) for g in pa["game_pk"]})
    lookup = read_pinned_json(screen_inputs / S.LOOKUP_NAME, screen_pins[S.LOOKUP_NAME])
    records, sources = [], []
    for s in PROXY_SEASONS:
        for pk in games[s]:
            rel = f"{s}/{pk}.json"
            path = raw_dir / rel
            if not path.is_file():
                raise InputRefused(f"no raw feed {rel} for a pinned {s} game")
            b = path.read_bytes()
            sources.append({"path": rel, "sha256": S._sha(b), "game_pk": pk, "season": s})
            feed = json.loads(b)
            records += starter_proxy(feed, pk, s)
            if s == TEST_SEASON:
                if str(pk) in lookup:
                    raise InputRefused(f"2026 game {pk} is already in the screen's frozen lookup")
                lookup[str(pk)] = lookup_entry(feed, pk)
    table_frame(records)
    records.sort(key=lambda r: (r["season"], r["official_date"], r["game_number"], r["game_pk"], r["fielding_side"]))
    manifest = {"sources": sources, "counts": {str(s): len(games[s]) for s in PROXY_SEASONS},
                "digest": S._sha(canonical(sources))}
    files = {LOOKUP_NAME: canonical(dict(sorted(lookup.items(), key=lambda kv: int(kv[0])))),
             TABLE_NAME: canonical(records), SOURCES_NAME: canonical(manifest)}
    for name, b in files.items():
        A.durable_write(out_dir / name, b)
        pins[name] = S._sha(b)
    reasons = {}
    for r in records:
        reasons.setdefault(str(r["season"]), {}).setdefault(r["reason"], 0)
        reasons[str(r["season"])][r["reason"]] += 1
    missing_pitchers = sum(1 for s in (TEST_SEASON,) for pk in games[s]
                           for k in ("away", "home") if lookup[str(pk)][k] is None)
    return {"pins": dict(sorted(pins.items())), "pins_digest": S.pins_digest(pins), "games": manifest["counts"],
            "proxy_reasons": reasons, "game_number_fallbacks": sum(r["game_number_fallback"] for r in records),
            "lookup_2026_missing_probable_sides": missing_pitchers}


# ---------------------------------------------------------------- the admitted run

def rank1(profiles, calendar: list[str]) -> list[int]:
    """The rank-1 hit on every calendar date, in date order; exactly one rank-1 row per date, no other date."""
    import pandas as pd
    top = profiles[profiles["rank"] == 1]
    dates = [_iso(d) for d in pd.to_datetime(top["date"])]
    if sorted(dates) != calendar or len(set(dates)) != len(dates):
        raise RunInvalid("the rank-1 rows are not exactly one per calendar date")
    by = dict(zip(dates, top["actual_hit"].astype(int).tolist()))
    return [by[d] for d in calendar]


def run(seed: int, data_dir: Path, inputs_dir: Path, *, walk_forward=None, now=None, _test_out_root=None) -> int:
    import pandas as pd
    from scripts.audit.c1 import admission as A
    out_root = OUT_ROOT if _test_out_root is None else _test_out_root
    if os.environ.get("BTS_LGBM_DETERMINISTIC") != "1":
        raise SystemExit("refusing: BTS_LGBM_DETERMINISTIC=1 is pre-registered (set before bts is imported)")
    if seed not in SEEDS:
        raise SystemExit(f"refusing: {seed} is not a registered seed {SEEDS}")
    head, adm, identity = admission_gate()
    foreign = A.foreign_imports()
    if foreign:
        raise SystemExit(f"refusing: modules from outside this checkout: {foreign}")
    pins = adm["input_pins"]
    raw26, _ = A.read_pinned(inputs_dir / FROZEN_PA, pins[FROZEN_PA])
    calendar = expected_calendar(pd.read_parquet(io.BytesIO(raw26)))
    register = (A.REPO / REGISTER_REL).read_text()
    ok, why, allow = seed_allowed(seed, register, out_root, identity=identity, pins=pins, calendar=calendar)
    if not ok:
        raise SystemExit(f"refusing: {why}")
    problem = guarded_unit_problem(seed, allow["budget"], S.proc_cgroup_text(), C1_DIR)
    if problem:
        raise SystemExit(f"refusing: {problem}")
    from bts.model.predict import LGB_PARAMS
    if not (LGB_PARAMS.get("deterministic") is True and LGB_PARAMS.get("force_row_wise") is True):
        raise SystemExit("refusing: LightGBM's params were built without the deterministic flags")
    settings = S.check_settings()
    os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
    root = out_root / f"seed_{seed}"
    with A.admission_lock(root):
        if A.claimed_runs(root, register):
            raise SystemExit(f"refusing: a claimed run already exists under {root}")
        refuse_inside_the_window()                 # the last check before the claim
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        run_dir = A.make_run_dir(root, f"{head[:7]}-{stamp}")
        claim_sha = A.write_claim(run_dir, head)
    S.install_closed_inputs(S.frozen_lookup(A.read_pinned(inputs_dir / LOOKUP_NAME, pins[LOOKUP_NAME])[0]))
    table = table_frame(read_pinned_json(inputs_dir / TABLE_NAME, pins[TABLE_NAME]))
    from bts.features.compute import compute_all_features
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
    walk_forward = walk_forward or blend_walk_forward
    t0 = S.cpu_seconds()
    df = compute_all_features(load_inputs(data_dir, inputs_dir, pins))
    check = S.self_check(df)
    df = S.framing_by(df, "fielding_catcher_id", NEW_COL)
    labels = S.original_portion_labels(df)
    asof = AsOfFraming(df)
    manifest = {"schema": "c2_framing_2026_run_v1", "head": head, "claim_sha256": claim_sha, "seed": seed,
                "identity": identity, "input_pins": pins, "inputs_digest": S.pins_digest(pins),
                "test_season": TEST_SEASON, "arms": list(ARMS), "basis": S.BASIS, "retrain_every": S.RETRAIN_EVERY,
                "lgb_params": dict(LGB_PARAMS), "feature_settings": settings, "scoring": dict(S.SCORING),
                "env": {k: os.environ.get(k) for k in ("BTS_LGBM_DETERMINISTIC", "BTS_LGBM_RANDOM_STATE", "TZ")},
                "self_check": check, "calendar": calendar, "allowance": allow,
                "resumed_portion_rows": S.resumed_counts(df), "features_cpu_s": S.cpu_seconds() - t0}
    S._json(run_dir / "manifest.json", manifest)
    if not check["identical"]:
        S._json(run_dir / "STOPPED.json", {"reason": "self_check", "self_check": check})
        return 4
    units, profiles, transforms = [], {}, {}
    for arm in ARMS:
        transform = None if arm == "baseline" else ArmTransform(arm, table, asof)
        configs = S.blend_configs("baseline" if arm == "baseline" else "A")
        c0, w0 = S.cpu_seconds(), time.monotonic()
        p = walk_forward(df, TEST_SEASON, retrain_every=S.RETRAIN_EVERY, blend_configs=configs,
                         game_probability_mode=S.BASIS, predict_day_transform=transform, strict_predict=True)
        cpu = S.cpu_seconds() - c0
        p, counts = S.relabel(p, labels)
        p["season"] = TEST_SEASON
        path = run_dir / f"profiles_{arm}_{TEST_SEASON}.parquet"
        p.to_parquet(path, index=False)
        unit = {"arm": arm, "season": TEST_SEASON, "cpu_s": cpu, "wall_s": time.monotonic() - w0, "labels": counts}
        if transform is not None:
            unit["catcher"] = {"counts": transform.counts, "identified_ids": len(transform.identified_ids)}
            transforms[arm] = unit["catcher"]
        units.append(unit)
        S._json(run_dir / "units.json", units)
        profiles[arm] = pd.read_parquet(path)       # score exactly the retained bytes
        if len(units) == 1 and cpu > allow["first_unit_stop"] * 3600:
            S._json(run_dir / "STOPPED.json", {"reason": "first_unit_cpu", "cpu_s": cpu,
                                               "limit_cpu_h": allow["first_unit_stop"]})
            return 3
    cards = {a: compute_full_scorecard(p, **S.SCORING) for a, p in profiles.items()}
    for a, c in cards.items():
        S._json(run_dir / f"scorecard_{a}.json", c)
    results = {"seed": seed, "head": head, "units": units, "total_cpu_s": S.cpu_seconds() - t0,
               "p_at_1": {a: c["p_at_1_by_season"] for a, c in cards.items()},
               "rank1": {a: rank1(p, calendar) for a, p in profiles.items()}, "arms": {}}
    for a in ARMS[1:]:
        diff = diff_scorecards(cards["baseline"], cards[a])
        S._json(run_dir / f"diff_{a}.json", diff)
        results["arms"][a] = arm_summary(diff)
    S._json(run_dir / "results.json", results)
    return 0


def arm_summary(diff: dict) -> dict:
    """An A arm's P@1 delta on 2026 and its reported streak metrics (no decision weight, §6)."""
    by = {str(k): v for k, v in (diff.get("p_at_1_by_season") or {}).items()}   # int keys before JSON, str after
    streak = diff.get("streak_metrics", {})
    return {"p_at_1_delta": float(by[str(TEST_SEASON)]["delta"]) if str(TEST_SEASON) in by else None,
            "p_57_exact": diff.get("p_57_exact"), "mean_max_streak": streak.get("mean_max_streak"),
            "longest_replay_streak": streak.get("longest_replay_streak")}


# ---------------------------------------------------------------- validation and the aggregate

def validate_run(d: Path, seed: int, *, out_root: Path, identity: dict, pins: dict, calendar: list[str],
                 repo: Path | None = None) -> dict:
    """One completed, claimed run, checked against trusted admission evidence (`identity`, `pins`) and the calendar
    derived from the pinned 2026 PA rows (never the run's own declarations). Raises RunInvalid; returns
    {manifest, results, rank1, summaries}. The checks follow `screen.validate_run`, for three arms on one season,
    plus: every retained profile has exactly one rank-1 row per calendar date and no other date; the rank-1 vectors,
    P@1, scorecards, diffs and summaries all reconcile with the retained profiles; and A-posted has at least one
    identified catcher value."""
    import pandas as pd
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
    from scripts.audit.c1 import admission as A
    repo = A.REPO if repo is None else repo
    d = Path(d)
    if d.resolve().parent != (out_root / f"seed_{seed}").resolve():
        raise RunInvalid(f"{d}: not in the canonical claim namespace {out_root / f'seed_{seed}'}")
    if (d / "STOPPED.json").exists():
        raise RunInvalid(f"{d}: a stopped run is not a complete seed")
    claim_bytes = S._read(d, "CLAIM.json")
    claim, man, res = S._json_of(d, "CLAIM.json"), S._json_of(d, "manifest.json"), S._json_of(d, "results.json")
    if not (isinstance(claim, dict) and claim.get("run") == d.name and isinstance(man, dict)
            and claim.get("code") == man.get("head")):
        raise RunInvalid(f"{d}: the claim does not name this run and its code")
    pins_m, lgb, ident, env = man.get("input_pins"), man.get("lgb_params") or {}, man.get("identity"), man.get("env") or {}
    checks = {
        "schema": man.get("schema") == "c2_framing_2026_run_v1",
        "seed": man.get("seed") == seed,
        "claim binding": man.get("claim_sha256") == S._sha(claim_bytes),
        "head": isinstance(man.get("head"), str) and bool(HEAD.fullmatch(man["head"])),
        "season": man.get("test_season") == TEST_SEASON,
        "arms": man.get("arms") == list(ARMS),
        "basis": man.get("basis") == S.BASIS,
        "retrain": man.get("retrain_every") == S.RETRAIN_EVERY,
        "settings": man.get("feature_settings") == S.SETTINGS,
        "scoring": man.get("scoring") == S.SCORING,
        "lgb determinism": lgb.get("deterministic") is True and lgb.get("force_row_wise") is True,
        "env": env.get("BTS_LGBM_DETERMINISTIC") == "1" and env.get("BTS_LGBM_RANDOM_STATE") == str(seed),
        "pins": pins_shape_problem(pins_m) is None,
        "pins digest": isinstance(pins_m, dict) and man.get("inputs_digest") == S.pins_digest(pins_m),
        "identity": isinstance(ident, dict) and set(ident) == set(IDENTITY_KEYS)
                    and all(isinstance(ident.get(k), str) and ident.get(k) for k in IDENTITY_KEYS),
        "self-check": (man.get("self_check") or {}).get("identical") is True,
        "admitted identity": isinstance(ident, dict) and {k: ident.get(k) for k in IDENTITY_KEYS}
                             == {k: identity.get(k) for k in IDENTITY_KEYS},
        "admitted pins": pins_m == pins,
        "calendar": man.get("calendar") == calendar,
        "results": isinstance(res, dict) and res.get("seed") == seed and res.get("head") == man.get("head"),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RunInvalid(f"{d}: invalid manifest/results: {bad}")
    problems = head_admitted(repo, identity, man["head"])
    if problems:
        raise RunInvalid(f"{d}: " + "; ".join(problems))
    units = res.get("units")
    if not (isinstance(units, list) and [(u.get("arm"), u.get("season")) for u in units] == UNIT_ORDER):
        raise RunInvalid(f"{d}: the three registered units are not all complete")
    posted = (units[1].get("catcher") or {}).get("counts", {}).get("pa_rows", {})
    if not (isinstance(posted.get("identified"), int) and posted["identified"] > 0):
        raise RunInvalid(f"{d}: A_posted has no identified catcher value: the catcher contrast is unavailable")
    cards = {a: S._json_of(d, f"scorecard_{a}.json") for a in ARMS}
    vectors = {}
    for a in ARMS:
        name = f"profiles_{a}_{TEST_SEASON}.parquet"
        S._read(d, name)
        try:
            part = pd.read_parquet(d / name)
            problem = S.profile_problem(part, TEST_SEASON)
        except Exception as e:
            raise RunInvalid(f"{d}: {name} is unreadable ({type(e).__name__})")
        if problem is not None:
            raise RunInvalid(f"{d}: {name} is not complete {TEST_SEASON} evidence: {problem}")
        try:
            vectors[a] = rank1(part, calendar)
        except RunInvalid as e:
            raise RunInvalid(f"{d}: {name}: {e}")
        try:
            card = json.loads(S._canon(compute_full_scorecard(part, **S.SCORING)))
        except Exception as e:
            raise RunInvalid(f"{d}: arm {a}'s retained profiles cannot be rescored ({type(e).__name__})")
        if S._canon({k: x for k, x in card.items() if k != "timestamp"}) != S._canon(
                {k: x for k, x in cards[a].items() if k != "timestamp"}):
            raise RunInvalid(f"{d}: arm {a}'s scorecard does not match a recomputation from its retained profiles")
        if sorted(card.get("p_at_1_by_season") or {}) != [str(TEST_SEASON)]:
            raise RunInvalid(f"{d}: arm {a}'s P@1 does not cover exactly {TEST_SEASON}")
        if (res.get("p_at_1") or {}).get(a) != card["p_at_1_by_season"]:
            raise RunInvalid(f"{d}: arm {a}'s P@1 does not reconcile across profiles, scorecard and results")
        if (res.get("rank1") or {}).get(a) != vectors[a]:
            raise RunInvalid(f"{d}: arm {a}'s rank-1 vector does not reconcile with its retained profile")
        p1 = card["p_at_1_by_season"][str(TEST_SEASON)]
        if not math.isclose(float(p1 if not isinstance(p1, dict) else p1.get("p_at_1")),
                            sum(vectors[a]) / len(calendar), rel_tol=0, abs_tol=1e-12):
            raise RunInvalid(f"{d}: arm {a}'s P@1 is not its rank-1 hits over the calendar")
    summaries = {}
    for a in ARMS[1:]:
        stored = S._json_of(d, f"diff_{a}.json")
        if S._canon(stored) != S._canon(json.loads(S._canon(diff_scorecards(cards["baseline"], cards[a])))):
            raise RunInvalid(f"{d}: arm {a}'s diff does not match its retained scorecards")
        summary = arm_summary(stored)
        if S._canon((res.get("arms") or {}).get(a)) != S._canon(summary):
            raise RunInvalid(f"{d}: arm {a}'s stored summary does not match its diff")
        summaries[a] = summary
    return {"manifest": man, "results": res, "rank1": vectors, "summaries": summaries}


def aggregate(run_dirs: list[Path], inputs_dir: Path, *, _test_out_root=None, _repo=None) -> dict:
    """§6: exactly one run directory per registered seed, each validated under the admitted identity, pins and the
    calendar re-derived here from the pinned `pa_2026`; all agree on everything that defines the run. A-posted decides;
    A-projected is descriptive."""
    import pandas as pd
    from scripts.audit.c1 import admission as A
    out_root = OUT_ROOT if _test_out_root is None else _test_out_root
    dirs = [Path(d).resolve() for d in run_dirs]
    if len(dirs) != len(SEEDS) or len(set(dirs)) != len(dirs):
        raise RunInvalid(f"need exactly {len(SEEDS)} distinct run directories, got {len(run_dirs)}")
    seeds = []
    for d in dirs:
        m = re.fullmatch(r"seed_(\d+)", d.parent.name)
        seeds.append(int(m.group(1)) if m else None)
    if None in seeds or sorted(seeds) != sorted(SEEDS):
        raise RunInvalid(f"seeds {seeds} are not exactly the registered {list(SEEDS)}")
    _, adm, identity = admission_gate(_repo)
    pins = adm["input_pins"]
    raw26, _ = A.read_pinned(Path(inputs_dir) / FROZEN_PA, pins[FROZEN_PA])
    calendar = expected_calendar(pd.read_parquet(io.BytesIO(raw26)))
    by = {s: validate_run(d, s, out_root=out_root, identity=identity, pins=pins, calendar=calendar, repo=_repo)
          for d, s in zip(dirs, seeds)}
    first = by[SEEDS[0]]["manifest"]
    for key in ("input_pins", "inputs_digest", "lgb_params", "feature_settings", "basis", "retrain_every",
                "test_season", "arms", "calendar", "scoring", "identity"):
        if any(by[s]["manifest"].get(key) != first.get(key) for s in SEEDS[1:]):
            raise RunInvalid(f"runs disagree on {key}")
    x = {a: np.array([[h - b for h, b in zip(by[s]["rank1"][a], by[s]["rank1"]["baseline"])] for s in SEEDS])
         for a in ARMS[1:]}
    return {"seeds": list(SEEDS), "heads": [by[s]["manifest"]["head"] for s in SEEDS], "identity": identity,
            "calendar_days": len(calendar), "total_cpu_h": sum(by[s]["results"]["total_cpu_s"] for s in SEEDS) / 3600,
            "A_posted": {"decides": True, **dispose(x["A_posted"])},
            "A_projected": {"decides": False, **dispose(x["A_projected"])},
            "per_seed": {str(s): {"summaries": by[s]["summaries"],
                                  "catcher": {u["arm"]: u.get("catcher") for u in by[s]["results"]["units"][1:]}}
                         for s in SEEDS}}


# ---------------------------------------------------------------- the launch wrapper and the command line

def launch_command(seed: int, budget: float, data_dir: Path, inputs_dir: Path) -> list[str]:
    k = SEEDS.index(seed) + 1
    return [".venv/bin/python", "-m", "scripts.audit.c1.launch", "run", "--name", f"c2-f26-seed{k}",
            "--cpu-hours", f"{budget:g}", "--max-hours", str(MAX_WALL_H), "--",
            "env", "BTS_LGBM_DETERMINISTIC=1", "TZ=America/New_York", ".venv/bin/python", "-m",
            "scripts.audit.c2_framing.f26", "run", "--seed", str(seed), "--data-dir", str(data_dir),
            "--inputs-dir", str(inputs_dir)]


def launch(seed: int, data_dir: Path, inputs_dir: Path, *, execute=subprocess.run, _test_out_root=None) -> int:
    """The reviewed launch: the same admission and seed-order checks as `run`, never from 00:45 to 03:10
    America/New_York, then the C1 launcher with Eric's per-seed budget (the launcher reserves it in full against the
    cap)."""
    import pandas as pd
    from scripts.audit.c1 import admission as A
    out_root = OUT_ROOT if _test_out_root is None else _test_out_root
    _, adm, identity = admission_gate()
    pins = adm["input_pins"]
    raw26, _ = A.read_pinned(inputs_dir / FROZEN_PA, pins[FROZEN_PA])
    calendar = expected_calendar(pd.read_parquet(io.BytesIO(raw26)))
    ok, why, allow = seed_allowed(seed, (A.REPO / REGISTER_REL).read_text(), out_root, identity=identity, pins=pins,
                                  calendar=calendar)
    if not ok:
        raise SystemExit(f"refusing: {why}")
    refuse_inside_the_window()
    return execute(launch_command(seed, allow["budget"], data_dir, inputs_dir), cwd=A.REPO).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scripts.audit.c2_framing.f26")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--inputs-dir", type=Path, default=OUT_ROOT / "inputs")
    pr = sub.add_parser("prepare", help="once, after X-37 and its own register row: the 2026 inputs and their pins")
    pr.add_argument("--data-dir", type=Path, required=True)
    pr.add_argument("--raw-dir", type=Path, required=True)
    pr.add_argument("--screen-inputs", type=Path, required=True)
    pr.add_argument("--out", type=Path, default=OUT_ROOT / "inputs")
    g = sub.add_parser("aggregate")
    g.add_argument("--inputs-dir", type=Path, default=OUT_ROOT / "inputs")
    g.add_argument("run_dirs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.seed, a.data_dir, a.inputs_dir)
    if a.cmd == "launch":
        return launch(a.seed, a.data_dir, a.inputs_dir)
    if a.cmd == "prepare":
        from scripts.audit.c1 import admission as A
        _, adm, _ = admission_gate(require_inputs=False)   # X-37 published and the reviewed code
        problem = prep_row_problem(A.REPO, (A.REPO / REGISTER_REL).read_text(), adm["exposure_commit"])
        if problem:                                           # before any 2026 file is opened or hashed
            raise SystemExit(f"refusing: {problem}")
        screen_pins = json.loads((A.REPO / S.ADMISSION_REL).read_text())["input_pins"]
        print(json.dumps(prepare(a.data_dir, a.raw_dir, a.screen_inputs, screen_pins, a.out), indent=1,
                         sort_keys=True))
        return 0
    print(json.dumps(aggregate(a.run_dirs, a.inputs_dir), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
