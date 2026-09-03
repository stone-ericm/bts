#!/usr/bin/env python3
"""Boundary-shadow one-step table-intent disagreement census (MECHANISM PHASE).

Registration: docs/superpowers/specs/2026-08-09-boundary-shadow-measurement.md
(r0 `4800f7a`, amended r1 `6db921c` after partial unblinding). This harness
answers ONE question: on realized decision-era states, where does quintile
boundary rescaling change the deployed action TABLE's intent? Rows are
unchained (not a counterfactual season); outcomes are deliberately absent
from the artifact — the outcome phase runs only under the registration's
follow-up rules.

State authority (never local pick-result replay — production is
contest-anchored, 2026-06-17 design):
- recorded:    the decision record's own persisted streak/saver (MDP skips,
               and every bts_daily_decision_v2 record).
- ledger_asof: latest contest_ledger.jsonl observation with
               recorded_at <= finalized_at, saver from the account flag
               (constant "active" since 2026-06-18 per saver_state.json).
- unknown:     everything else — counted, never diffed.

Gate A validates the ledger_asof resolver against every recorded-state row
(halt on mismatch). Gate B validates raw deployed-boundary lookups against
recorded actions with clamp attribution (halt when unattributable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bts.simulate.mdp import load_policy, lookup_action  # noqa: E402
from bts.strategy import SEASON_END_DATE  # noqa: E402
from bts.util import atomic_write_text  # noqa: E402

SCHEMA_VERSION = "bts_boundary_shadow_census_v1"
REGISTRATION_COMMITS = {"r0": "4800f7a", "r1_amended": "6db921c"}
QUANTS = (0.2, 0.4, 0.6, 0.8)
ACCEPTED_DECISION_SCHEMAS = {"bts_daily_decision_v1", "bts_daily_decision_v2", "bts_daily_decision_v3"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STREAK_BANDS = ((0, 2), (3, 7), (8, 9), (10, 15), (16, 999))


class CensusHalt(RuntimeError):
    """A parity/validity failure that must stop the census (bug, not data)."""


# --------------------------------------------------------------------------
# boundaries

def quintile_boundaries(ps) -> list[float] | None:
    """Four boundaries at .2/.4/.6/.8 (pandas linear interpolation).

    Halt on non-finite / out-of-range inputs. Returns None (variant INVALID)
    when the boundaries are not strictly increasing — never jitter or dedup.
    """
    values = [float(p) for p in ps]
    if not values:
        raise CensusHalt("empty p sample for boundary construction")
    for v in values:
        if not math.isfinite(v) or not (0.0 <= v <= 1.0):
            raise CensusHalt(f"invalid p value {v!r} in boundary sample")
    series = pd.Series(values)
    bounds = [float(series.quantile(q, interpolation="linear")) for q in QUANTS]
    if not all(b2 > b1 for b1, b2 in zip(bounds, bounds[1:])):
        return None
    return bounds


# --------------------------------------------------------------------------
# ledger as-of state

@dataclass(frozen=True)
class LedgerObs:
    recorded_at: datetime
    active_streak: int
    source_date: str | None


def _parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_ledger(path: Path) -> list[LedgerObs]:
    obs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        obs.append(LedgerObs(
            recorded_at=_parse_utc(row["recorded_at"]),
            active_streak=int(row["active_streak"]),
            source_date=row.get("source_date"),
        ))
    obs.sort(key=lambda o: o.recorded_at)
    return obs


def asof_contest(ledger: list[LedgerObs], ts: datetime) -> LedgerObs | None:
    best = None
    for o in ledger:
        if o.recorded_at <= ts:
            best = o
        else:
            break
    return best


def resolve_state(rec: dict, ledger: list[LedgerObs], *, saver_active: bool):
    """-> (state_source, streak, saver). Recorded state wins; else ledger as-of."""
    if rec.get("streak") is not None:
        saver = rec.get("saver_available")
        return "recorded", int(rec["streak"]), (bool(saver) if saver is not None else saver_active)
    obs = asof_contest(ledger, _parse_utc(rec["finalized_at"]))
    if obs is None:
        return "unknown", None, None
    return "ledger_asof", obs.active_streak, saver_active


# --------------------------------------------------------------------------
# actions

def attribution(raw: str, recorded: str) -> str:
    """Classify raw-table-intent vs recorded-action. Halt when unattributable.

    The only documented clamps downgrade an MDP double to a single
    (allow_double, different-game executability) — no clamp creates or
    removes a skip, and none upgrades toward double.
    """
    if raw == recorded:
        return "parity"
    if raw == "double" and recorded == "single":
        return "clamped_double_downgrade"
    raise CensusHalt(f"unattributable action mismatch: raw={raw} recorded={recorded}")


def _days_remaining(date_str: str) -> int:
    end = datetime.strptime(SEASON_END_DATE, "%Y-%m-%d").date()
    return max(0, (end - date_cls.fromisoformat(date_str)).days)


def _band(streak: int) -> str:
    for lo, hi in STREAK_BANDS:
        if lo <= streak <= hi:
            return f"{lo}-{hi}" if hi < 999 else f">={lo}"
    return "?"


# --------------------------------------------------------------------------
# inputs

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _collect_decisions(picks_dir: Path, start: str, end: str):
    rows, excluded = [], []
    for day_dir in sorted(Path(picks_dir).iterdir()):
        if not day_dir.is_dir() or not DATE_RE.match(day_dir.name):
            continue
        path = day_dir / "decision.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get("schema_version") not in ACCEPTED_DECISION_SCHEMAS:
            excluded.append({"date": day_dir.name, "reason": "schema"})
            continue
        if rec.get("date") != day_dir.name:
            raise CensusHalt(f"path/body date mismatch under {day_dir}")
        if not (start <= rec["date"] <= end):
            excluded.append({"date": rec["date"], "reason": "outside_era"})
            continue
        rows.append((path, rec))
    return rows, excluded


def _boundary_sample_decisions(decisions):
    """Only reach-57 decisions may shape the boundary samples: tail-objective
    days (2026-09-03) pick regardless of the reach-57 bins and would shift the
    quintiles used on reach-57 rows (Codex r3)."""
    from bts.daily_decision import decision_objective
    return [(p, rec) for p, rec in decisions if decision_objective(rec) == "reach57"]


def _flat_primary_ps(picks_dir: Path, dates: set[str] | None = None):
    out = {}
    for path in sorted(Path(picks_dir).glob("*.json")):
        stem = path.name[:-5]
        if not DATE_RE.match(stem):
            continue
        if dates is not None and stem not in dates:
            continue
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pick = body.get("pick") or {}
        p = pick.get("p_game_hit")
        if p is not None:
            out[stem] = float(p)
    return out


def _git(args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# census

def run_census(*, picks_dir: Path, ledger_path: Path, policy_path: Path,
               saver_active: bool, start: str, end: str, as_of: str,
               output: Path) -> dict:
    picks_dir = Path(picks_dir)
    policy_table, deployed_bounds, season_length = load_policy(Path(policy_path))
    ledger = load_ledger(ledger_path)
    decisions, excluded = _collect_decisions(picks_dir, start, end)

    consumed = {str(p): _sha256(p) for p, _ in decisions}
    consumed[str(ledger_path)] = _sha256(ledger_path)
    consumed[str(policy_path)] = _sha256(policy_path)
    policy_sha = consumed[str(policy_path)]

    def raw(bounds, streak, saver, p, dstr):
        return lookup_action(policy_table, list(bounds), streak,
                             _days_remaining(dstr), bool(saver), float(p),
                             int(season_length))

    # --- boundary variants -------------------------------------------------
    primary_ps, primary_dates = [], []
    for _, rec in _boundary_sample_decisions(decisions):
        if (rec.get("primary") or {}).get("p_game_hit") is not None:
            primary_ps.append(float(rec["primary"]["p_game_hit"]))
            primary_dates.append(rec["date"])
    flat_same = _flat_primary_ps(picks_dir, set(primary_dates))
    flat_season = _flat_primary_ps(picks_dir)

    def variant(name, ps, provenance):
        vals = quintile_boundaries(ps) if ps else None
        return {"name": name, "provenance": provenance, "n": len(ps),
                "values": vals, "valid": vals is not None}

    boundaries = {
        "deployed": {"name": "deployed", "provenance": "mdp_policy.npz",
                     "n": None, "values": list(deployed_bounds), "valid": True},
        "primary": variant("primary", primary_ps,
                           "decision-record primary p, decision era"),
        "s1_flat_same_dates": variant(
            "s1_flat_same_dates", list(flat_same.values()),
            "flat-file rank-1 p, decision-era dates"),
        "s2_flat_season": variant(
            "s2_flat_season", list(flat_season.values()),
            "flat-file rank-1 p, season-to-date"),
    }

    # deterministic stability probes on the PRIMARY sample
    stability = {}
    weeks = {}
    for dstr, p in zip(primary_dates, primary_ps):
        weeks.setdefault(date_cls.fromisoformat(dstr).isocalendar()[1], []).append(p)
    for wk in sorted(weeks):
        rest = [p for w, ps in weeks.items() if w != wk for p in ps]
        stability[f"loco_week_{wk}"] = variant(
            f"loco_week_{wk}", rest, f"primary minus ISO week {wk}")
    for shift, label in ((-7, "end_minus_7d"), (7, "end_plus_7d")):
        cut = (date_cls.fromisoformat(end) + timedelta(days=shift)).isoformat()
        ps = [p for d, p in zip(primary_dates, primary_ps) if d <= cut]
        extra = ({d: p for d, p in flat_season.items() if end < d <= cut}
                 if shift > 0 else {})
        stability[label] = variant(label, ps + list(extra.values()),
                                   f"primary window endpoint {cut}")

    # --- per-row census ----------------------------------------------------
    rows, gate_a = [], {"checked": 0, "mismatches": []}
    diff_variants = [k for k in ("primary", "s1_flat_same_dates", "s2_flat_season")
                     if boundaries[k]["valid"]]
    for path, rec in decisions:
        dstr = rec["date"]
        p = (rec.get("primary") or {}).get("p_game_hit")
        row = {"date": dstr, "action_recorded": rec.get("action"),
               "source": rec.get("source"), "p": p,
               "policy_identity": "inferred"}
        stamp = _pick_policy_stamp(picks_dir, dstr)
        if stamp is not None:
            if stamp != policy_sha:
                raise CensusHalt(
                    f"{dstr}: pick-file policy stamp {stamp[:12]} conflicts "
                    f"with loaded policy {policy_sha[:12]}")
            row["policy_identity"] = "verified"

        if rec.get("source") != "mdp" or p is None:
            row.update(state_source="excluded_non_mdp", diffs={})
            rows.append(row)
            continue
        from bts.daily_decision import decision_objective
        if decision_objective(rec) != "reach57":
            # 2026-09-03: tail-objective decisions (57 unreachable) are a different
            # rule; the census measures the reach-57 table's boundary behaviour.
            row.update(state_source="excluded_tail_objective", diffs={})
            rows.append(row)
            continue

        src, streak, saver = resolve_state(rec, ledger, saver_active=saver_active)
        row.update(state_source=src, streak=streak, saver=saver)

        if src == "recorded":
            obs = asof_contest(ledger, _parse_utc(rec["finalized_at"]))
            gate_a["checked"] += 1
            if obs is None or obs.active_streak != streak:
                gate_a["mismatches"].append(
                    {"date": dstr, "recorded": streak,
                     "ledger_asof": (obs.active_streak if obs else None)})

        if src == "unknown":
            row["diffs"] = {}
            rows.append(row)
            continue

        raw_deployed = raw(deployed_bounds, streak, saver, p, dstr)
        row["raw_deployed"] = raw_deployed
        row["parity"] = attribution(raw_deployed, rec.get("action"))
        row["band"] = _band(streak)
        row["diffs"] = {}
        for name in diff_variants:
            cand = raw(boundaries[name]["values"], streak, saver, p, dstr)
            if cand != raw_deployed:
                row["diffs"][name] = {"from": raw_deployed, "to": cand}
        rows.append(row)

    if gate_a["mismatches"]:
        raise CensusHalt(f"gate A: ledger_asof resolver disagrees with recorded "
                         f"state on {gate_a['mismatches']}")

    # --- summaries ---------------------------------------------------------
    def summarize(name):
        diff_rows = [r for r in rows if name in r.get("diffs", {})]
        transitions = {}
        for r in diff_rows:
            key = f"{r['diffs'][name]['from']}->{r['diffs'][name]['to']}"
            transitions[key] = transitions.get(key, 0) + 1
        return {"diff_dates": [r["date"] for r in diff_rows],
                "n_diff": len(diff_rows),
                "by_band": {b: sum(1 for r in diff_rows if r.get("band") == b)
                            for b in sorted({r.get("band") for r in diff_rows} - {None})},
                "transitions": transitions}

    summaries = {name: summarize(name) for name in diff_variants}

    # Registration r1: stability probes report boundaries AND action-diff
    # counts. Computed over the same state-known rows; never trigger-bearing.
    for sname, svar in stability.items():
        if not svar["valid"]:
            svar["n_diff"] = None
            continue
        n_diff = 0
        for r in rows:
            if r.get("state_source") in ("recorded", "ledger_asof"):
                cand = raw(svar["values"], r["streak"], r["saver"], r["p"], r["date"])
                if cand != r["raw_deployed"]:
                    n_diff += 1
        svar["n_diff"] = n_diff
    primary_dates_set = set(summaries.get("primary", {}).get("diff_dates", []))
    all_sets = [set(s["diff_dates"]) for s in summaries.values()]
    coverage = {
        "rows_total": len(rows),
        "state_source_counts": {
            s: sum(1 for r in rows if r.get("state_source") == s)
            for s in sorted({r.get("state_source") for r in rows})},
        "excluded": excluded,
        "diff_union": sorted(set().union(*all_sets)) if all_sets else [],
        "diff_intersection": sorted(set.intersection(*all_sets)) if all_sets else [],
    }
    follow_up = {
        "rule": ("registration r1: >=8 unique parity-passing state-known intent-diff "
                 "dates in the PRIMARY variant (operational threshold inherited from "
                 "r0; disclosed as binding-adjacent after partial unblinding)"),
        "primary_n_diff": len(primary_dates_set),
        "triggered": len(primary_dates_set) >= 8,
    }

    manifest = hashlib.sha256(
        json.dumps(sorted(consumed.items())).encode()).hexdigest()
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "role": ("mechanism census — one-step table-intent disagreement at "
                 "deployed states; no production claim; results withheld "
                 "per registration"),
        "registration": REGISTRATION_COMMITS,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "params": {"start": start, "end": end, "as_of": as_of,
                   "saver_active_flag": saver_active,
                   "season_end_date": SEASON_END_DATE,
                   "season_length": int(season_length),
                   "quantiles": list(QUANTS)},
        "provenance": {
            "execution_commit": _git(["rev-parse", "HEAD"]),
            "worktree_dirty": bool(_git(["status", "--porcelain"])),
            "policy_npz_sha256": policy_sha,
            "input_manifest_sha256": manifest,
            "inputs_sha256": consumed,
        },
        "boundaries": boundaries,
        "stability": stability,
        "gates": {"gate_a": gate_a},
        "coverage": coverage,
        "rows": rows,
        "summaries": summaries,
        "follow_up": follow_up,
        "reproduce": (f"TZ=America/New_York uv run python "
                      f"scripts/audit/boundary_shadow_census.py "
                      f"--picks-dir <picks> --ledger <ledger> --policy <npz> "
                      f"--start {start} --end {end} --as-of {as_of} "
                      f"--output <path>"),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(artifact, indent=2))
    return artifact


def _pick_policy_stamp(picks_dir: Path, date_str: str) -> str | None:
    path = Path(picks_dir) / f"{date_str}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("policy_npz_sha256")
    except (json.JSONDecodeError, OSError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks-dir", required=True, type=Path)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--saver-active", default=True, type=lambda v: v == "true",
                    help="Account saver flag over the era (constant since 2026-06-18).")
    ap.add_argument("--start", default="2026-06-23")
    ap.add_argument("--end", default="2026-08-09")
    ap.add_argument("--as-of", required=True,
                    help="Explicit snapshot date; no date.today() defaults.")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    artifact = run_census(
        picks_dir=args.picks_dir, ledger_path=args.ledger,
        policy_path=args.policy, saver_active=args.saver_active,
        start=args.start, end=args.end, as_of=args.as_of, output=args.output)
    fu = artifact["follow_up"]
    print(f"rows={artifact['coverage']['rows_total'] if 'coverage' in artifact else len(artifact['rows'])} "
          f"primary_diffs={fu['primary_n_diff']} triggered={fu['triggered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
