"""Skip-policy shadow: a counterfactual "pick-the-band" policy logged alongside production.

WHY THIS EXISTS (docs/audit/2026-06-20-skip-policy-shadow.md): the deployed MDP SKIPS at
streak>=8 when the top candidate's predicted hit prob is below the 0.796 quality-bin boundary.
That boundary was fit on an ACTUAL-PA probability scale but production serves lower ESTIMATED-PA
probabilities, so almost every top pick collapses below it and gets skipped. A calibrated
estimated-PA re-solve put the true breakeven at ~0.744, and the skipped candidates' realized hit
rate straddles it — so backtest could not settle whether the skip rule is +EV. Only live data can.
This shadow accumulates it: when the scheduler records a genuine MDP skip in `decision.json`,
the shadow reads that record, logs what taking the single would have done, and reconciles the
realized outcome, comparing the accumulating hit rate to the 0.744 breakeven.

GROUND TRUTH via `decision.json` (NOT a separate marker). The scheduler writes
`data/picks/<date>/decision.json` (bts_daily_decision_v1) at each finalization point; when
`action=="skip"` and `source=="mdp"` that is the authoritative "MDP evaluated an EXECUTABLE
candidate and chose to skip it" signal. The shadow reads genuine MDP skips directly from there.

This is a SHADOW POLICY, not a shadow model (cf. shadow_eval.py). Pure logic + injected deps
(hit checker) so tests need no MLB API.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from bts.util import atomic_write_text

DECISION_SCHEMA = "bts_skip_policy_shadow_v1"
STATUS_SCHEMA = "bts_skip_policy_shadow_status_v2"

# Calibrated estimated-PA breakeven for the streak>=8 pick-vs-skip decision (the candidate
# true hit-prob at which Q(single)==Q(skip)); robust ~0.742-0.752 across boundaries/horizons.
# Derivation is VERSIONED (audit F10): scripts/audit/skip_breakeven_derivation.py — the
# original /tmp diagnostic is gone; that script re-derives it from repo artifacts.
BREAKEVEN_P = 0.744

# Pre-registered verdict looks (audit F10). The old design re-tested a 95% Wilson CI every
# night as n grew — repeated looks inflate the chance of an eventual chance crossing being
# presented as a verdict. Now the verdict is evaluated ONLY at these resolved-n checkpoints,
# each at a Bonferroni-split alpha (0.05/3 two-sided -> z=2.394), computed deterministically
# from the FIRST c checkpoint-ELIGIBLE records in date order, so the nightly stateless
# rebuild replays exactly the same looks. A decisive look is terminal. Eligibility (Codex
# review L3): only records older than the void-staleness window count toward checkpoint
# membership — younger records can still flip (late resolution) or vanish (prune_superseded
# is same-day), which would reshuffle an already-fired look. Once past the window a record's
# fate is sealed (hit/miss terminal in reconcile; pending forced to void), so membership is
# immutable by construction. Recent records feed the monitoring CI only.
CHECKPOINTS = (30, 60, 90)
Z_CHECKPOINT = 2.394
# A pending outcome older than this is treated as void (game is final by now; unresolved =
# postponed/scratched/data-gap) so live-but-unfinished games are retried, not voided immediately.
STALE_AFTER_DAYS = 3
CHECKPOINT_ELIGIBLE_AFTER_DAYS = STALE_AFTER_DAYS + 1

_RANK_FIELDS = ("batter_id", "batter_name", "team", "game_pk", "p_game_hit")
_RESOLVED = ("hit", "miss", "void")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (point, lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _utc_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Shadow decision records (one per genuine skip day): {date}.policy_shadow.json
# ---------------------------------------------------------------------------

def decision_path(date: str, picks_dir) -> Path:
    return Path(picks_dir) / f"{date}.policy_shadow.json"


def build_divergent_record(date: str, dec: dict, *, now=None) -> dict:
    """A skip day = a divergence: deployed skipped, the pick-the-band shadow takes the single on
    the exact candidate from the authoritative decision.json record."""
    cand = dec.get("primary") or {}
    return {
        "schema_version": DECISION_SCHEMA,
        "date": date,
        "recorded_at": _utc_iso(now),
        "deployed_action": "skip",
        "shadow_action": "single",
        "divergent": True,
        "streak": dec.get("streak"),
        "saver_available": dec.get("saver_available"),
        "rank1": {k: cand.get(k) for k in _RANK_FIELDS},
        "shadow_pick_result": None,  # filled by reconcile_decision
    }


def _read_regime(date: str, picks_dir) -> dict | None:
    """Production-regime fingerprint from the day's saved candidate pick file, or None.

    Same identity realized_calibration pools by (audit F6): (policy_npz_sha256,
    feature_env_hash). Stored per record so a future stratification can split the
    accumulating sample on regime changes — the 0.744 breakeven came from one model
    era and records must stay attributable (audit F10).
    """
    try:
        body = json.loads((Path(picks_dir) / f"{date}.json").read_text())
    except (OSError, ValueError):
        return None
    policy_sha = body.get("policy_npz_sha256")
    env_hash = body.get("feature_env_hash")
    if policy_sha is None and env_hash is None:
        return None
    # pick_run_time: the {date}.json pick file is mutable and an MDP-skip
    # cycle does NOT re-save it, so this stamp is best-effort (a mid-day
    # deploy can leave it one regime behind — Codex review L5). Recording
    # the pick's run_time lets future stratification detect staleness.
    return {
        "policy_npz_sha256": policy_sha,
        "feature_env_hash": env_hash,
        "pick_run_time": body.get("run_time"),
    }


def record_skip_from_decision(date: str, picks_dir, *, now=None) -> dict | None:
    """Write {date}.policy_shadow.json from the authoritative decision.json.

    Returns None (writes nothing) if: a record already exists (idempotent — never clobber a
    reconciled outcome); or the day's decision.json is not an MDP skip
    (action!="skip" or source!="mdp").
    """
    from bts.daily_decision import load_decision
    if decision_path(date, picks_dir).exists():
        return None
    dec = load_decision(date, picks_dir)
    if not dec or dec.get("action") != "skip" or dec.get("source") != "mdp":
        return None
    record = build_divergent_record(date, dec, now=now)
    record["regime"] = _read_regime(date, picks_dir)
    atomic_write_text(decision_path(date, picks_dir), json.dumps(record, indent=2))
    return record


def prune_superseded(picks_dir, *, now: datetime | None = None) -> list[str]:
    """Delete a RECENT policy_shadow record whose date's decision.json is no longer an MDP skip.

    Handles the case where a provisional skip was later superseded by a committed pick written
    to decision.json (e.g. late delivery after a fallback) — a same-day/next-day event. Records
    older than the checkpoint-eligibility window are NEVER pruned (Codex round-2 R5): they may
    be members of an already-fired pre-registered look, and deleting one would reshuffle the
    first-c window and un-decide a terminal verdict. An old date whose decision.json vanishes
    is an anomaly to investigate, not a supersession. Returns the removed dates.
    """
    from bts.daily_decision import load_decision
    now = now or datetime.now(timezone.utc)
    removed = []
    for f in sorted(Path(picks_dir).glob("*.policy_shadow.json")):
        date = f.name[: -len(".policy_shadow.json")]
        try:
            rec_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (now - rec_date).days > CHECKPOINT_ELIGIBLE_AFTER_DAYS:
            continue  # membership of fired looks is immutable — never prune aged records
        dec = load_decision(date, picks_dir)
        if not dec or dec.get("action") != "skip" or dec.get("source") != "mdp":
            try:
                f.unlink()
                removed.append(date)
            except OSError:
                pass
    return removed


def record_pending_skips(picks_dir, *, lookback_days: int = 10, now=None) -> list[str]:
    """Record a shadow entry for every recent MDP skip in decision.json that lacks one.

    Iterates data/picks/*/decision.json; record_skip_from_decision enforces idempotency and
    filters non-MDP-skips, so only genuine gaps are filled (cron-outage safety).
    """
    now = now or datetime.now(timezone.utc)
    recorded = []
    for dec_path in sorted(Path(picks_dir).glob("*/decision.json")):
        date = dec_path.parent.name
        try:
            d = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        age = (now - d).days
        if age < 0 or age > lookback_days:
            continue
        if record_skip_from_decision(date, picks_dir, now=now) is not None:
            recorded.append(date)
    return recorded


def load_decision_records(picks_dir) -> list[dict]:
    records = []
    for f in sorted(Path(picks_dir).glob("*.policy_shadow.json")):
        try:
            records.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


# ---------------------------------------------------------------------------
# Reconciliation + status
# ---------------------------------------------------------------------------

def reconcile_decision(record: dict, *, hit_checker, now: datetime | None = None,
                       stale_after_days: int = STALE_AFTER_DAYS) -> bool:
    """Resolve the realized outcome of the shadow's would-be pick. Returns True if changed.

    ``hit_checker(rank1) -> 'hit'|'miss'|'void'|None`` is injected. None (or a checker error) means
    UNRESOLVED-NOW (game not final / scratch / transient API failure) — the record stays PENDING
    and is retried, not voided (a live west-coast game at the nightly cron is never lost). A pending
    record still unresolved after ``stale_after_days`` is finally marked void.
    """
    if not record.get("divergent"):
        return False
    if record.get("shadow_pick_result") in _RESOLVED:
        return False
    try:
        result = hit_checker(record.get("rank1"))
    except Exception:
        result = None  # transient OR permanent checker failure -> unresolved; staleness handles it
    if result in _RESOLVED:
        record["shadow_pick_result"] = result
        return True
    now = now or datetime.now(timezone.utc)
    try:
        rec_date = datetime.strptime(record["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_days = (now - rec_date).days
    except (KeyError, ValueError, TypeError):
        age_days = 0
    if age_days > stale_after_days:
        record["shadow_pick_result"] = "void"
        return True
    return False


def reconcile_pending(picks_dir, *, hit_checker, now=None, stale_after_days=STALE_AFTER_DAYS) -> int:
    """Resolve realized outcomes for pending decision files. Returns # changed. A per-record
    failure leaves that record pending and continues (one API error must not abort the update)."""
    changed = 0
    for f in sorted(Path(picks_dir).glob("*.policy_shadow.json")):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            did = reconcile_decision(rec, hit_checker=hit_checker, now=now, stale_after_days=stale_after_days)
        except Exception:
            continue
        if did:
            atomic_write_text(f, json.dumps(rec, indent=2))
            changed += 1
    return changed


def _checkpoint_verdict(results: list[str], breakeven: float,
                        checkpoints: tuple = CHECKPOINTS,
                        z: float = Z_CHECKPOINT) -> tuple[str, dict]:
    """Pre-registered sequential verdict (audit F10).

    ``results`` = 'hit'/'miss' outcomes of resolved divergent days in DATE order.
    Walks the checkpoints in order; each look tests the Wilson CI (at the
    Bonferroni-split z) of the FIRST c results against the breakeven. The first
    decisive look is terminal. Returns (verdict, basis) where basis records
    which look produced the verdict — n between checkpoints contributes to
    monitoring only, never to a new look.
    """
    basis = {"checkpoint": None, "n_used": None, "hits_used": None,
             "z": z, "checkpoints": list(checkpoints), "ci": None}
    n = len(results)
    if not checkpoints or n < checkpoints[0]:
        return "insufficient_n", basis
    verdict = "straddles_breakeven"
    for c in checkpoints:
        if n < c:
            break
        hits_c = sum(1 for r in results[:c] if r == "hit")
        _, lo, hi = _wilson(hits_c, c, z=z)
        basis.update({"checkpoint": c, "n_used": c, "hits_used": hits_c,
                      "ci": [lo, hi]})
        if hi < breakeven:
            return "below_breakeven", basis
        if lo > breakeven:
            return "above_breakeven", basis
    return verdict, basis


def build_skip_policy_shadow_status(records: list[dict], *, breakeven_p: float = BREAKEVEN_P,
                                    checkpoints: tuple = CHECKPOINTS,
                                    z_checkpoint: float = Z_CHECKPOINT,
                                    as_of=None,
                                    generated_at: str | None = None, git_commit: str | None = None) -> dict:
    """Aggregate decision records into the monitoring status artifact.

    Headline = the realized hit rate of the candidates the deployed policy SKIPPED. The
    running Wilson CI is a MONITOR (display only); the verdict comes exclusively from the
    pre-registered checkpoint looks (audit F10 — nightly re-testing a fixed-N interval is
    not time-uniform).
    """
    divergent = [r for r in records if r.get("divergent")]
    resolved = [r for r in divergent if r.get("shadow_pick_result") in ("hit", "miss")]
    pending = [r for r in divergent if r.get("shadow_pick_result") is None]
    voids = [r for r in divergent if r.get("shadow_pick_result") == "void"]

    hits = sum(1 for r in resolved if r["shadow_pick_result"] == "hit")
    n = len(resolved)
    point, lo, hi = _wilson(hits, n)

    # Checkpoint eligibility (Codex review L3): only records old enough that
    # their fate is sealed can be look members; younger ones monitor only.
    from datetime import timedelta as _timedelta
    as_of = as_of or datetime.now(timezone.utc).date()
    cutoff = (as_of - _timedelta(days=CHECKPOINT_ELIGIBLE_AFTER_DAYS)).isoformat()
    eligible = [
        r["shadow_pick_result"]
        for r in sorted(resolved, key=lambda r: r.get("date") or "")
        if (r.get("date") or "9999") <= cutoff
    ]
    verdict, verdict_basis = _checkpoint_verdict(
        eligible, breakeven_p, checkpoints, z_checkpoint,
    )
    verdict_basis["eligible_n"] = len(eligible)
    verdict_basis["eligibility_cutoff_date"] = cutoff

    rows = [
        {"date": r.get("date"), "streak": r.get("streak"),
         "batter_name": (r.get("rank1") or {}).get("batter_name"),
         "p_game_hit": (r.get("rank1") or {}).get("p_game_hit"),
         "shadow_pick_result": r.get("shadow_pick_result")}
        for r in sorted(divergent, key=lambda r: r.get("date") or "")
    ]

    return {
        "schema_version": STATUS_SCHEMA,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "initiative": {
            "name": "skip_policy_shadow_v1",
            "description": (
                "Counterfactual 'pick-the-band' shadow. The scheduler writes decision.json at each "
                "genuine MDP skip (action=skip, source=mdp); this accumulates the realized hit rate "
                "of those candidates vs the calibrated breakeven (~0.744) to settle whether the "
                "streak>=8 skip rule is +EV on the production scale."
            ),
            "activation_config": "nightly cron: bts skip-policy-shadow-update (reads decision.json)",
            "decision_file_pattern": "*.policy_shadow.json",
            "decision_json_pattern": "<date>/decision.json (action=skip, source=mdp)",
            "breakeven_p": breakeven_p,
            "breakeven_derivation": "scripts/audit/skip_breakeven_derivation.py",
            "checkpoints": list(checkpoints),
            "z_checkpoint": z_checkpoint,
            "design_doc": "docs/audit/2026-06-20-skip-policy-shadow.md",
        },
        "counts": {
            "decision_files": len(records),
            "divergent_days": len(divergent),
            "resolved_divergent": n,
            "pending": len(pending),
            "void": len(voids),
        },
        "shadow_band_hit_rate": {
            "resolved": n,
            "hits": hits,
            "rate": (point if n else None),
            "wilson_ci": ([lo, hi] if n else None),   # monitoring display, NOT the verdict basis
            "breakeven_p": breakeven_p,
            "verdict": verdict,
            "verdict_basis": verdict_basis,
        },
        "rows": rows,
    }


def write_status(picks_dir, status_path, *, breakeven_p=BREAKEVEN_P,
                 checkpoints=CHECKPOINTS, generated_at=None, git_commit=None) -> dict:
    status = build_skip_policy_shadow_status(
        load_decision_records(picks_dir), breakeven_p=breakeven_p,
        checkpoints=checkpoints, generated_at=generated_at or _utc_iso(),
        git_commit=git_commit)
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(status, indent=2))
    return status


def make_hit_checker():
    """Realized-outcome callable backed by the MLB Stats API (game_pk is reliable in decision.json).

    Returns 'hit'/'miss'/'void', or None when not resolvable yet (game not final / batter not
    found) so the record stays pending and is retried — NOT voided immediately. 'void' is a
    suspended game with no pre-suspension PA for the batter (the resumed portion is never
    evaluated for BTS); build_skip_policy_shadow_status excludes voids from the band hit rate.
    """
    from bts.picks import check_hit

    def checker(rank1):
        if not rank1:
            return None
        return check_hit(rank1.get("game_pk"), rank1.get("batter_id"),
                         rank1.get("batter_name"), team=rank1.get("team"),
                         return_status=True)
    return checker
