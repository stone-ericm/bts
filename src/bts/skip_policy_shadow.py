"""Skip-policy shadow: a counterfactual "pick-the-band" policy logged alongside production.

WHY THIS EXISTS (docs/audit/2026-06-20-skip-policy-shadow.md): the deployed MDP SKIPS at
streak>=8 when the top candidate's predicted hit prob is below the 0.796 quality-bin boundary.
That boundary was fit on an ACTUAL-PA probability scale but production serves lower ESTIMATED-PA
probabilities, so almost every top pick collapses below it and gets skipped. A calibrated
estimated-PA re-solve put the true breakeven at ~0.744, and the skipped candidates' realized hit
rate straddles it — so backtest could not settle whether the skip rule is +EV. Only live data can.
This shadow accumulates it: when production records a genuine MDP skip, the shadow logs what taking
the single would have done and reconciles the realized outcome, comparing the accumulating hit rate
to the 0.744 breakeven.

GROUND TRUTH via a decision marker (NOT reconstruction). The faithful signal — "the MDP evaluated
an EXECUTABLE candidate and chose to skip it" — is not recoverable from saved slates / pick files /
skip_summary (4 review rounds confirmed: select_pick returns None for no-eligible / status-failure
too, the policy's skip region is state-dependent, and the saved slate is pre-filter). So the live
pick path writes a small marker AT the skip decision (`record_mdp_skip_decision`, called from
strategy.select_pick on action=="skip", best-effort/never-raises, not on the shadow-model path),
recording the actual action + the EXECUTABLE declined candidate. The shadow reads that marker. This
is a minimal additive write to the cascade — it records, it does not change any pick.

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
STATUS_SCHEMA = "bts_skip_policy_shadow_status_v1"
SKIP_DECISION_SCHEMA = "bts_mdp_skip_decision_v1"  # the cascade-written marker

# Calibrated estimated-PA breakeven for the streak>=8 pick-vs-skip decision (the candidate
# true hit-prob at which Q(single)==Q(skip)); robust ~0.742-0.752 across boundaries/horizons.
BREAKEVEN_P = 0.744
# Minimum resolved divergent days before the verdict is treated as a readout (not noise).
MIN_DIVERGENT_DAYS = 30
# A pending outcome older than this is treated as void (game is final by now; unresolved =
# postponed/scratched/data-gap) so live-but-unfinished games are retried, not voided immediately.
STALE_AFTER_DAYS = 3

_RANK_FIELDS = ("batter_id", "batter_name", "team", "game_pk", "pitcher_name", "p_game_hit")
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
# Cascade-written decision marker (the ground-truth seam). record_mdp_skip_decision is called
# from strategy.select_pick the moment the MDP chooses to skip; everything else here READS it.
# ---------------------------------------------------------------------------

def skip_decision_path(date: str, picks_dir) -> Path:
    return Path(picks_dir) / date / "skip_decision.json"


def record_mdp_skip_decision(date: str, picks_dir, *, candidate: dict, streak=None,
                             saver_available=None, now=None) -> None:
    """Persist the genuine MDP skip + the EXECUTABLE candidate it declined.

    Called by the live pick path ONLY when ``decide_action`` returned "skip" (never on the
    shadow-model path). Best-effort: the caller wraps it so production is never affected.
    """
    record = {
        "schema_version": SKIP_DECISION_SCHEMA,
        "date": date,
        "action": "skip",
        "streak": streak,
        "saver_available": (None if saver_available is None else bool(saver_available)),
        "candidate": {k: candidate.get(k) for k in _RANK_FIELDS},
        "recorded_at": _utc_iso(now),
    }
    path = skip_decision_path(date, picks_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=2))


def load_skip_decision(date: str, picks_dir) -> dict | None:
    path = skip_decision_path(date, picks_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Shadow decision records (one per genuine skip day): {date}.policy_shadow.json
# ---------------------------------------------------------------------------

def decision_path(date: str, picks_dir) -> Path:
    return Path(picks_dir) / f"{date}.policy_shadow.json"


def build_divergent_record(date: str, marker: dict, *, now=None) -> dict:
    """A skip day = a divergence: deployed skipped, the pick-the-band shadow takes the single on
    the exact candidate the MDP declined (from the marker)."""
    cand = marker.get("candidate") or {}
    return {
        "schema_version": DECISION_SCHEMA,
        "date": date,
        "recorded_at": _utc_iso(now),
        "deployed_action": "skip",
        "shadow_action": "single",
        "divergent": True,
        "streak": marker.get("streak"),
        "saver_available": marker.get("saver_available"),
        "rank1": {k: cand.get(k) for k in _RANK_FIELDS},
        "shadow_pick_result": None,  # filled by reconcile_decision
    }


def _final_decision(date: str, picks_dir, *, pick_loader=None, delivered_fn=None) -> str | None:
    """The day's FINAL production decision: 'pick', 'skip', or None (neither).

    Authoritative signal = `picks.pick_was_delivered` — was a pick DURABLY DELIVERED to a human
    (posted / DM'd)? Both the `<date>.json` pick file and the skip marker are PROVISIONAL —
    `bts preview` pre-writes the pick file, the scheduler saves candidates pre-lock, the fallback
    re-delivers a cached pick (stale run_time), and the marker is overwritten — so mere existence
    is not authoritative. A delivered pick is production's final action; otherwise a skip marker
    means a genuine MDP skip. (Prod runs public/DM delivery, so a real final pick sets the
    pick_was_delivered fields; a private/local-only lock is intentionally not "delivered".)
    """
    if pick_loader is None:
        from bts.picks import load_pick
        pick_loader = load_pick
    if delivered_fn is None:
        from bts.picks import pick_was_delivered
        delivered_fn = pick_was_delivered
    daily = pick_loader(date, Path(picks_dir))
    if daily is not None and getattr(daily, "pick", None) is not None and delivered_fn(daily):
        return "pick"
    if load_skip_decision(date, picks_dir):
        return "skip"
    return None


def record_skip_from_marker(date: str, picks_dir, *, now=None, pick_loader=None, delivered_fn=None) -> dict | None:
    """Write {date}.policy_shadow.json from the cascade's skip marker.

    Returns None (writes nothing) if: a record already exists (idempotent — never clobber a
    reconciled outcome); or the day's FINAL decision was not a skip (production durably DELIVERED a
    pick — the authoritative signal, since the pick file and marker are both provisional).
    """
    if decision_path(date, picks_dir).exists():
        return None
    if _final_decision(date, picks_dir, pick_loader=pick_loader, delivered_fn=delivered_fn) != "skip":
        return None
    record = build_divergent_record(date, load_skip_decision(date, picks_dir), now=now)
    path = decision_path(date, picks_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=2))
    return record


def prune_superseded(picks_dir, *, pick_loader=None, delivered_fn=None) -> list[str]:
    """Delete any policy_shadow record whose date's FINAL decision was NOT a skip — a provisional
    skip later superseded by a DELIVERED pick. Returns the removed dates."""
    removed = []
    for f in sorted(Path(picks_dir).glob("*.policy_shadow.json")):
        date = f.name[: -len(".policy_shadow.json")]
        if _final_decision(date, picks_dir, pick_loader=pick_loader, delivered_fn=delivered_fn) != "skip":
            try:
                f.unlink()
                removed.append(date)
            except OSError:
                pass
    return removed


def record_pending_skips(picks_dir, *, lookback_days: int = 10, now=None, pick_loader=None,
                         delivered_fn=None) -> list[str]:
    """Record a shadow entry for every recent skip marker that lacks one (cron-outage safety).

    record_skip_from_marker enforces idempotency + pick-precedence, so a marker on a date that was
    ultimately a pick is never recorded.
    """
    now = now or datetime.now(timezone.utc)
    recorded = []
    for marker_path in sorted(Path(picks_dir).glob("*/skip_decision.json")):
        date = marker_path.parent.name
        try:
            d = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        age = (now - d).days
        if age < 0 or age > lookback_days:
            continue
        if record_skip_from_marker(date, picks_dir, now=now, pick_loader=pick_loader,
                                   delivered_fn=delivered_fn) is not None:
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


def _verdict(lo: float, hi: float, n: int, breakeven: float, min_n: int) -> str:
    """below_breakeven -> picking is -EV (skip validated); above_breakeven -> picking is +EV
    (skip costs); straddles/insufficient_n -> not yet resolvable."""
    if n < min_n:
        return "insufficient_n"
    if hi < breakeven:
        return "below_breakeven"
    if lo > breakeven:
        return "above_breakeven"
    return "straddles_breakeven"


def build_skip_policy_shadow_status(records: list[dict], *, breakeven_p: float = BREAKEVEN_P,
                                    min_divergent_days: int = MIN_DIVERGENT_DAYS,
                                    generated_at: str | None = None, git_commit: str | None = None) -> dict:
    """Aggregate decision records into the monitoring status artifact.

    Headline = the realized hit rate of the candidates the deployed policy SKIPPED, with a Wilson
    CI and a verdict against the calibrated breakeven.
    """
    divergent = [r for r in records if r.get("divergent")]
    resolved = [r for r in divergent if r.get("shadow_pick_result") in ("hit", "miss")]
    pending = [r for r in divergent if r.get("shadow_pick_result") is None]
    voids = [r for r in divergent if r.get("shadow_pick_result") == "void"]

    hits = sum(1 for r in resolved if r["shadow_pick_result"] == "hit")
    n = len(resolved)
    point, lo, hi = _wilson(hits, n)
    verdict = _verdict(lo, hi, n, breakeven_p, min_divergent_days)

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
                "Counterfactual 'pick-the-band' shadow. The live pick path records a marker at each "
                "genuine MDP skip (the executable declined candidate); this accumulates the realized "
                "hit rate of those candidates vs the calibrated breakeven (~0.744) to settle whether "
                "the streak>=8 skip rule is +EV on the production scale."
            ),
            "activation_config": "nightly cron: bts skip-policy-shadow-update (+ select_pick writes the marker)",
            "decision_file_pattern": "*.policy_shadow.json",
            "skip_marker_pattern": "<date>/skip_decision.json",
            "breakeven_p": breakeven_p,
            "min_divergent_days_for_readout": min_divergent_days,
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
            "wilson_ci": ([lo, hi] if n else None),
            "breakeven_p": breakeven_p,
            "verdict": verdict,
        },
        "rows": rows,
    }


def write_status(picks_dir, status_path, *, breakeven_p=BREAKEVEN_P,
                 min_divergent_days=MIN_DIVERGENT_DAYS, generated_at=None, git_commit=None) -> dict:
    status = build_skip_policy_shadow_status(
        load_decision_records(picks_dir), breakeven_p=breakeven_p,
        min_divergent_days=min_divergent_days, generated_at=generated_at or _utc_iso(),
        git_commit=git_commit)
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(status, indent=2))
    return status


def make_hit_checker():
    """Realized-outcome callable backed by the MLB Stats API (game_pk is reliable in the marker).

    Returns 'hit'/'miss', or None when not resolvable yet (game not final / batter not found) so
    the record stays pending and is retried — NOT voided immediately.
    """
    from bts.picks import check_hit

    def checker(rank1):
        if not rank1:
            return None
        res = check_hit(rank1.get("game_pk"), rank1.get("batter_id"),
                        rank1.get("batter_name"), team=rank1.get("team"))
        return "hit" if res is True else ("miss" if res is False else None)
    return checker
