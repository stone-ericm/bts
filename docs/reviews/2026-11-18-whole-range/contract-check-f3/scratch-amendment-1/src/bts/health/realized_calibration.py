"""Tier 2: realized calibration check (75-80% predicted-P bucket overconfidence).

The complement to predicted_vs_realized.py. That check detects DRIFT in the
gap between predicted and realized P over time. This check detects the
ABSOLUTE LEVEL of miscalibration in the 75-80% bucket — where most prod
picks land — vs realized hit rates.

**Attribution fix 2026-05-01**: previously used streak ``result`` as proxy
for primary-pick hit. That's biased on double-down days because streak
"hit" requires BOTH picks to hit, so a DD pick that did hit gets attributed
as "miss" whenever the primary missed. The fix: when ``data_dir`` is
provided, look up the actual per-pick day-hit from the season's PA frame.
The biased path remains as a safety fallback when pa frame isn't available.

The corrected attribution shows real over-confidence is ~+6.6pp overall
and ~+12.3pp in the [0.75, 0.80) bucket — **less alarming than the
proxy-based "+14pp" finding from 2026-04-29**, which was inflated by the
DD attribution bias. Thresholds are recalibrated accordingly.

Severity ladder (applied per configured bucket; since 2026-07-12 the default
buckets are the 75-80% primary band AND a 70-75% DD-leg-only band — the DD
band had no absolute-level monitor while its legs ran 0.545 realized vs
0.731 stated over the season):
  predicted - realized < 8pp:     no alert (well-calibrated under proper attribution)
  >= 8pp:                         INFO  (worth observing)
  >= 15pp:                        WARN  (significantly overconfident)
  >= 25pp:                        CRITICAL (true distribution-shift signal)

Lookback window: last 30 days. Minimum bucket count: 5 picks.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "realized_calibration"

DEFAULT_THRESHOLDS = {
    "info_pp": 8.0,
    "warn_pp": 15.0,
    "critical_pp": 25.0,
    "lookback_days": 30,
    "min_bucket_n": 5,
    # CRITICAL additionally requires this n: at the 25pp bar, SE(realized)
    # is ~10pp only once n≥20 (0.74·0.26/0.10² ≈ 19), so smaller samples
    # cap at WARN — the attention digest carries a persistent WARN to the
    # operator without a 2σ-ish reading claiming "real signal" outright.
    "min_bucket_n_critical": 20,
    # ...UNLESS the exact Poisson-binomial tail under the stated p's is
    # effectively impossible (review r2#4: 0-for-8 at 0.73 has tail 2.8e-5 —
    # a grading/serving pipeline failure must not hide behind the n gate).
    "critical_tail_epsilon": 1e-3,
    # Buckets: (low, high, slot filter, label); each alerts independently
    # (distinct incident_key) on the same pp ladder. The original single
    # [0.75, 0.80) bucket watches where most PRIMARY picks land. The
    # [0.70, 0.75) DD-leg bucket was added 2026-07-12: season data showed
    # DD legs realizing 0.545 vs stated 0.731 in that band while primaries
    # in the SAME band were calibrated (-2.8pp) — a pooled bucket dilutes
    # the slot-specific signal below the WARN bar, and no absolute-level
    # monitor watched the DD band at all (predicted_vs_realized + dd_pair
    # are drift-based, so a chronic shortfall spanning their baselines is
    # invisible to them by construction).
    "buckets": [
        {"low": 0.75, "high": 0.80, "slots": None, "label": "75-80%"},
        {"low": 0.70, "high": 0.75, "slots": ["double_down"], "label": "70-75% DD-leg"},
    ],
}


def _poisson_binomial_tail_le(ps: list[float], k: int) -> float:
    """Exact P(X <= k) for X = sum of independent Bernoulli(ps). O(n²) DP —
    bucket sizes are tens at most."""
    probs = [1.0]
    for p in ps:
        nxt = [0.0] * (len(probs) + 1)
        for j, pr in enumerate(probs):
            nxt[j] += pr * (1 - p)
            nxt[j + 1] += pr * p
        probs = nxt
    return sum(probs[: k + 1])


def _build_day_hit_lookup(data_dir: Path, today: date, lookback_days: int) -> dict:
    """Build (batter_id, date) -> day_had_any_hit lookup from current season's PA frame.

    Returns empty dict if no parquet exists; caller falls back to streak-result proxy.
    """
    try:
        import pandas as pd
    except ImportError:
        return {}
    # Local import: build.py imports pandas at module level, so importing it lazily here
    # preserves this function's graceful pandas-absent degradation above.
    from bts.data.build import read_pa_for_bts_scoring
    cutoff = today - timedelta(days=lookback_days)
    candidates = [data_dir / f"pa_{y}.parquet" for y in (today.year, today.year - 1)]
    parts = []
    for p in candidates:
        if p.exists():
            try:
                parts.append(read_pa_for_bts_scoring(p, ["batter_id", "date", "is_hit"]))
            except Exception as e:
                log.warning(f"failed to load {p} for calibration attribution: {e}")
    if not parts:
        return {}
    pa_df = pd.concat(parts, ignore_index=True)
    pa_df["date"] = pd.to_datetime(pa_df["date"]).dt.date
    pa_df = pa_df[(pa_df["date"] >= cutoff) & (pa_df["date"] <= today)]
    daily = (
        pa_df.groupby(["batter_id", "date"])["is_hit"]
        .max()
        .reset_index()
    )
    return {(int(r["batter_id"]), r["date"]): int(r["is_hit"]) for _, r in daily.iterrows()}


def _current_deploy_iso(repo_dir: Path) -> str | None:
    """Return the ISO timestamp the current model was DEPLOYED, or None.

    Used to filter out picks generated by older model iterations. Picks before
    the current deploy can't be pooled with current-model picks for calibration
    (per project_bts_production_realized_contaminated.md).

    Prefers `data/.last_deploy_iso` — a wall-clock stamp written by the deploy
    workflow on every deploy/rollback. It is monotonic, so it fixes both
    commit-time≠deploy-time and the canary-rollback-moves-HEAD's-date-backward
    case. Falls back to HEAD's git commit time (`%cI`) when the stamp is absent
    (older boxes / local runs) — approximate, but no worse than before.
    """
    stamp = Path(repo_dir) / "data" / ".last_deploy_iso"
    try:
        val = stamp.read_text().strip()
        if val:
            return val
    except OSError:
        pass
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "log", "-1", "--format=%cI", "HEAD"],
            stderr=subprocess.PIPE, text=True, timeout=5,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _iso_before(run_time: str, since_deploy_iso: str) -> bool:
    """True if instant ``run_time`` is strictly before ``since_deploy_iso``.

    Compares timezone-aware datetimes, NOT ISO strings: run_time is written in
    +00:00 while ``since_deploy_iso`` (git %cI) carries the committer's local
    offset, so a lexicographic compare misclassifies picks whenever the offsets
    differ. Unparseable run_time → treated as pre-deploy (conservative skip).
    A naive timestamp is assumed UTC.
    """
    try:
        rt = datetime.fromisoformat(run_time)
        dep = datetime.fromisoformat(since_deploy_iso)
    except (ValueError, TypeError):
        return True
    if rt.tzinfo is None:
        rt = rt.replace(tzinfo=timezone.utc)
    if dep.tzinfo is None:
        dep = dep.replace(tzinfo=timezone.utc)
    return rt < dep


def _regime_fingerprint(body: dict) -> tuple | None:
    """Production-regime identity of a pick, or None if unstamped.

    (policy_npz_sha256, feature_env_hash) — stamps every pick file carries
    since provenance v1. Deliberately EXCLUDES model_pickle_sha256: the blend
    retrains daily (blend_<date>.pkl), so its sha changes every pick day and
    would fragment the pool back to n≈1 (Codex review #2; box-verified
    2026-07-06..08: model sha differed daily, policy+env stable across a
    deploy). Two picks sharing policy + feature-env come from the same
    probability regime for calibration purposes (audit F6). Known residual:
    a predictor-CODE-only change doesn't flip either stamp and thus doesn't
    reset the pool — an explicit regime-version stamp is the future fix."""
    fp = (
        body.get("policy_npz_sha256"),
        body.get("feature_env_hash"),
    )
    return fp if all(fp) else None


def check(
    picks_dir: Path,
    today: date | None = None,
    thresholds: dict | None = None,
    data_dir: Path | None = None,
    since_deploy_iso: str | None = None,
) -> list[Alert]:
    """Returns INFO/WARN/CRITICAL alert per overconfident bucket.

    Slot outcomes come from production-graded ``slot_results`` when present
    (authoritative), then the PA-frame join when ``data_dir`` is provided,
    then the day-result proxy (primary slot on non-DD days only — day-level
    results misattribute DD picks, the 2026-05-01 bias).

    Pooling (audit F6): picks are pooled by production-REGIME fingerprint
    (model/policy/feature-env hashes stamped on every pick) — the pool is all
    in-window picks matching the newest stamped pick's fingerprint. Deploys
    that don't change the regime (docs/shadow/ops) no longer reset the sample;
    a genuine model/policy/env change still does. ``since_deploy_iso`` is the
    FALLBACK filter for unstamped (pre-provenance) pick sets: it drops picks
    whose ``run_time`` predates the current deploy (see
    project_bts_production_realized_contaminated.md). With neither available,
    the legacy unfiltered behavior applies — noise, not signal, on a
    long-running project.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if thresholds and "buckets" not in thresholds and (
        "bucket_low" in thresholds or "bucket_high" in thresholds
    ):
        # Legacy single-bucket override (pre-2026-07-12 schema): translate
        # instead of silently ignoring it (review r2#7 — a box override
        # supplying only bucket_low/high would otherwise monitor the
        # defaults while claiming to monitor something else).
        low = float(thresholds.get("bucket_low", 0.75))
        high = float(thresholds.get("bucket_high", 0.80))
        log.warning(
            "realized_calibration: legacy bucket_low/bucket_high override "
            "translated to a single-bucket spec [%s, %s); use 'buckets'.",
            low, high,
        )
        t["buckets"] = [{"low": low, "high": high, "slots": None,
                         "label": f"{low * 100:.0f}-{high * 100:.0f}%"}]
    if today is None:
        today = date.today()
    if not picks_dir.exists():
        return []

    # Build proper attribution lookup if PA frame is available.
    day_hit_lookup = {}
    if data_dir is not None:
        day_hit_lookup = _build_day_hit_lookup(data_dir, today, t["lookback_days"])
    using_pa_attribution = bool(day_hit_lookup)

    cutoff = today - timedelta(days=t["lookback_days"])
    buckets = t["buckets"]
    in_bucket: list[list[tuple[float, int]]] = [[] for _ in buckets]
    attribution_counts = {"slot_results": 0, "pa": 0, "proxy": 0}
    skipped_pre_deploy = 0
    skipped_other_regime = 0
    try:
        files = sorted(picks_dir.glob("*.json"))
    except OSError as e:
        log.warning(f"could not list {picks_dir}: {e}")
        return []
    candidates: list[tuple[date, dict]] = []
    for f in files:
        if f.name.startswith("._"):
            continue
        if ".shadow." in f.name or "scheduler" in f.name or "streak" in f.name:
            continue
        try:
            body = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        try:
            pick_date = date.fromisoformat(body.get("date", ""))
        except (ValueError, TypeError):
            continue
        if pick_date < cutoff or pick_date > today:
            continue
        candidates.append((pick_date, body))

    # Audit F6: pool by production-regime fingerprint. The current regime is
    # the fingerprint of the NEWEST pick by date — if that pick is unstamped
    # (pre-provenance data, or a partial stamp after an I/O failure), do NOT
    # silently adopt an older pick's regime (Codex review #7); fall back to
    # the wall-clock deploy filter for the whole computation instead. With a
    # regime in hand, a no-regime-change deploy (docs/shadow/ops) no longer
    # erases the accumulated sample, while a genuine policy/feature-env
    # change still resets it.
    current_regime = None
    if candidates:
        newest_body = max(candidates, key=lambda c: c[0])[1]
        current_regime = _regime_fingerprint(newest_body)

    for pick_date, body in candidates:
        if current_regime is not None:
            if _regime_fingerprint(body) != current_regime:
                skipped_other_regime += 1
                continue
        elif since_deploy_iso is not None:
            run_time = body.get("run_time", "")
            if not run_time or _iso_before(run_time, since_deploy_iso):
                skipped_pre_deploy += 1
                continue
        result = body.get("result")
        if result not in ("hit", "miss"):
            continue
        slot_results = body.get("slot_results") or {}
        dd_present = bool(body.get("double_down"))
        # Iterate primary + double_down; each graded slot feeds every bucket
        # whose range AND slot filter it matches. Grading priority
        # (2026-07-12, mirrors predicted_vs_realized): production-graded
        # slot_results first (authoritative live-feed grading), PA-frame join
        # second, day-result proxy last — primary-only AND only on non-DD
        # days (day result misattributes DD picks; the 2026-05-01 bias the
        # old fallback knowingly kept is now excluded instead).
        for slot_key in ("pick", "double_down"):
            outcome = slot_results.get(slot_key)
            if outcome == "void":
                continue
            slot = body.get(slot_key) or {}
            p = slot.get("p_game_hit")
            if p is None:
                continue
            if outcome in ("hit", "miss"):
                day_hit = 1 if outcome == "hit" else 0
                attribution_counts["slot_results"] += 1
            elif using_pa_attribution:
                bid = slot.get("batter_id")
                if bid is None:
                    continue  # PA-frame join needs batter_id
                looked = day_hit_lookup.get((int(bid), pick_date))
                if looked is None:
                    continue  # late data; skip rather than guess
                day_hit = int(looked)
                attribution_counts["pa"] += 1
            elif slot_key == "pick" and not dd_present:
                day_hit = 1 if result == "hit" else 0
                attribution_counts["proxy"] += 1
            else:
                continue
            for bi, spec in enumerate(buckets):
                if spec.get("slots") is not None and slot_key not in spec["slots"]:
                    continue
                if spec["low"] <= p < spec["high"]:
                    in_bucket[bi].append((float(p), day_hit))

    if current_regime is not None:
        deploy_filter = (
            f"regime-fingerprint (skipped {skipped_other_regime} other-regime)"
        )
    elif since_deploy_iso:
        deploy_filter = (
            f"since-deploy={since_deploy_iso[:10]} (skipped {skipped_pre_deploy} pre-deploy)"
        )
    else:
        deploy_filter = "ALL-PICKS (iteration-contaminated)"
    attribution = (
        f"all-slots slot_results:{attribution_counts['slot_results']}"
        f"/pa:{attribution_counts['pa']}/proxy:{attribution_counts['proxy']}"
    )

    alerts: list[Alert] = []
    for bi, spec in enumerate(buckets):
        obs = in_bucket[bi]
        if len(obs) < t["min_bucket_n"]:
            continue
        mean_predicted = sum(p for p, _ in obs) / len(obs)
        realized_rate = sum(h for _, h in obs) / len(obs)
        overconf_pp = (mean_predicted - realized_rate) * 100
        if overconf_pp < t["info_pp"]:
            continue
        if overconf_pp >= t["critical_pp"] and (
            len(obs) >= t["min_bucket_n_critical"]
            or _poisson_binomial_tail_le(
                [p for p, _ in obs], sum(h for _, h in obs)
            ) <= t["critical_tail_epsilon"]
        ):
            level = "CRITICAL"
        elif overconf_pp >= t["warn_pp"] or overconf_pp >= t["critical_pp"]:
            level = "WARN"
        else:
            level = "INFO"
        msg = (
            f"{spec['label']} bucket overconfident by {overconf_pp:+.1f}pp over last "
            f"{t['lookback_days']}d (n={len(obs)}, predicted {mean_predicted:.3f}, "
            f"realized {realized_rate:.3f}, attribution={attribution}, filter={deploy_filter})"
        )
        if level == "CRITICAL":
            msg += ". Real current-model overconfidence signal — investigate distribution shift."
        alerts.append(Alert(
            level=level, source=SOURCE, message=msg,
            # Per-bucket dedup identity: the DD-band and primary-band buckets
            # are distinct incidents for the same-day health-DM dedup.
            incident_key=f"{SOURCE}:{spec['label']}",
        ))
    return alerts
