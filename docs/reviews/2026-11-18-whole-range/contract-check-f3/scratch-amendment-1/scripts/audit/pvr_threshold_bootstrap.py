"""Simulated null for the predicted_vs_realized drift statistic (v2).

Queued by the 2026-07-12 incident review (#9) and reworked after its round-2
review: v1 resampled days with replacement (randomizing the chronological
clustering it claimed to preserve), ignored the 35-calendar-day lookback, and
never simulated the two-consecutive-day attention policy its conclusions
leaned on.

v2 holds the season's OBSERVED schedule fixed — the real dates, each date's
real slot count and stated p's — and simulates only the outcomes as
independent Bernoulli(stated p) (the calibrated null). Window semantics
replicate the check exactly: resolved days within the 35-calendar-day
lookback of each check day, then the last 14/28 of those, slots pooled,
min gates 10/20 days. One check per resolved day (production runs EOD once
per day; off-day EOD re-checks reuse yesterday's window and add no new
alarm surface beyond the same-day DM dedup).

Reported per candidate threshold: season-level P(any exceedance) and — for
the WARN/attention question — P(a calendar-CONSECUTIVE two-day WARN-band
crossing), with binomial 95% CIs.

Production clock (review r3#1): the health suite runs at EOD on every
scheduled game day (skip days included — only no-games days return before
health), and attention requires WARNs on calendar-adjacent days. So the
simulation evaluates the statistic on EVERY calendar day in season (the
windows slide and evict even when no new slot resolves — a skip day can
repeat or CHANGE the value), counts only WARN-BAND crossings
(warn ≤ drift < critical; a CRITICAL does not feed WARN streaks), and
requires calendar adjacency. Approximation: MLB plays essentially daily in
this span, so "every calendar day" ≈ the true health-run calendar.

Caveats (recorded, not solved): same-day DD slots are simulated independent
(they are different-game by construction, so dependence is slate-level and
weak); the null assumes stated p's are honest, and non-stationary true
calibration widens the real alarm surface; n=107 slots is one season.

Usage:
    uv run python scripts/audit/pvr_threshold_bootstrap.py <slot_dataset.csv>
The input CSV (date,slot,p,outcome per graded slot) is rebuilt on the box by
scripts/audit/build_slot_dataset.py; the run below records its sha256.
"""
import csv
import hashlib
import math
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from statistics import mean

LOOKBACK_CAL_DAYS = 35
THRESHOLDS = (0.08, 0.12, 0.15, 0.20, 0.25)
TRIALS = 20_000


def load_days(path):
    by_day = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["outcome"] in ("hit", "miss"):
            by_day[date.fromisoformat(r["date"])].append(float(r["p"]))
    days = sorted(by_day)
    return days, [by_day[d] for d in days]


def drift_series(check_days, resolved_dates, day_slots, day_outcomes):
    """The check's statistic on each CALENDAR check day: resolved days within
    the 35-calendar-day lookback, last 14/28 of those, slots pooled, gates
    10/20. check_days ⊇ resolved_dates — on a day with no new resolved slot
    the windows still slide (eviction can change the value)."""
    out = []
    for d in check_days:
        cutoff = d - timedelta(days=LOOKBACK_CAL_DAYS)
        idx = [i for i, rd in enumerate(resolved_dates) if cutoff <= rd <= d]
        w14, w28 = idx[-14:], idx[-28:]
        if len(w14) < 10 or len(w28) < 20:
            out.append(None)
            continue

        def gap(window):
            ps = [p for i in window for p in day_slots[i]]
            ys = [y for i in window for y in day_outcomes[i]]
            return mean(ps) - mean(ys)

        out.append(gap(w14) - gap(w28))
    return out


def wilson95(k, n):
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "slot_dataset_2026.csv"
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    dates, day_slots = load_days(path)
    n_days = len(dates)
    n_slots = sum(len(d) for d in day_slots)
    print(f"input sha256={digest[:16]}…  {n_days} resolved days "
          f"({dates[0]}..{dates[-1]}), {n_slots} slots")

    # Health checks run every calendar game-day; MLB plays ~daily in-span.
    check_days = [dates[0] + timedelta(days=i)
                  for i in range((dates[-1] - dates[0]).days + 1)]

    random.seed(20260712)
    critical_thr = max(THRESHOLDS)
    any_exceed = {thr: 0 for thr in THRESHOLDS}
    consec_warn_band = {thr: 0 for thr in THRESHOLDS}
    for _ in range(TRIALS):
        outcomes = [[1 if random.random() < p else 0 for p in slots]
                    for slots in day_slots]
        ds = drift_series(check_days, dates, day_slots, outcomes)
        for thr in THRESHOLDS:
            hits = [x is not None and x >= thr for x in ds]
            if any(hits):
                any_exceed[thr] += 1
            # WARN-band only (a CRITICAL does not feed WARN streaks), and
            # calendar-adjacent by construction of check_days.
            band = [x is not None and thr <= x < critical_thr for x in ds]
            if any(a and b for a, b in zip(band, band[1:])):
                consec_warn_band[thr] += 1

    print(f"calibrated-null season alarm surface (fixed observed schedule, "
          f"{len(check_days)} calendar check days, {TRIALS} trials, seed 20260712):")
    print(f"{'thr':>6} {'P(any day ≥ thr)':>24} {'P(2 cal-consec WARN-band)':>30}")
    for thr in THRESHOLDS:
        a, c = any_exceed[thr], consec_warn_band[thr]
        alo, ahi = wilson95(a, TRIALS)
        clo, chi = wilson95(c, TRIALS)
        print(f"{thr:>6.2f} {a / TRIALS:>10.4f} [{alo:.4f},{ahi:.4f}]"
              f" {c / TRIALS:>12.4f} [{clo:.4f},{chi:.4f}]")


if __name__ == "__main__":
    main()
