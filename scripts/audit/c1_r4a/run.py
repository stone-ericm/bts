"""C1 rank 4a, T6: orchestration (registration §§2-3 and the X-E1 freeze-manifest rules; build plan T6).

    .venv/bin/python -m scripts.audit.c1_r4a.run --stage fit        # at 08:00 ET after contest date 40, once
    .venv/bin/python -m scripts.audit.c1_r4a.run --stage evaluate   # at 08:00 ET after contest date 90, once

**1. Admission:** the shared gate (`scripts.audit.c1.admission`), with exposure row X-32 and scope "calibration fit
and test".
- `admission.json` names the reviewed commit, the review report and the exposure commit, plus `input_pins`:
  `{"calendar": <sha256>}`, for the calendar frozen under checklist A3 (`CALENDAR_REL`, outside the code closure).
- X-32 must cite the pins' digest. The record is read once.
- The roots are fixed; there is no override.

**2. Timing:** each stage is refused before its moment, 08:00 ET on the day after contest date 40 (fit) or 90
(evaluate). Each stage runs at most once: an earlier claimed run of the stage blocks it, without Eric's exact
INVALIDATE ruling.

**3. The outcome-free freeze, before the claim:**
- the code, the accepted identity, the calendar and the window;
- each window date's served slate, as exact bytes read once, with its sha256 and size, or absent;
- the outcome-free population of each slate (eligible rows, exclusions, rank 1).

It is written durably. Slates are forecasts, not outcomes.

**4. Claim:** a durable `CLAIM.json` precedes the first outcome-bearing read.

**5. Outcomes:** for each eligible row's game, the archived feed (`data/raw/2027/<game_pk>.json`) is read once and
hashed, and the same bytes are parsed. Its hash goes in `outcomes.json`. A missing or unreadable feed makes that
game's rows unknown (counted).

**6. Results:**
- **fit:** the MAP intercept on fit dates 1-30, if at least 25 have known eligible rows; otherwise inconclusive.
- **evaluate:** consumes exactly the one accepted fit run's `results.json` (bound by sha256), then the registered
  primary, guardrail and dispositions, plus the descriptive secondaries.

Every count is reported: scheduled, captured, eligible, scoreable, rank-1-known, and the exclusions by reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from scripts.audit.c1 import admission as A
from scripts.audit.c1_r4a import calendar as C
from scripts.audit.c1_r4a import evaluate as E
from scripts.audit.c1_r4a import fit as F
from scripts.audit.c1_r4a import outcomes as O
from scripts.audit.c1_r4a import population as P

REPO = A.REPO
DATA = Path.home() / "projects" / "bts" / "data"
ET = ZoneInfo("America/New_York")
ADMISSION_REL = "scripts/audit/c1_r4a/admission.json"
REGISTER_REL = "docs/audit/2026-09-22-exposure-register.md"
REGISTRATION = "docs/sota_audit/2026-10-04-prereg-c1-calibration.md"
CALENDAR_REL = "docs/sota_audit/c1-r4a-calendar-2027.json"
EXPOSURE_ROW = "X-32"
SCOPE = "calibration fit and test"
CLOSURE = ("scripts/__init__.py", "scripts/audit/__init__.py", "scripts/audit/c1", "scripts/audit/c1_r4a", "src/bts",
           "pyproject.toml", "uv.lock", REGISTRATION)
STAGES = {"fit": 40, "evaluate": 90}
HEX64 = __import__("re").compile(r"[0-9a-f]{64}")


def now_et() -> datetime:
    return datetime.now(ET)


def pins_digest(pins: dict) -> str:
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def load_admission() -> tuple[dict, str]:
    b = (REPO / ADMISSION_REL).read_bytes()
    return json.loads(b), hashlib.sha256(b).hexdigest()


def admission_gate() -> tuple[str, dict, dict]:
    adm, adm_sha = load_admission()
    pins = adm.get("input_pins")
    if not (isinstance(pins, dict) and set(pins) == {"calendar"} and HEX64.fullmatch(str(pins["calendar"]))):
        raise SystemExit("refusing: admission input_pins must give exactly the calendar's sha256")
    head, reasons = A.admission_check(REPO, adm, closure=CLOSURE, admission_rel=ADMISSION_REL, register_rel=REGISTER_REL,
                                      exposure_row=EXPOSURE_ROW, scope_phrase=SCOPE, inputs_digest=pins_digest(pins))
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head, adm, {**A.accepted_identity(REPO, adm), "admission_sha256": adm_sha}


def stage_opens(cal: C.Calendar, stage: str) -> datetime:
    """08:00 ET on the day after the stage's contest date."""
    d = cal.date_of(STAGES[stage])
    return datetime.combine(d + timedelta(days=1), time(8, 0), ET)


def _one_level(d: Path) -> None:
    """Create `d` only if its parent exists, and publish the new entry (fsync the parent)."""
    import os
    if d.exists():
        return
    try:
        d.mkdir()
    except FileNotFoundError:
        raise SystemExit(f"refusing: {d.parent} does not exist (no recursive creation)") from None
    fd = os.open(d.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, obj) -> str:
    data = (json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n").encode()
    A.durable_write(path, data)
    return hashlib.sha256(data).hexdigest()


def freeze_slates(cal: C.Calendar, dates, slates_dir: Path) -> tuple[list, dict]:
    """Outcome-free: each window date's slate bytes (read once, hashed) and its population."""
    inventory, pops = [], {}
    for d in dates:
        p = slates_dir / f"{d.isoformat()}.json"
        entry = {"date": d.isoformat(), "number": cal.number(d), "path": str(p.relative_to(slates_dir.parent.parent)),
                 "present": p.exists()}
        if p.exists():
            raw = p.read_bytes()
            entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            try:
                pop = P.population(raw, d=d)
            except P.PopulationError as exc:
                entry["refused"] = str(exc)[:300]
            else:
                pops[d] = pop
                entry.update(eligible=len(pop.eligible), excluded=pop.excluded, projected=pop.projected,
                             rank1={k: pop.rank1[k] for k in ("batter_id", "game_pk", "p_game_hit")} if pop.rank1 else None)
        inventory.append(entry)
    return inventory, pops


def join_outcomes(pops: dict, feeds_dir: Path) -> tuple[dict, list, Counter]:
    """{date: [(p, y, is_rank1), ...]} over known outcomes, the feed inventory, and the outcome exclusions."""
    feeds: dict = {}
    inventory, excluded = [], Counter()
    joined: dict = {}
    for d, pop in sorted(pops.items()):
        rows = []
        for r in pop.eligible:
            pk = r["game_pk"]
            if pk not in feeds:
                f = feeds_dir / f"{pk}.json"
                if not f.exists():
                    feeds[pk] = None
                    inventory.append({"game_pk": pk, "present": False})
                else:
                    raw = f.read_bytes()
                    inventory.append({"game_pk": pk, "present": True, "sha256": hashlib.sha256(raw).hexdigest()})
                    try:
                        feeds[pk] = O.game_outcomes(raw, game_pk=pk)
                    except O.OutcomeError as exc:
                        feeds[pk] = None
                        inventory[-1]["refused"] = str(exc)[:300]
            g = feeds[pk]
            outcome = g.outcome(r["batter_id"]) if g is not None else "unknown"
            if outcome in ("hit", "no_hit"):
                rows.append((float(r["p_game_hit"]), 1 if outcome == "hit" else 0, r is pop.rank1))
            else:
                excluded[outcome] += 1
                if r is pop.rank1:
                    excluded[f"rank1_{outcome}"] += 1
        joined[d] = rows
    return joined, inventory, excluded


def _secondary(dates_rows, a: float) -> dict:
    """Descriptive only: the Brier difference, stated minus realized (before and after the map; all rows and rank-1
    rows), and a reliability table by decile of p."""
    p = np.array([x[0] for rows in dates_rows for x in rows])
    y = np.array([x[1] for rows in dates_rows for x in rows], dtype=float)
    r1 = np.array([x[2] for rows in dates_rows for x in rows], dtype=bool)
    if p.size == 0:
        return {"rows": 0}
    m = F.apply_map(p, a)
    out = {"rows": int(p.size), "brier_identity": float(np.mean((p - y) ** 2)), "brier_map": float(np.mean((m - y) ** 2)),
           "stated_minus_realized": {"identity": float(p.mean() - y.mean()), "map": float(m.mean() - y.mean())}}
    if r1.any():
        out["rank1_stated_minus_realized"] = {"identity": float(p[r1].mean() - y[r1].mean()),
                                              "map": float(m[r1].mean() - y[r1].mean()), "n": int(r1.sum())}
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    out["reliability_deciles"] = [{"decile": int(b), "n": int((bins == b).sum()),
                                   "mean_p": float(p[bins == b].mean()), "realized": float(y[bins == b].mean())}
                                  for b in range(10) if (bins == b).any()]
    return out


def accepted_fit(fit_root: Path, register_text: str) -> tuple[dict, str, str]:
    """The one accepted fit run: claimed, completed (results.json), not invalidated. Returns (results, sha, run)."""
    runs = [d for d in sorted(fit_root.iterdir()) if d.is_dir() and (d / "CLAIM.json").exists()] if fit_root.exists() else []
    good = []
    for d in runs:
        claim_sha = hashlib.sha256((d / "CLAIM.json").read_bytes()).hexdigest()
        inv = fit_root / f"INVALIDATION_{d.name}.json"
        invalidated = inv.exists() and not A.invalidation_problems(json.loads(inv.read_text()), d.name, claim_sha,
                                                                    register_text)
        if not invalidated and (d / "results.json").exists():
            good.append(d)
    if len(good) != 1:
        raise SystemExit(f"refusing: evaluation needs exactly one accepted fit run, found {len(good)}")
    raw = (good[0] / "results.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), good[0].name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    args = ap.parse_args(argv)
    head, adm, identity = admission_gate()
    raw_cal = (REPO / CALENDAR_REL).read_bytes()
    cal = C.load(raw_cal, adm["input_pins"]["calendar"])
    opens = stage_opens(cal, args.stage)
    if now_et() < opens:
        raise SystemExit(f"refusing: the {args.stage} stage opens at {opens.isoformat()}")
    out_root = DATA / "hetzner_results" / "c1" / "r4a"
    stage_root = out_root / args.stage
    _one_level(out_root)              # hetzner_results/c1 must already exist (no recursive creation, rank-3 r3 R3-6)
    register_text = (REPO / REGISTER_REL).read_text()
    with A.admission_lock(stage_root):
        blocked = A.claimed_runs(stage_root, register_text)
        if blocked:
            raise SystemExit(f"refusing: an earlier claimed {args.stage} run without Eric's invalidation: {blocked}")
        return _run(args.stage, head, adm, identity, cal, stage_root, out_root, register_text)


def _run(stage, head, adm, identity, cal, stage_root, out_root, register_text) -> int:
    dates = cal.fit_dates() if stage == "fit" else cal.test_dates()
    fit_ref = None
    if stage == "evaluate":
        fit_results, fit_sha, fit_run = accepted_fit(out_root / "fit", register_text)
        fit_ref = {"run": fit_run, "results_sha256": fit_sha, "a": fit_results.get("a"),
                   "inconclusive": fit_results.get("inconclusive")}
    inventory, pops = freeze_slates(cal, dates, DATA / "picks" / "slates")
    run_dir = A.make_run_dir(stage_root, f"{head[:7]}-{now_et().astimezone(ZoneInfo('UTC')):%Y%m%dT%H%M%SZ}")
    _write(run_dir / "freeze.json", {"stage": stage, "code": head, "admission": adm, "accepted_review": identity,
                                     "calendar": {"path": CALENDAR_REL, "sha256": cal.sha256},
                                     "window": [dates[0].isoformat(), dates[-1].isoformat()] if dates else None,
                                     "scheduled_dates": len(dates), "slates": inventory, "fit": fit_ref})
    A.write_claim(run_dir, head)                                   # before the first outcome-bearing read
    joined, feeds, out_excl = join_outcomes(pops, DATA / "raw" / "2027")
    _write(run_dir / "outcomes.json", {"feeds": feeds, "excluded": dict(out_excl)})
    known = {d: rows for d, rows in joined.items() if rows}
    counts = {"scheduled": len(dates), "captured": len(pops), "eligible_rows": sum(len(p.eligible) for p in pops.values()),
              "scoreable_dates": len(known), "rank1_known_dates": sum(1 for rows in known.values() if any(r[2] for r in rows)),
              "population_exclusions": dict(sum((Counter(p.excluded) for p in pops.values()), Counter())),
              "outcome_exclusions": dict(out_excl)}
    if stage == "fit":
        if len(known) < F.MIN_FIT_DATES:
            res = {"stage": "fit", "inconclusive": True, "reason": "insufficient support", "a": None, "counts": counts}
        else:
            a = F.fit_intercept([([r[0] for r in rows], [r[1] for r in rows]) for rows in known.values()])
            res = {"stage": "fit", "inconclusive": False, "a": a, "counts": counts}
    else:
        if fit_ref["inconclusive"] or fit_ref["a"] is None:
            res = {"stage": "evaluate", "disposition": "inconclusive", "reason": "no fitted map", "counts": counts}
        else:
            a = float(fit_ref["a"])
            per_date = [([r[0] for r in rows], [r[1] for r in rows]) for rows in known.values()]
            diffs = E.date_differences(per_date, a=a)
            r1 = [([r[0]], [r[1]]) for rows in known.values() for r in rows if r[2]]
            r1_diffs = E.date_differences(r1, a=a) if r1 else []
            lo, hi = E.bootstrap_interval(diffs) if diffs else (None, None)
            guard = E.guardrail_p95(r1_diffs) if r1_diffs else None
            disp = E.disposition(diff=float(np.mean(diffs)) if diffs else float("nan"), upper=hi if hi is not None else float("nan"),
                                 guard_p95=guard if guard is not None else float("nan"), n_dates=len(diffs), n_rank1=len(r1_diffs))
            res = {"stage": "evaluate", "a": a, "primary": {"mean_difference": float(np.mean(diffs)) if diffs else None,
                                                            "interval": [lo, hi]},
                   "guardrail_p95": guard, **disp, "counts": counts,
                   "secondary_descriptive": _secondary(known.values(), a)}
    _write(run_dir / "results.json", res)
    print(f"[c1-4a {stage}] {json.dumps({k: res[k] for k in res if k != 'secondary_descriptive'}, default=str)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
