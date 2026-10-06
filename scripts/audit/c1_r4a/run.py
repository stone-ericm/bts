"""C1 rank 4a, T6: orchestration (registration §§2-4 and the X-E1 freeze-manifest rules; build plan T6).

**Launch (box, through the C1 launcher only, 0.25 CPU-hours each; from `~/projects/bts`):**

    .venv/bin/python -m scripts.audit.c1.launch run --name c1-r4a-fit --cpu-hours 0.25 --max-hours 1 -- \\
        "$HOME/projects/bts/.venv/bin/python" -m scripts.audit.c1_r4a.run --stage fit
    .venv/bin/python -m scripts.audit.c1.launch run --name c1-r4a-evaluate --cpu-hours 0.25 --max-hours 1 -- \\
        "$HOME/projects/bts/.venv/bin/python" -m scripts.audit.c1_r4a.run --stage evaluate

The fit opens at 08:00 ET after contest date 40 and the evaluation at 08:00 ET after date 90. The launcher adds the
cycle caps, the pauses and the sleep-window check.

**1. Admission:**
- **The gate:** the shared one (`scripts.audit.c1.admission`), with exposure row X-32 and scope "calibration fit and
  test".
- **`admission.json`:**
  - it names the reviewed commit, the review report and the exposure commit;
  - its `input_pins` are exactly `{"calendar": <sha256>, "serving_contract": <sha256>}`. Both files are frozen under
    checklist A3, outside the code closure (`CALENDAR_REL`, `CONTRACT_REL`), and X-32 cites the pins' digest.
- **The review report** is an R-specific receipt in the gate's format: one `## Verdict`, `**SIGN.**`, and one
  `Reviewed-commit:` line. A two-item report (the study and the slate) is never the witness.
- **Foreign imports:** every loaded `bts` / `scripts` module must come from this checkout. This is checked after the
  gate, before the claim and before the results (review r1 R6).
- **Fixed roots:** there is no override.

**2. Timing:**
- **The calendar stop:** the C1 calendar stop closes the cycle when the contest-end day ends (00:00 ET after it). It
  takes precedence: a stage not yet run is then recorded inconclusive, with no outcome read (registration §3).
- **Opening:** before then, each stage is refused before its opening.
- **Once:** each stage runs at most once. An earlier claimed run of the stage blocks it, without Eric's exact
  INVALIDATE ruling.

**3. The outcome-free freeze, before the claim (review r1 R2, R3, R5, R8):**
- **The manifest:**
  - the code, its closure blobs, and the registration and reader hashes;
  - the admission and the accepted review;
  - the calendar and the serving contract (their pins and contents);
  - the window, and for the evaluation the accepted fit's binding (point 6).
- **Each window date's served slate:**
  - it is read once, plain or gzipped (`stored`);
  - its exact stored buffers are retained under the run (`slates/`);
  - its population (`populations.json`: the full ordered eligible rows, rank 1, and the exclusions by reason and
    state) is computed from those same bytes.
- **The slate's status:**
  - `absent`;
  - `refused`: unreadable, a conflicting pair, an unsupported schema, or a missing serving witness;
  - `changed`: an evidenced unregistered recipe change, against the serving contract (`provenance`);
  - `ok`.
- **Durability:** the freeze is written durably and holds the hashes of every retained file. Slates are forecasts,
  not outcomes.

**4. The claim:** a durable `CLAIM.json`, bound to the freeze's sha256, precedes the first outcome-bearing read.

**5. Refusals and changes, decided before any outcome read:**
- **Refused:** a `refused` slate refuses acceptance. The stage's result is `refused` (exit 3); no outcome is read and
  no fit or evaluation exists. Rerunning needs Eric's INVALIDATE.
- **Changed:** otherwise, a `changed` slate makes the stage inconclusive ("unregistered recipe change"), with no
  refit, no window reset and no pooling.

**6. The accepted fit (evaluation; review r1 R4):**
- **Before the claim:** exactly one uninvalidated claimed fit run must exist. Its `COMPLETE.json` must bind its claim,
  freeze, calendar, contract and results by sha256. Only that identity and digest are frozen.
- **After the claim:** the fit's `results.json` is read once, checked against the digest, and parsed from the same
  bytes. A fitted result needs a finite float `a`.
- **A changed or invalid result:** the evaluation is refused.
- **A fit without a map** (inconclusive or refused): the evaluation is inconclusive.

**7. Outcomes:**
- **Reading:** for each eligible row's game, the archived feed (`data/raw/2027/<game_pk>.json[.gz]`) is read once. Its
  exact buffers are retained under the run (`feeds/`), and the same bytes are parsed.
- **Unknown games:** a missing, unreadable, conflicting, malformed or unreconciled feed makes the game's rows unknown
  (counted).
- **What is written:** `outcomes.json` (the feed manifest) and `joined.json` (the known rows scored).

**8. Results:**
- **fit:** the MAP intercept on fit dates 1-30, if at least 25 have known eligible rows, with the descriptive
  fit-window interval; otherwise inconclusive.
- **evaluate:** the registered primary, guardrail and dispositions, and the descriptive secondaries
  (`evaluate.secondary`).
- **The counts:** scheduled, present-captured, parsed, eligible, primary-scoreable and rank-1-known dates; the row
  exclusions by reason and state; and the refused and changed dates.
- **Completion:** `COMPLETE.json`, written last, binds every artifact's sha256.
- **Replay:** `replay(run_dir)` recomputes a completed run's results from its retained artifacts alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from scripts.audit.c1 import admission as A
from scripts.audit.c1_r4a import calendar as C
from scripts.audit.c1_r4a import evaluate as E
from scripts.audit.c1_r4a import fit as F
from scripts.audit.c1_r4a import outcomes as O
from scripts.audit.c1_r4a import population as P
from scripts.audit.c1_r4a import provenance as PV
from scripts.audit.c1_r4a import stored as ST

REPO = A.REPO
CODE = A.REPO                     # the checkout whose code runs (foreign imports are checked against it)
DATA = Path.home() / "projects" / "bts" / "data"
ET = ZoneInfo("America/New_York")
ADMISSION_REL = "scripts/audit/c1_r4a/admission.json"
REGISTER_REL = "docs/audit/2026-09-22-exposure-register.md"
REGISTRATION = "docs/sota_audit/2026-10-04-prereg-c1-calibration.md"
CALENDAR_REL = "docs/sota_audit/c1-r4a-calendar-2027.json"
CONTRACT_REL = "docs/sota_audit/c1-r4a-serving-contract-2027.json"
EXPOSURE_ROW = "X-32"
SCOPE = "calibration fit and test"
CLOSURE = ("scripts/__init__.py", "scripts/audit/__init__.py", "scripts/audit/c1", "scripts/audit/c1_r4a", "src/bts",
           "pyproject.toml", "uv.lock", REGISTRATION)
READERS = ("scripts/audit/c1_r4a/calendar.py", "scripts/audit/c1_r4a/population.py", "scripts/audit/c1_r4a/outcomes.py",
           "scripts/audit/c1_r4a/provenance.py", "scripts/audit/c1_r4a/stored.py", "scripts/audit/c1_r4a/fit.py",
           "scripts/audit/c1_r4a/evaluate.py", "scripts/audit/c1_r4a/run.py", "src/bts/data/schema.py")
PINS = ("calendar", "serving_contract")
STAGES = {"fit": 40, "evaluate": 90}
EXIT_REFUSED = 3
HEX64 = re.compile(r"[0-9a-f]{64}")


class Refused(RuntimeError):
    """A post-claim refusal: the stage's result is recorded as refused."""


def now_et() -> datetime:
    return datetime.now(ET)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pins_digest(pins: dict) -> str:
    return sha(json.dumps(pins, sort_keys=True).encode())


def load_admission() -> tuple[dict, str]:
    b = (REPO / ADMISSION_REL).read_bytes()
    return json.loads(b), sha(b)


def admission_gate() -> tuple[str, dict, dict]:
    adm, adm_sha = load_admission()
    pins = adm.get("input_pins")
    if not (isinstance(pins, dict) and set(pins) == set(PINS)
            and all(isinstance(pins[k], str) and HEX64.fullmatch(pins[k]) for k in PINS)):
        raise SystemExit("refusing: admission input_pins must give exactly the calendar's and the serving contract's "
                         "sha256")
    head, reasons = A.admission_check(REPO, adm, closure=CLOSURE, admission_rel=ADMISSION_REL, register_rel=REGISTER_REL,
                                      exposure_row=EXPOSURE_ROW, scope_phrase=SCOPE, inputs_digest=pins_digest(pins))
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head, adm, {**A.accepted_identity(REPO, adm), "admission_sha256": adm_sha}


def foreign_check() -> None:
    bad = A.foreign_imports(CODE)
    if bad:
        raise SystemExit(f"refusing: modules loaded from outside this checkout: {bad[:5]}")


def stage_opens(cal: C.Calendar, stage: str) -> datetime:
    """08:00 ET on the day after the stage's contest date."""
    return datetime.combine(cal.date_of(STAGES[stage]) + timedelta(days=1), time(8, 0), ET)


def cycle_closes(cal: C.Calendar) -> datetime:
    """The C1 calendar stop: the end of the contest's last day (00:00 ET after it)."""
    return datetime.combine(cal.contest_end + timedelta(days=1), time(0, 0), ET)


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _one_level(d: Path) -> None:
    """Create `d` only if its parent exists, and publish the new entry (fsync the parent)."""
    if d.exists():
        return
    try:
        d.mkdir()
    except FileNotFoundError:
        raise SystemExit(f"refusing: {d.parent} does not exist (no recursive creation)") from None
    _fsync_dir(d.parent)


def _dump(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def _write(path: Path, obj) -> str:
    data = _dump(obj)
    A.durable_write(path, data)
    return sha(data)


def _retain(d: Path, files) -> None:
    """Keep the exact consumed buffers under the run (one directory level, fsynced)."""
    _one_level(d)
    for _, name, b in files:
        A.durable_write(d / name, b)


def _blob(path: str) -> str:
    return A._git(CODE, "rev-parse", f"HEAD:{path}", check=False).stdout.strip()


def _run_name(head: str) -> str:
    return f"{head[:7]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def _manifest(stage, head, adm, identity, cal, contract, dates) -> dict:
    return {"stage": stage, "code": head, "admission": adm, "accepted_review": identity,
            "closure": {c: _blob(c) for c in CLOSURE},
            "registration_sha256": sha((CODE / REGISTRATION).read_bytes()),
            "readers": {r: sha((CODE / r).read_bytes()) for r in READERS},
            "calendar": {"path": CALENDAR_REL, "sha256": cal.sha256, "contest_end": cal.contest_end.isoformat(),
                         "closes": cycle_closes(cal).isoformat()},
            "serving_contract": {"path": CONTRACT_REL, "sha256": contract.sha256, "contract": contract.obj},
            "window": [dates[0].isoformat(), dates[-1].isoformat()], "scheduled_dates": len(dates),
            "opens": stage_opens(cal, stage).isoformat()}


# ---- the freeze ---------------------------------------------------------------------------------------------------

def freeze_entry(cal, d, stored, contract) -> tuple[dict, P.Population | None]:
    """Outcome-free: the slate's status, its stored-file manifest and (when `ok`) its population."""
    entry = {"date": d.isoformat(), "number": cal.number(d), "stored": None if stored is None else stored.manifest()}
    if stored is None:
        return {**entry, "status": "absent"}, None
    if stored.error:
        return {**entry, "status": "refused", "reasons": [stored.error]}, None
    try:
        pop = P.population(stored.decoded, d=d)
    except P.PopulationError as exc:
        return {**entry, "status": "refused", "reasons": [str(exc)[:300]]}, None
    status, reasons = PV.check(pop.envelope, d, contract)
    entry.update(slate_sha256=pop.slate_sha256, written_at=pop.written_at, eligible=len(pop.eligible),
                 excluded=pop.excluded, excluded_by_state=pop.excluded_by_state, states=pop.projected,
                 rank1_index=pop.rank1_index, serving=pop.envelope.get("serving"))
    if status == "missing":
        return {**entry, "status": "refused", "reasons": reasons}, None
    return {**entry, "status": status, "reasons": reasons}, (pop if status == "ok" else None)


def population_record(pops: dict) -> dict:
    return {d.isoformat(): {"eligible": p.eligible, "rank1_index": p.rank1_index, "excluded": p.excluded,
                            "excluded_by_state": p.excluded_by_state, "states": p.projected,
                            "slate_sha256": p.slate_sha256, "written_at": p.written_at}
            for d, p in sorted(pops.items())}


# ---- the accepted fit ---------------------------------------------------------------------------------------------

def accepted_fit_binding(fit_root: Path, register_text: str, cal_sha: str, contract_sha: str) -> dict:
    """Outcome-free: the one uninvalidated claimed fit run and the digest of its sealed results (review r1 R4)."""
    claimed = sorted(d for d in fit_root.iterdir() if d.is_dir() and (d / "CLAIM.json").exists()) \
        if fit_root.exists() else []
    live = []
    for d in claimed:
        claim_sha = sha((d / "CLAIM.json").read_bytes())
        inv = fit_root / f"INVALIDATION_{d.name}.json"
        try:
            invalidated = inv.exists() and not A.invalidation_problems(json.loads(inv.read_text()), d.name, claim_sha,
                                                                        register_text)
        except ValueError:
            invalidated = False
        if not invalidated:
            live.append(d)
    if len(live) != 1:
        raise SystemExit(f"refusing: evaluation needs exactly one uninvalidated fit claim, found {len(live)}")
    d = live[0]
    try:
        complete_raw = (d / "COMPLETE.json").read_bytes()
        claim_raw, freeze_raw = (d / "CLAIM.json").read_bytes(), (d / "freeze.json").read_bytes()
        rec, claim, freeze = json.loads(complete_raw), json.loads(claim_raw), json.loads(freeze_raw)
    except (OSError, ValueError):
        raise SystemExit(f"refusing: the fit run {d.name} has no readable completion record") from None
    ok = (isinstance(rec, dict) and isinstance(claim, dict) and isinstance(freeze, dict)
          and rec.get("stage") == "fit" and rec.get("run") == d.name and freeze.get("stage") == "fit"
          and rec.get("claim_sha256") == sha(claim_raw) and rec.get("freeze_sha256") == sha(freeze_raw)
          and claim.get("freeze_sha256") == sha(freeze_raw) and claim.get("run") == d.name
          and rec.get("code") == freeze.get("code") == claim.get("code")
          and rec.get("calendar_sha256") == cal_sha and rec.get("contract_sha256") == contract_sha
          and isinstance(rec.get("results_sha256"), str) and HEX64.fullmatch(rec["results_sha256"]) is not None)
    if not ok:
        raise SystemExit(f"refusing: the fit run {d.name}'s completion record does not bind its claim, freeze, "
                         "calendar, serving contract and results")
    return {"run": d.name, "complete_sha256": sha(complete_raw), "claim_sha256": rec["claim_sha256"],
            "results_sha256": rec["results_sha256"]}


def load_fit_result(fit_root: Path, binding: dict) -> dict:
    """After the evaluation's claim: the bound fit result, read once, checked and parsed from the same bytes."""
    try:
        raw = (fit_root / binding["run"] / "results.json").read_bytes()
    except OSError as exc:
        raise Refused(f"the bound fit result is unreadable: {exc}") from None
    if sha(raw) != binding["results_sha256"]:
        raise Refused("the bound fit result changed after its completion record")
    try:
        res = json.loads(raw)
    except ValueError:
        raise Refused("the bound fit result is not JSON") from None
    a = res.get("a") if isinstance(res, dict) else None
    ok = isinstance(res, dict) and res.get("stage") == "fit" and (
        (res.get("status") == "fitted" and type(a) is float and math.isfinite(a))
        or (res.get("status") in ("inconclusive", "refused") and a is None))
    if not ok:
        raise Refused("the bound fit result is not a valid fit result")
    return res


# ---- outcomes and scoring -----------------------------------------------------------------------------------------

def join_outcomes(pops: dict, read_feed, retain_dir: Path | None) -> tuple[dict, list, dict]:
    """{date: [known row, ...]}, the feed manifest, and the outcome exclusions (by outcome, by state, rank 1).
    `read_feed(game_pk)` returns the game's `stored.Stored` or None."""
    games, manifest = {}, []
    excl, excl_state, rank1_excl = Counter(), Counter(), Counter()
    joined = {}
    for d, pop in sorted(pops.items()):
        rows = []
        for i, r in enumerate(pop.eligible):
            pk = r["game_pk"]
            if pk not in games:
                games[pk] = _game(pk, read_feed(pk), manifest, retain_dir)
            g = games[pk]
            outcome = g.outcome(r["batter_id"]) if g is not None else "unknown"
            if outcome in ("hit", "no_hit"):
                rows.append({"batter_id": r["batter_id"], "game_pk": pk, "p": r["p_game_hit"],
                             "y": 1 if outcome == "hit" else 0, "rank1": i == pop.rank1_index})
            else:
                excl[outcome] += 1
                excl_state[f"{outcome}|{P.state(r)}"] += 1
                if i == pop.rank1_index:
                    rank1_excl[outcome] += 1
        joined[d.isoformat()] = rows
    return joined, manifest, {"by_outcome": dict(excl), "by_state": dict(excl_state), "rank1": dict(rank1_excl)}


def _game(pk, stored, manifest, retain_dir):
    entry = {"game_pk": pk, "stored": None if stored is None else stored.manifest()}
    manifest.append(entry)
    if stored is None:
        entry["status"] = "absent"
        return None
    if retain_dir is not None:
        _retain(retain_dir, stored.files)
    if stored.error:
        entry["status"] = "unreadable"
        return None
    try:
        g = O.game_outcomes(stored.decoded, game_pk=pk)
    except O.OutcomeError as exc:
        entry.update(status="refused", reason=str(exc)[:300])
        return None
    entry.update(status="complete" if g.complete else "incomplete", reason=g.reason)
    return g


def _per_date(joined: dict) -> list:
    """[(p, y, rank-1 position or None)] for each date with known rows, in date order."""
    out = []
    for _, rows in sorted(joined.items()):
        if rows:
            r1 = next((i for i, r in enumerate(rows) if r["rank1"]), None)
            out.append(([r["p"] for r in rows], [r["y"] for r in rows], r1))
    return out


def score(stage: str, joined: dict, fit_a: float | None) -> dict:
    per = _per_date(joined)
    if stage == "fit":
        if len(per) < F.MIN_FIT_DATES:
            return {"stage": "fit", "status": "inconclusive", "reason": "insufficient support", "a": None,
                    "fit_dates": len(per)}
        dates = [(p, y) for p, y, _ in per]
        a = F.fit_intercept(dates)
        if not math.isfinite(a):
            raise RuntimeError("non-finite MAP")
        return {"stage": "fit", "status": "fitted", "a": float(a), "fit_dates": len(per),
                "interval_descriptive": F.posterior_interval(dates, a)}
    diffs = E.date_differences([(p, y) for p, y, _ in per], a=fit_a)
    r1_diffs = E.date_differences([([p[r]], [y[r]]) for p, y, r in per if r is not None], a=fit_a)
    lo, hi = E.bootstrap_interval(diffs) if diffs else (None, None)
    guard = E.guardrail_p95(r1_diffs) if r1_diffs else None
    mean = float(np.mean(diffs)) if diffs else None
    disp = E.disposition(diff=mean if mean is not None else float("nan"), upper=hi if hi is not None else float("nan"),
                         guard_p95=guard if guard is not None else float("nan"), n_dates=len(diffs),
                         n_rank1=len(r1_diffs))
    return {"stage": "evaluate", "status": "evaluated", "a": fit_a,
            "primary": {"mean_difference": mean, "interval": [lo, hi], "dates": len(diffs)},
            "guardrail": {"p95": guard, "dates": len(r1_diffs)}, **disp,
            "secondary_descriptive": E.secondary(per, a=fit_a)}


def counts(scheduled: int, entries, pops, joined=None, out_excl=None) -> dict:
    """The registered counts (§3). Without `joined` (a stage stopped before any outcome read) only the outcome-free
    counts are given."""
    pop_excl, pop_excl_state, states = Counter(), Counter(), Counter()
    for p in pops.values():
        pop_excl.update(p.excluded)
        pop_excl_state.update(p.excluded_by_state)
        states.update(p.projected)
    out = {"scheduled": scheduled, "present_captured": sum(e["status"] != "absent" for e in entries),
           "parsed": len(pops), "refused": sum(e["status"] == "refused" for e in entries),
           "changed": sum(e["status"] == "changed" for e in entries),
           "eligible_dates": sum(1 for p in pops.values() if p.eligible),
           "eligible_rows": sum(len(p.eligible) for p in pops.values()), "eligible_by_state": dict(states),
           "population_exclusions": dict(pop_excl), "population_exclusions_by_state": dict(pop_excl_state)}
    if joined is not None:
        known = {d: rows for d, rows in joined.items() if rows}
        out.update(primary_scoreable_dates=len(known),
                   rank1_known_dates=sum(1 for rows in known.values() if any(r["rank1"] for r in rows)),
                   known_rows=sum(len(rows) for rows in known.values()), outcome_exclusions=out_excl)
    return out


# ---- the stage ----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    args = ap.parse_args(argv)
    head, adm, identity = admission_gate()
    foreign_check()
    cal = C.load((REPO / CALENDAR_REL).read_bytes(), adm["input_pins"]["calendar"])
    contract = PV.load((REPO / CONTRACT_REL).read_bytes(), adm["input_pins"]["serving_contract"])
    now = now_et()
    closed = now >= cycle_closes(cal)
    if not closed and now < stage_opens(cal, args.stage):
        raise SystemExit(f"refusing: the {args.stage} stage opens at {stage_opens(cal, args.stage).isoformat()}")
    out_root = DATA / "hetzner_results" / "c1" / "r4a"
    stage_root = out_root / args.stage
    _one_level(out_root)              # hetzner_results/c1 must already exist (no recursive creation, rank-3 r3 R3-6)
    register_text = (REPO / REGISTER_REL).read_text()
    with A.admission_lock(stage_root):
        blocked = A.claimed_runs(stage_root, register_text)
        if blocked:
            raise SystemExit(f"refusing: an earlier claimed {args.stage} run without Eric's invalidation: {blocked}")
        return _run(args.stage, head, adm, identity, cal, contract, stage_root, out_root, register_text, closed)


def _claim(run_dir: Path, head: str, freeze_sha: str) -> str:
    foreign_check()
    rec = {"run": run_dir.name, "code": head, "pid": os.getpid(), "claimed_utc": datetime.now(timezone.utc).isoformat(),
           "freeze_sha256": freeze_sha}
    data = (json.dumps(rec) + "\n").encode()
    A.durable_write(run_dir / "CLAIM.json", data)
    return sha(data)


def _finish(run_dir, stage, head, cal, contract, hashes: dict, res: dict) -> int:
    foreign_check()
    hashes["results_sha256"] = _write(run_dir / "results.json", res)
    _write(run_dir / "COMPLETE.json", {"stage": stage, "run": run_dir.name, "code": head, **hashes,
                                       "calendar_sha256": cal.sha256, "contract_sha256": contract.sha256})
    print(f"[c1-4a {stage}] {json.dumps({k: res[k] for k in res if k != 'secondary_descriptive'}, default=str)}",
          file=sys.stderr)
    return EXIT_REFUSED if res.get("status") == "refused" else 0


def _run(stage, head, adm, identity, cal, contract, stage_root, out_root, register_text, closed) -> int:
    dates = cal.fit_dates() if stage == "fit" else cal.test_dates()
    manifest = _manifest(stage, head, adm, identity, cal, contract, dates)
    if closed:                                       # the C1 calendar stop: nothing is read
        run_dir = A.make_run_dir(stage_root, _run_name(head))
        freeze_sha = _write(run_dir / "freeze.json", {**manifest, "calendar_stop": True})
        claim_sha = _claim(run_dir, head, freeze_sha)
        return _finish(run_dir, stage, head, cal, contract, {"claim_sha256": claim_sha, "freeze_sha256": freeze_sha}, {
            "stage": stage, "status": "inconclusive", "disposition": "inconclusive", "a": None,
            "reason": f"the C1 calendar stop: the cycle closed at {cycle_closes(cal).isoformat()}"})
    binding = accepted_fit_binding(out_root / "fit", register_text, cal.sha256, contract.sha256) \
        if stage == "evaluate" else None
    stored = {d: ST.read(DATA / "picks" / "slates" / d.isoformat()) for d in dates}
    run_dir = A.make_run_dir(stage_root, _run_name(head))
    entries, pops = [], {}
    for d in dates:
        if stored[d] is not None:
            _retain(run_dir / "slates", stored[d].files)
        entry, pop = freeze_entry(cal, d, stored[d], contract)
        entries.append(entry)
        if pop is not None:
            pops[d] = pop
    pop_sha = _write(run_dir / "populations.json", population_record(pops))
    refused = [e for e in entries if e["status"] == "refused"]
    changed = [e for e in entries if e["status"] == "changed"]
    freeze_sha = _write(run_dir / "freeze.json", {**manifest, "slates": entries, "populations_sha256": pop_sha,
                                                  "fit": binding})
    hashes = {"freeze_sha256": freeze_sha, "populations_sha256": pop_sha}
    hashes["claim_sha256"] = _claim(run_dir, head, freeze_sha)     # before the first outcome-bearing read
    base = {"stage": stage, "a": None, "counts": counts(len(dates), entries, pops)}
    if refused:
        return _finish(run_dir, stage, head, cal, contract, hashes, {
            **base, "status": "refused", "disposition": "refused",
            "reason": "a present slate is unsupported or lacks its serving witness (X-E1)",
            "refusals": [{"date": e["date"], "reasons": e["reasons"]} for e in refused]})
    if changed:
        return _finish(run_dir, stage, head, cal, contract, hashes, {
            **base, "status": "inconclusive", "disposition": "inconclusive", "reason": "unregistered recipe change",
            "changes": [{"date": e["date"], "reasons": e["reasons"]} for e in changed]})
    fit_a = None
    if stage == "evaluate":
        try:
            fit = load_fit_result(out_root / "fit", binding)
        except Refused as exc:
            return _finish(run_dir, stage, head, cal, contract, hashes, {
                **base, "status": "refused", "disposition": "refused", "reason": str(exc)})
        if fit["status"] != "fitted":
            return _finish(run_dir, stage, head, cal, contract, hashes, {
                **base, "status": "inconclusive", "disposition": "inconclusive", "reason": "no fitted map"})
        fit_a = fit["a"]
    feeds_dir = DATA / "raw" / "2027"
    _one_level(run_dir / "feeds")
    joined, feeds, out_excl = join_outcomes(pops, lambda pk: ST.read(feeds_dir / str(pk)), run_dir / "feeds")
    hashes["outcomes_sha256"] = _write(run_dir / "outcomes.json", {"feeds": feeds, "excluded": out_excl})
    hashes["joined_sha256"] = _write(run_dir / "joined.json", joined)
    res = {**score(stage, joined, fit_a), "counts": counts(len(dates), entries, pops, joined, out_excl)}
    return _finish(run_dir, stage, head, cal, contract, hashes, res)


# ---- replay -------------------------------------------------------------------------------------------------------

def _retained(d: Path, stored_manifest: dict) -> ST.Stored:
    """Rebuild a Stored from the run's retained buffers, checking each against its manifest sha256."""
    out = ST.Stored()
    for f in stored_manifest["files"]:
        b = (d / f["file"]).read_bytes()
        if sha(b) != f["sha256"]:
            raise RuntimeError(f"retained {f['file']} does not match its manifest")
        out.files.append((f["format"], f["file"], b))
    out.used, out.error, out.note = stored_manifest["used"], stored_manifest["error"], stored_manifest["note"]
    if out.used is not None:
        out.decoded = ST._decode(out.used, next(b for fmt, _, b in out.files if fmt == out.used))
        if sha(out.decoded) != stored_manifest["decoded_sha256"]:
            raise RuntimeError("a retained buffer does not decode to its manifest")
    return out


def replay(run_dir: Path) -> dict:
    """A completed run's results, recomputed from its retained artifacts alone (no source archive is read). The
    populations and joined rows must reproduce their frozen hashes; the result is returned for comparison."""
    complete = json.loads((run_dir / "COMPLETE.json").read_bytes())
    if "joined_sha256" not in complete:
        raise RuntimeError("replay covers scored runs; this run stopped before any outcome read")
    freeze_raw = (run_dir / "freeze.json").read_bytes()
    if sha(freeze_raw) != complete["freeze_sha256"]:
        raise RuntimeError("the freeze does not match the completion record")
    freeze = json.loads(freeze_raw)
    contract = PV.Contract(freeze["serving_contract"]["contract"], freeze["serving_contract"]["sha256"])
    pops = {}
    for e in freeze["slates"]:
        if e["status"] == "ok":
            d = date.fromisoformat(e["date"])
            pop = P.population(_retained(run_dir / "slates", e["stored"]).decoded, d=d)
            if PV.check(pop.envelope, d, contract)[0] != "ok":
                raise RuntimeError(f"{e['date']}: the retained slate no longer checks ok")
            pops[d] = pop
    if sha(_dump(population_record(pops))) != complete["populations_sha256"]:
        raise RuntimeError("the recomputed populations differ from the frozen ones")
    by_pk = {f["game_pk"]: f["stored"] for f in json.loads((run_dir / "outcomes.json").read_bytes())["feeds"]}
    joined, _, out_excl = join_outcomes(
        pops, lambda pk: None if by_pk.get(pk) is None else _retained(run_dir / "feeds", by_pk[pk]), None)
    if sha(_dump(joined)) != complete["joined_sha256"]:
        raise RuntimeError("the recomputed joined rows differ from the frozen ones")
    fit_a = load_fit_result(run_dir.parent.parent / "fit", freeze["fit"])["a"] if freeze["stage"] == "evaluate" \
        else None
    res = {**score(freeze["stage"], joined, fit_a),
           "counts": counts(freeze["scheduled_dates"], freeze["slates"], pops, joined, out_excl)}
    return json.loads(_dump(res))


if __name__ == "__main__":
    sys.exit(main())
