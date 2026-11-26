"""P-05 audit: the E[season-best] tail policy, 2026-09-03 → 2026-09-27, on the frozen W0.7 snapshot.

Protocol (docs/audit/2026-09-22-exposure-register.md §B P-05): design
docs/audit/2026-09-03-emax-tail-policy.md §3 — "Stop rule, explicit: skip iff
min(57, s + 2d) <= m"; first-day acceptance = decision.json objective=emax_season_best,
best_status=trusted, effective_best=18, degraded_reason=null. W1.4b obligation: audit
objective/state/provenance per date 9/03→9/27, stop behaviour on 9/19,
delivered/entered/private separation. A few dates validate the mechanism, not the
rates or E[best] optimality — this reader reports no hit rates.

Per date it joins decision.json (v3), the pick file (if any), every scheduler journal
`Policy:` line, the contest ledger (what the contest itself recorded as entered) and the
frozen rounds calendar, then checks the design's rules. Nothing is written except --out.

Run on the box:  .venv/bin/python scripts/audit/p05_tail_policy_audit.py --out /tmp/p05.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

DEFAULT_SNAPSHOT = Path("data/hetzner_results/season_2026_snapshot/final-20260928")
WINDOW = ("2026-09-03", "2026-09-27")
SEASON_END = "2026-09-27"
TARGET = 57
POLICY_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T\S+ .*Policy: objective=(\S+) action=(\S+) streak=(\d+) best=(\d+) "
    r"\((\w+)\) effective_best=(\d+) days=(\d+) tail=(\S+)")


def expected_objective(streak: int, days: int) -> str:
    """Design §1: reach57 iff streak + 2*days >= 57 (from state alone), else E[season-best]."""
    return "reach57" if streak + 2 * days >= TARGET else "emax_season_best"


def stop_rule_skip(streak: int, days: int, best: int) -> bool:
    """Design §3: skip iff min(57, s + 2d) <= m."""
    return min(TARGET, streak + 2 * days) <= best


def parse_policy_line(line: str) -> dict | None:
    m = POLICY_RE.match(line)
    if not m:
        return None
    d, obj, act, s, b, st, eb, days, tail = m.groups()
    return {"date": d, "objective": obj, "action": act, "streak": int(s), "best": int(b),
            "best_status": st, "effective_best": int(eb), "days": int(days), "tail": tail}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-05 tail-policy audit")
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    snap = a.snapshot
    picks = snap / "data" / "picks"

    # artifact binding
    tail_path, base_path = snap / "data/models/mdp_tail_policy.npz", snap / "data/models/mdp_policy.npz"
    npz = np.load(tail_path, allow_pickle=False)
    artifact = {"tail_sha256": _sha(tail_path), "base_sha256": _sha(base_path),
                "tail_declares_base_sha256": str(npz["base_policy_sha256"]),
                "tail_objective": str(npz["objective"]), "tail_schema": str(npz["schema_version"])}
    artifact["binding_ok"] = artifact["tail_declares_base_sha256"] == artifact["base_sha256"]

    # calendar (frozen rounds capture) and what the contest recorded as entered
    rounds_file = sorted(glob.glob(str(snap / "data/leaderboard/static_snapshots/rounds/*.json.gz")))[-1]
    rounds = {x["date"][:10]: x["id"] for x in json.loads(gzip.open(rounds_file).read())["rounds"]}
    ledger_last = None
    for line in open(picks / "account_state/contest_ledger.jsonl"):
        ledger_last = line
    contest = {p["roundId"]: p for p in json.loads(ledger_last)["predictions"]}

    journal: dict[str, list[dict]] = {}
    for line in open(snap / "journal_bts-scheduler_retained.txt", errors="replace"):
        rec = parse_policy_line(line)
        if rec:
            journal.setdefault(rec["date"], []).append(rec)

    rows, violations = [], []
    day = date.fromisoformat(WINDOW[0])
    while day.isoformat() <= WINDOW[1]:
        d = day.isoformat()
        day += timedelta(days=1)
        dec_path = picks / d / "decision.json"
        dec = json.loads(dec_path.read_text()) if dec_path.exists() else None
        pick_path = picks / f"{d}.json"
        pick = json.loads(pick_path.read_text()) if pick_path.exists() else None
        jl = journal.get(d, [])
        days_calendar = sum(1 for k in rounds if d <= k <= SEASON_END)
        cp = contest.get(rounds.get(d))
        row = {"date": d, "round": rounds.get(d), "days_calendar": days_calendar,
               "decision": None if dec is None else {k: dec.get(k) for k in (
                   "schema_version", "action", "source", "streak", "objective", "best_streak", "best_status",
                   "effective_best", "tail_policy_sha256", "degraded_reason", "delivery_status", "scoreable",
                   "state_source", "state_status", "contest_source_date")},
               "pick_file": None if pick is None else {
                   "tail_policy_sha256": pick.get("tail_policy_sha256"),
                   "policy_decision_objective": (pick.get("policy_decision") or {}).get("objective"),
                   "delivered_at": pick.get("delivered_at"), "notification_sent": pick.get("notification_sent"),
                   "bluesky_posted": pick.get("bluesky_posted")},
               "journal_policy_lines": len(jl),
               "journal_distinct": sorted({json.dumps({k: v for k, v in r.items() if k != "date"}, sort_keys=True) for r in jl}),
               # entered = the contest holds graded slots for the round. "void" is the contest's label
               # for a miss at streak 0 (nothing to lose), not a missing entry.
               "contest_entered": cp is not None and bool(cp.get("roundPredictions")),
               "contest_result": None if cp is None else cp.get("result"),
               "checks": {}}
        c = row["checks"]
        if dec is None:
            violations.append(f"{d}: no decision.json")
        else:
            s, m = dec.get("streak"), dec.get("effective_best")
            days = jl[-1]["days"] if jl else days_calendar
            c["days_journal_equals_calendar"] = (not jl) or all(r["days"] == days_calendar for r in jl)
            c["objective_matches_rule"] = dec.get("objective") == expected_objective(s, days)
            c["stop_rule_matches_action"] = (dec.get("action") == "skip") == stop_rule_skip(s, days, m)
            c["best_trusted_18"] = dec.get("best_status") == "trusted" and dec.get("best_streak") == 18 and m == 18
            c["tail_sha_is_artifact"] = dec.get("tail_policy_sha256") == artifact["tail_sha256"]
            c["not_degraded"] = dec.get("degraded_reason") is None
            c["journal_agrees_with_decision"] = bool(jl) and all(
                r["objective"] == dec.get("objective") and r["action"] == dec.get("action") and r["streak"] == s
                and r["effective_best"] == m and artifact["tail_sha256"].startswith(r["tail"]) for r in jl)
            if pick is not None:
                c["pick_file_tail_sha_is_artifact"] = pick.get("tail_policy_sha256") == artifact["tail_sha256"]
            for k, v in c.items():
                if not v:
                    violations.append(f"{d}: {k}")
        row["class"] = ("stopped (tail skip)" if dec and dec.get("action") == "skip"
                        else "delivered+entered" if dec and dec.get("delivery_status") == "delivered" and row["contest_entered"]
                        else "private (computed, not delivered/entered)" if pick is not None and not row["contest_entered"]
                        else "other")
        rows.append(row)

    report = {"read": "P-05 tail-policy audit (plan W1.4b)", "snapshot": str(snap),
              "snapshot_manifest_sha256": _sha(snap.parent / f"{snap.name}.sha256"),
              "window": WINDOW, "artifact": artifact, "violations": violations, "rows": rows}
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    print(f"artifact: tail {artifact['tail_sha256'][:12]} base {artifact['base_sha256'][:12]} binding_ok={artifact['binding_ok']}")
    print(f"{'date':<11}{'act':<7}{'s':>3}{'d':>4}{'m':>4} {'objective':<17}{'deliv':<15}{'entered':<9}{'lines':>5} class")
    for r in rows:
        dd = r["decision"] or {}
        print(f"{r['date']:<11}{str(dd.get('action')):<7}{str(dd.get('streak')):>3}{r['days_calendar']:>4}"
              f"{str(dd.get('effective_best')):>4} {str(dd.get('objective')):<17}{str(dd.get('delivery_status')):<15}"
              f"{str(r['contest_entered']):<9}{r['journal_policy_lines']:>5} {r['class']}")
    print("violations:", violations or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
