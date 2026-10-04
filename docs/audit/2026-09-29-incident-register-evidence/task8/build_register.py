"""Task 8: build the published register from the drafts and the Task 5/6 results (W1.5 Phase 1).

    uv run --with jsonschema==4.23.0 python docs/audit/2026-09-29-incident-register-evidence/task8/build_register.py

Reads the 119 drafts (``route_h/drafts/[EL]*.json``), applies the reviewed corrections in ``corrections.json``, fills the
fixture fields and writes ``docs/audit/2026-09-29-incident-register.json``. Then it runs ``records.validate`` in
publication mode with the repository as the evidence root, and the publication pin check. The fixture fields are
filled as follows:
- historical_replay: one entry per replay-manifest entry. A spec entry is ``certified`` only when its acceptance
  verdict is ``accepted`` AND the reviewer's decision in ``task56/REVIEW-RUNS.md`` is accept; otherwise it is
  unavailable or not_applicable, with the manifest's reason.
- current_defence: one entry per link of the defence manifest, under the same rule. A refused absence spec is
  ``unavailable`` with the ruling-10 reason and a pointer to its refusal record.
- expected_failure / characterization: bound to the pair accepted at the pin (file + sha256). I-077's controls are
  every E77 control that pair passed; I-203 (L03) and I-204 (L04) get their characterization entries.

The publication pin check: the pair, the replays and the defences ran at ``f453283``. The published build must have the
same ``src/``, ``tests/``, ``scripts/``, ``pyproject.toml``, ``uv.lock`` and expected-failure registry bytes; any
difference is printed and fails the build.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EV = "docs/audit/2026-09-29-incident-register-evidence"
PIN = "f453283d0a0ec39a939b22c6500b5fb36f5452c0"
PAIR = f"{EV}/expected_failures_runs/acceptance-f453283.json"
OUT = "docs/audit/2026-09-29-incident-register.json"
CLOSURE = ["src", "tests", "scripts", "pyproject.toml", "uv.lock", f"{EV}/expected_failures.json"]
T = "tests/test_incident_register_2026.py::"

sys.path.insert(0, str(REPO))
from scripts.audit.incident_register import records  # noqa: E402


def sha(path: str) -> str:
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def load(path: str):
    return json.loads((REPO / path).read_text())


def decisions() -> dict[str, str]:
    """The reviewer's last decision per run label (a label's later row supersedes an earlier one)."""
    out = {}
    for line in (REPO / EV / "task56/REVIEW-RUNS.md").read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Run") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        label = cells[0].split(" ")[0]
        out[label] = "accept" if cells[2].startswith("**accept") else cells[2].strip("*")
    return out


def replay_entries(rid: str, manifest: list, dec: dict) -> list:
    out = []
    for e in (x for x in manifest if x["incident"] == rid):
        links = ", ".join(map(str, e["links"]))
        if e["spec"] is None:
            status = "not_applicable" if e["unavailable"].startswith("no fix:") else "unavailable"
            out.append({"status": status, "reason": f"link(s) {links}: {e['unavailable']}"})
            continue
        acc_path = f"{EV}/historical_replay/results-f453283/{e['spec']}/acceptance.json"
        acc = load(acc_path)
        assert acc["spec"]["label"] == e["spec"] and acc["spec"]["links"] == e["links"], e["spec"]
        if acc["verdict"] != "accepted" or dec.get(e["spec"]) != "accept":
            out.append({"status": "unavailable", "reason": f"link(s) {links}: {e['spec']} runner verdict "
                        f"{acc['verdict']}, reviewer decision {dec.get(e['spec'])}"})
            continue
        out.append({"status": "certified", "label": acc["label_kind"], "fix_set": acc["spec"]["fix_set"],
                    "symptom_nodes": [s["node"] for s in acc["spec"]["symptom_nodes"]],
                    "acceptance": acc_path, "acceptance_sha256": sha(acc_path), "reviewer_decision": "accept",
                    "reason": f"link(s) {links}: {e['spec']}"})
    return out


def defence_entries(rid: str, manifest: list, dec: dict, notes: dict) -> list:
    out = []
    for e in (x for x in manifest if x["incident"] == rid):
        n, spec = e["link"], e["spec"]
        if spec is None:
            r = e["reason"]
            status = "not_applicable" if r.startswith(("not_applicable:", "not a fixed link:")) else "unavailable"
            out.append({"link": n, "status": status, "reason": r})
            continue
        base = f"{EV}/current_defence/results-f453283/{spec}"
        acc_path = f"{base}/acceptance.json"
        acc = load(acc_path)
        if acc["verdict"] != "accepted":
            reason = "; ".join(acc.get("reasons", []))
            assert "absence" in reason, (spec, reason)
            tests = [k["node"] for k in load(f"{EV}/current_defence/specs/{spec}.json")["killing"]]
            out.append({"link": n, "status": "unavailable",
                        "reason": "absence not certifiable by the Phase 1 recorder (plan ruling 10): the spec "
                                  f"{spec} was refused before any run; refusal record {acc_path}; its regression "
                                  f"tests: {', '.join(tests)}"})
            continue
        if dec.get(spec) != "accept":
            out.append({"link": n, "status": "unavailable", "reason": f"{spec}: reviewer decision {dec.get(spec)}"})
            continue
        assert sha(f"{base}/mutant.patch") == acc["patch_sha256"], spec
        spec_doc = acc["spec"]                              # the spec as it ran (baseline substituted)
        entry = {"link": n, "status": "certified", "level": spec_doc["level"], "patch": f"{base}/mutant.patch",
                 "patch_sha256": acc["patch_sha256"], "killing_nodes": [k["node"] for k in spec_doc["killing"]],
                 "acceptance": acc_path, "acceptance_sha256": sha(acc_path), "reviewer_decision": "accept"}
        entry["reason"] = spec + (f": {notes[spec]}" if spec in notes else "")
        out.append(entry)
    return out


def main() -> int:
    corr = load(f"{EV}/task8/corrections.json")
    rm = load(f"{EV}/historical_replay/manifest.json")
    dm = load(f"{EV}/current_defence/manifest.json")
    dec = decisions()
    pair = load(PAIR)
    assert pair["verdict"] == "accepted" and pair["resolved_ref"] == PIN
    pair_sha = sha(PAIR)
    passed = pair["passed_nodes"]

    drafts = sorted((REPO / EV / "route_h/drafts").glob("[EL]*.json"))
    recs = [json.loads(p.read_text()) for p in drafts]
    by_id = {r["id"]: r for r in recs}
    assert len(recs) == len(by_id) == 119

    for c in corr["add_evidence"]:
        ev = by_id[c["record"]]["evidence"]
        assert c["evidence"]["id"] not in {e["id"] for e in ev}, c
        ev.append(c["evidence"])
    for c in corr["fix_deployed"]:                        # reviewed corrections to install states
        f = next(f for f in by_id[c["record"]]["fix"] if f["link"] == c["link"])
        assert f["deployed"] == c["was"], (c["record"], c["link"], f["deployed"])
        f["deployed"] = c["now"]
    for c in corr["routes"]:
        r = by_id[c["record"]]
        assert r["routes"] == c["was"], (c["record"], r["routes"])
        r["routes"] = c["now"]
    for c in corr["notes"]:
        by_id[c["record"]].setdefault("notes", []).append(c["note"])

    replay_ids = {e["incident"] for e in rm}
    defence_ids = {e["incident"] for e in dm}
    for r in recs:
        fx = r["fixtures"]
        if r["id"] in replay_ids:
            assert not fx["historical_replay"], r["id"]
            fx["historical_replay"] = replay_entries(r["id"], rm, dec)
        if r["id"] in defence_ids:
            assert not fx["current_defence"], r["id"]
            fx["current_defence"] = defence_entries(r["id"], dm, dec, corr["defence_notes"])
        for kind in ("expected_failure", "characterization"):
            for x in fx[kind]:
                x["acceptance"], x["acceptance_sha256"] = PAIR, pair_sha

    e77 = by_id["I-077"]["fixtures"]["characterization"]
    assert len(e77) == 1
    e77[0]["controls"] = [n for n in passed if n.startswith(T + "test_e77_")]
    by_id["I-203"]["fixtures"]["characterization"] = [{
        "nodes": [T + "test_l03_postponed_game_is_never_delivered_by_the_cached_fallback"],
        "exception": "tests.test_incident_register_2026.PostponedPickDelivered",
        "controls": [n for n in passed if n.startswith(T + "test_l03_")], "acceptance": PAIR,
        "acceptance_sha256": pair_sha}]
    by_id["I-204"]["fixtures"]["characterization"] = [{
        "nodes": [T + "test_l04_hit_applied_once_across_a_crash_and_restart",
                  T + "test_l04_saver_miss_applied_once_across_a_crash_and_restart",
                  T + "test_l04_polling_death_then_the_cron_scorer_applies_once"],
        "exception": "tests.test_incident_register_2026.ResultAppliedTwice",
        "controls": [n for n in passed if n.startswith(T + "test_l04_")], "acceptance": PAIR,
        "acceptance_sha256": pair_sha}]

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    drift = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", PIN, "HEAD", "--", *CLOSURE],
                           capture_output=True, text=True, check=True).stdout.split()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *CLOSURE],
                           capture_output=True, text=True, check=True).stdout.split("\n")
    dirty = [d for d in dirty if d.strip()]

    out = {"schema": "w15_incident_register_v1", "pin": PIN, "built_at_head": head,
           "publication_pin_check": {"closure": CLOSURE, "differs_from_pin": drift + dirty},
           "records": recs}
    (REPO / OUT).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    errs = records.validate(copy.deepcopy(recs), publish=True, evidence_root=REPO)
    print(f"{len(recs)} records -> {OUT}; publication-mode errors: {len(errs)}")
    for e in errs:
        print("  ", e)
    print(f"publication pin check against {PIN[:7]} at {head[:7]}: "
          + ("closure identical" if not (drift or dirty) else f"DIFFERS: {drift + dirty}"))
    cert_r = sum(h["status"] == "certified" for r in recs for h in r["fixtures"]["historical_replay"])
    cert_d = sum(d["status"] == "certified" for r in recs for d in r["fixtures"]["current_defence"])
    print(f"certified replay entries {cert_r}, certified defence entries {cert_d}")
    return 1 if errs or drift or dirty else 0


if __name__ == "__main__":
    sys.exit(main())
