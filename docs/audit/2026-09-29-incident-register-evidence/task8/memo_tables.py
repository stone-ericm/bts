"""Print the memo's generated tables from the published register (W1.5 Phase 1, Task 8).

    python3 docs/audit/2026-09-29-incident-register-evidence/task8/memo_tables.py > /tmp/tables.md

The memo (docs/audit/2026-09-29-incident-register.md) embeds this output verbatim between its GENERATED markers, so every
count and status there is the register's.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
REG = json.loads((REPO / "docs/audit/2026-09-29-incident-register.json").read_text())
EXCL = json.loads((REPO / "docs/audit/2026-09-29-incident-register-evidence/route_h/drafts/exclusions.json").read_text())
R = REG["records"]
DISP = ["observed_incident", "deployed_latent_defect", "near_miss_control_held", "unresolved_candidate",
        "pre_ship_exclusion"]
TIERS = ["A", "B", "tier_pending"]


def cell(s: str, n: int = 120) -> str:
    s = " ".join(str(s).split()).replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def replay_cat(reason: str) -> str:
    r = reason.split(": ", 1)[1] if reason.startswith("link") else reason
    for key, cat in (("the fix touches no test", "fix adds no test"), ("the fix is outside src/", "outside src/"),
                     ("new API only", "new API only"), ("no fix:", "no fix (manual step)"),
                     ("no test in the fix witnesses", "no symptom test in the fix"),
                     ("asserts the fix's new", "the fix's own message only"),
                     ("new log line", "log line only"), ("diagnostic_only", "diagnostic only (a call count)")):
        if r.startswith(key) or key in r[:120]:
            return cat
    return cell(r, 50)


def defence_cat(reason: str) -> str:
    for key, cat in (("absence not certifiable", "absence (ruling 10)"), ("outside src/", "outside src/"),
                     ("no current defence", "no current test"), ("superseded", "fix superseded"),
                     ("not_applicable:", "no fix (manual step)"), ("not a fixed link", "no fix (manual step)"),
                     ("disabled on the production path", "guarded mode off in production"),
                     ("live MLB", "test reaches the live MLB API"),
                     ("no recordable positive witness", "no recordable witness"),
                     ("withdrawn after review: elapsed-time", "withdrawn: elapsed-time decision (ruling 13)")):
        if key in reason[:200]:
            return cat
    return cell(reason, 50)


def fixed_links(r):
    return [f["link"] for f in r["fix"] if isinstance(f["implemented"], dict)]


def links_of(h):
    return h["reason"].split(":", 1)[0].replace("link(s) ", "")


def main():
    out = []
    p = out.append
    c = Counter((r["disposition"], r["tier"]) for r in R)
    p("<!-- GENERATED:counts -->")
    p("| Disposition | A | B | tier_pending | Total |")
    p("|---|---|---|---|---|")
    for d in DISP:
        row = [c[(d, t)] for t in TIERS]
        p(f"| {d} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    p(f"| **total** | " + " | ".join(str(sum(c[(d, t)] for d in DISP)) for t in TIERS) + f" | **{len(R)}** |")
    p("")
    cls = Counter(r["classes"]["primary"] for r in R)
    p("Primary effect class: " + ", ".join(f"{k} {v}" for k, v in sorted(cls.items(), key=lambda x: -x[1])) + ".")
    p("<!-- /GENERATED:counts -->")
    p("")

    p("<!-- GENERATED:fixtures -->")
    p("| Record | Link | Historical replay (§9.2) | Current defence (§9.3) |")
    p("|---|---|---|---|")
    tot = Counter()
    for r in R:
        fx = r["fixtures"]
        if not fx["historical_replay"] and not fx["current_defence"]:
            continue
        reps = fx["historical_replay"]
        for d in sorted(fx["current_defence"], key=lambda x: x["link"]):
            n = d["link"]
            rep = next((h for h in reps if str(n) in links_of(h).split(", ")), None)
            if rep is None:
                rtxt = "—"
            elif rep["status"] == "certified":
                rtxt = f"**certified** ({rep['reason'].split(': ')[1]})"
                tot["replay certified"] += 1
            else:
                rtxt = f"{rep['status']}: {replay_cat(rep['reason'])}"
                tot[f"replay {rep['status']}"] += 1
            if d["status"] == "certified":
                dtxt = f"**certified, {d['level']}** ({d['reason'].split(':')[0]})"
                tot["defence certified"] += 1
                tot[f"defence {d['level']}"] += 1
            else:
                dtxt = f"{d['status']}: {defence_cat(d['reason'])}"
                tot[f"defence {d['status']}"] += 1
                tot[f"defence cat {defence_cat(d['reason'])}"] += 1
            if rep is not None and rep["status"] != "certified":
                tot[f"replay cat {replay_cat(rep['reason'])}"] += 1
            p(f"| {r['id']} | {n} | {rtxt} | {dtxt} |")
            rc = rep is not None and rep["status"] == "certified"
            dc = d["status"] == "certified"
            tot["pair " + ("both" if rc and dc else "replay only" if rc else "defence only" if dc else "neither")] += 1
    p("<!-- /GENERATED:fixtures -->")
    p("")
    p("<!-- GENERATED:fixture-totals -->")
    p("| Over the 62 link entries | certified | unavailable | not_applicable |")
    p("|---|---|---|---|")
    p(f"| Historical replay | {tot['replay certified']} | {tot['replay unavailable']} | {tot['replay not_applicable']} |")
    p(f"| Current defence | {tot['defence certified']} ({tot['defence production_path']} production_path, "
      f"{tot['defence component']} component) | {tot['defence unavailable']} | {tot['defence not_applicable']} |")
    p(f"| Links: both certified / replay only / defence only / neither | {tot['pair both']} / {tot['pair replay only']} / "
      f"{tot['pair defence only']} / {tot['pair neither']} | | |")
    p("")
    for kind in ("replay", "defence"):
        cats = sorted(((k.split(" cat ", 1)[1], v) for k, v in tot.items() if k.startswith(f"{kind} cat ")),
                      key=lambda x: -x[1])
        p(f"Why a {kind} is not certified: " + "; ".join(f"{k} {v}" for k, v in cats) + ".")
        p("")
    p("<!-- /GENERATED:fixture-totals -->")
    p("")

    p("<!-- GENERATED:expected -->")
    p("| Record | Kind | Nodes | Exception | Controls |")
    p("|---|---|---|---|---|")
    for r in R:
        for kind in ("expected_failure", "characterization"):
            for x in r["fixtures"][kind]:
                p(f"| {r['id']} | {kind} | {len(x['nodes'])} | `{x['exception'].rsplit('.', 1)[1]}` | "
                  f"{len(x.get('controls', []))} |")
        for x in r["fixtures"].get("config") or []:
            p(f"| {r['id']} | config (§9.9) | {len(x['nodes'])} | — | pins: {cell(x['pins'], 90)} |")
        if r["fixtures"].get("deferred"):
            p(f"| {r['id']} | deferred | — | — | {cell(r['fixtures']['deferred'], 90)} |")
    p("<!-- /GENERATED:expected -->")
    p("")

    for d in DISP:
        p(f"<!-- GENERATED:records:{d} -->")
        p("| Record | Tier | Class | Stream | Title | Fix |")
        p("|---|---|---|---|---|---|")
        for r in (x for x in R if x["disposition"] == d):
            fl = fixed_links(r)
            nl = len(r["mechanism"])
            states = Counter(f["implemented"] if isinstance(f["implemented"], str) else "fixed" for f in r["fix"])
            fix = ", ".join(f"{k} {v}" for k, v in states.items()) or "—"
            stream = r["stream"] if isinstance(r["stream"], str) else "/".join(r["stream"])
            p(f"| {r['id']}{' (plan)' if r['plan_named'] else ''} | {r['tier']} | {r['classes']['primary']} | "
              f"{stream} | {cell(r['title'], 140)} | {fix} ({nl} links) |")
        p(f"<!-- /GENERATED:records:{d} -->")
        p("")

    p("<!-- GENERATED:pending -->")
    p("| Record | Disposition | Why the tier is pending |")
    p("|---|---|---|")
    for r in (x for x in R if x["tier"] == "tier_pending"):
        p(f"| {r['id']} | {r['disposition']} | {cell(r.get('tier_reason') or '', 160)} |")
    p("<!-- /GENERATED:pending -->")
    p("")

    p("<!-- GENERATED:unresolved -->")
    p("| Record | Tier | Missing evidence |")
    p("|---|---|---|")
    for r in (x for x in R if x["disposition"] == "unresolved_candidate"):
        me = r.get("missing_evidence")
        me = "; ".join(me) if isinstance(me, list) else me
        p(f"| {r['id']} | {r['tier']} | {cell(me or '', 170)} |")
    p("<!-- /GENERATED:unresolved -->")
    p("")

    p("<!-- GENERATED:exclusions -->")
    p("| Episode | Former id | Reason |")
    p("|---|---|---|")
    for e in EXCL:
        p(f"| {e['working_id']} | {e.get('former_id') or '—'} | {cell(e['reason'], 170)} |")
    p("<!-- /GENERATED:exclusions -->")
    p("")

    p("<!-- GENERATED:routes -->")
    p("| Disposition | H and X | H only | X only | neither | R |")
    p("|---|---|---|---|---|---|")
    for d in DISP:
        rc = Counter((bool(r["routes"].get("H")), bool(r["routes"].get("X"))) for r in R if r["disposition"] == d)
        rs = {r["routes"].get("R") for r in R if r["disposition"] == d}
        p(f"| {d} | {rc[(True, True)]} | {rc[(True, False)]} | {rc[(False, True)]} | {rc[(False, False)]} | "
          + ", ".join(sorted(map(str, rs))) + " |")
    p("<!-- /GENERATED:routes -->")
    p("")

    p("<!-- GENERATED:watchdog -->")
    w = Counter((r["watchdog"] or {}).get("boundary", "none") if isinstance(r["watchdog"], dict) else "none" for r in R)
    p("| Watchdog boundary | Records |")
    p("|---|---|")
    for k, v in w.most_common():
        p(f"| {k} | {v} |")
    p("<!-- /GENERATED:watchdog -->")
    return "\n".join(out)


def splice(memo: Path, generated: str) -> None:
    """Replace every <!-- GENERATED:x --> ... <!-- /GENERATED:x --> block in the memo with the generated one."""
    import re
    blocks = dict(re.findall(r"(<!-- GENERATED:([^ ]+) -->.*?<!-- /GENERATED:\2 -->)", generated, re.S)[i][::-1]
                  for i in range(len(re.findall(r"<!-- GENERATED:[^ ]+ -->", generated))))
    text = memo.read_text()
    for name, block in blocks.items():
        pat = re.compile(rf"<!-- GENERATED:{re.escape(name)} -->.*?<!-- /GENERATED:{re.escape(name)} -->", re.S)
        if pat.search(text):
            text = pat.sub(lambda _m: block, text)
    memo.write_text(text)


if __name__ == "__main__":
    import sys
    g = main()
    if sys.argv[1:2] == ["--write"]:
        splice(REPO / "docs/audit/2026-09-29-incident-register.md", g)
    else:
        print(g)
