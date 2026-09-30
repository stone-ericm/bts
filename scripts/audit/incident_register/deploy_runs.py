"""Typed deploy-run transitions from retained GitHub Actions logs (design §4 R4; plan rev 2 Task 7).

Only lines that match a fixed output template of ``.github/workflows/deploy.yml`` are read; no other
log text is stored, returned or displayed. ``gh run view <id> --log`` serves each line as
``<job>\\t<step>\\t<ISO-8601 Z timestamp> <text>``. The ssh action echoes the deploy script before
running it, so a template counts only when its text starts right after the timestamp (optionally
after an ``out: ``/``err: `` prefix) and matches to the end of the line — the echoed
``echo "Deployed $NEW_SHA"`` source never matches.

An expired log (HTTP 410) is ``unavailable_expired``; a template that is absent is ``absent``,
never success. ``installed_timeline`` turns the runs into segments of what the box had checked out:
``log`` (observed in a retained log), ``unknown`` (after a run whose log expired; the next retained
run's pre-deploy SHA is kept only as a ``candidate``) or ``drift`` (a pre-deploy SHA that disagrees
with the previous logged install, i.e. the box changed outside the workflow).
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

# the timestamp is kept at the precision the log prints it (Codex phase-1 r5 #7: fractions were dropped,
# and two lines of one second collapsed into an empty first-live interval)
_LINE = re.compile(r"^[^\t]*\t[^\t]*\t(?P<ts>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z "
                   r"(?:(?:out|err): )?(?P<text>.*)$")
_SHA = r"(?P<sha>[0-9a-f]{7,40})"
TEMPLATES = {
    "pre_sha": re.compile(rf"Pre-deploy SHA: {_SHA}$"),
    "deployed": re.compile(rf"Deployed {_SHA}$"),
    "canary_passed": re.compile(rf"CANARY PASSED — {_SHA} is live and healthy$"),
    "canary_failed": re.compile(rf"CANARY FAILED — rolling back to {_SHA}$"),
    "rolled_back": re.compile(rf"Rolled back cleanly to {_SHA}$"),
    "rollback_unhealthy": re.compile(r"CRITICAL: rollback also unhealthy — manual intervention required$"),
    "clean_completion": re.compile(
        r"bts-scheduler\.service completed cleanly and is waiting for Restart=always relaunch$"),
    "pytest_summary": re.compile(r"=+ (?P<counts>\d+ [a-z]+(?:, \d+ [a-z]+)*) in [0-9.]+s(?: \([0-9:]+\))? =+$"),
}
_COUNT_KEYS = {"passed", "failed", "error", "errors", "skipped", "deselected", "xfailed", "xpassed",
               "warning", "warnings", "rerun"}


def _hits(log_text: str) -> dict[str, list[tuple[str, dict]]]:
    hits: dict[str, list[tuple[str, dict]]] = {}
    for raw in log_text.splitlines():
        m = _LINE.match(raw)
        if not m:
            continue
        text = m.group("text")
        for name, pat in TEMPLATES.items():
            t = pat.match(text)
            if t:
                hits.setdefault(name, []).append((m.group("ts") + "Z", t.groupdict()))
                break
    return hits


def _single(hits, name, anomalies):
    got = hits.get(name, [])
    if len(got) > 1:
        anomalies.append(f"multiple_{name}")
        return None, None
    if not got:
        return None, None
    ts, groups = got[0]
    return groups.get("sha"), ts


def extract(log_text: str) -> dict:
    """Typed fields only: short SHAs, ISO timestamps, integer counts and closed categories."""
    hits = _hits(log_text)
    anomalies: list[str] = []
    pre, pre_at = _single(hits, "pre_sha", anomalies)
    deployed, deployed_at = _single(hits, "deployed", anomalies)
    passed_sha, canary_at = _single(hits, "canary_passed", anomalies)
    failed_sha, failed_at = _single(hits, "canary_failed", anomalies)
    rolled_sha, rolled_at = _single(hits, "rolled_back", anomalies)
    if passed_sha and failed_sha:
        canary = "conflict"
        anomalies.append("canary_conflict")
    elif "canary_passed" in hits:
        canary = "passed"
    elif "canary_failed" in hits:
        canary = "failed"
    else:
        canary = "absent"
    if passed_sha and deployed and not (passed_sha.startswith(deployed) or deployed.startswith(passed_sha)):
        anomalies.append("canary_sha_mismatch")
    if failed_sha and pre and not (failed_sha.startswith(pre) or pre.startswith(failed_sha)):
        anomalies.append("rollback_target_mismatch")
    # the rollback's own log line names what was reinstalled; it must agree with the intended target
    # (Codex phase-1 r4 #9: a 'rolled back cleanly to C' after 'rolling back to A' was read as A)
    target = failed_sha or pre
    if rolled_sha and target and not (rolled_sha.startswith(target) or target.startswith(rolled_sha)):
        anomalies.append("rolled_back_sha_mismatch")
    if "rollback_unhealthy" in hits:
        rollback = "unhealthy"
    elif "rolled_back" in hits:
        rollback = "clean"
    else:
        rollback = "none"
    gate = None
    summaries = hits.get("pytest_summary", [])
    if len(summaries) > 1:
        anomalies.append("multiple_pytest_summary")
    elif summaries:
        gate = {}
        for part in summaries[0][1]["counts"].split(", "):
            n, key = part.split(" ", 1)
            if key in _COUNT_KEYS:
                gate[key] = int(n)
            else:
                anomalies.append("unrecognised_pytest_count")
    return {"pre_sha": pre, "pre_at": pre_at, "deployed_sha": deployed, "deployed_at": deployed_at,
            "canary": canary, "canary_at": canary_at or failed_at, "rollback": rollback,
            "rolled_back_at": rolled_at, "rolled_back_sha": rolled_sha, "clean_completion": "clean_completion" in hits,
            "test_gate": gate, "template_hits": {k: len(v) for k, v in hits.items()},
            "anomalies": anomalies}


def fetch_log(run_id: int, *, runner=subprocess.run) -> tuple[str, str | None]:
    """``(status, text)``; stderr is classified, never returned."""
    proc = runner(["gh", "run", "view", str(run_id), "--log"], capture_output=True, text=True)
    if proc.returncode == 0:
        return "retained", proc.stdout
    if "HTTP 410" in (proc.stderr or ""):
        return "unavailable_expired", None
    return "unavailable_error", None


def build(runs: list[dict], *, fetch=fetch_log) -> list[dict]:
    """One typed record per run (metadata from ``gh run list --json`` + the log extraction)."""
    out = []
    for r in sorted(runs, key=lambda r: r["createdAt"]):
        status, text = fetch(r["databaseId"])
        rec = {"run_id": r["databaseId"], "created_at": r["createdAt"], "head_sha": r["headSha"],
               "head_branch": r["headBranch"], "event": r["event"], "conclusion": r["conclusion"],
               "log": status}
        if text is not None:
            rec.update(extract(text))
        else:
            rec.update({"pre_sha": None, "pre_at": None, "deployed_sha": None, "deployed_at": None,
                        "canary": "absent", "canary_at": None, "rollback": "none", "rolled_back_at": None,
                        "rolled_back_sha": None, "clean_completion": False, "test_gate": None, "template_hits": {}, "anomalies": []})
        out.append(rec)
    return out


def _instant(ts: str) -> tuple[datetime, datetime]:
    """``[floor, ceil)`` of a printed UTC timestamp: the printed value truncates the true instant, which
    lies at or after it and before it plus one unit of the printed precision (1 s with no fraction;
    below a microsecond, 1 µs)."""
    body = ts[:-1] if ts.endswith("Z") else ts
    frac = body.partition(".")[2]
    floor = datetime.fromisoformat(body).replace(tzinfo=timezone.utc)
    step = timedelta(seconds=1) if not frac else timedelta(microseconds=max(1, 10 ** (6 - len(frac))) if len(frac) <= 6 else 1)
    return floor, floor + step


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _same(a: str | None, b: str | None) -> bool:
    return bool(a and b) and (a.startswith(b) or b.startswith(a))


def observations(runs: list[dict]) -> list[dict]:
    """Every installed SHA a retained log observed, as time points (Codex phase-1 r3 #5):
    ``pre_deploy`` (before checkout), ``deployed`` (after checkout + service restart) and
    ``rolled_back``. A canary pass is a separate health observation, not an installation time."""
    obs = []
    for order, r in enumerate(sorted(runs, key=lambda r: r["created_at"])):
        if r["log"] != "retained":
            continue
        if r.get("pre_sha"):
            obs.append({"at": r["pre_at"], "sha": r["pre_sha"], "kind": "pre_deploy", "run_id": r["run_id"], "_k": (order, 0)})
        if r.get("deployed_sha"):
            obs.append({"at": r["deployed_at"], "sha": r["deployed_sha"], "kind": "deployed", "run_id": r["run_id"], "_k": (order, 1)})
        # only the SHA the rollback line itself names is an observation, never the intended target
        if r.get("rollback") == "clean" and r.get("rolled_back_at") and r.get("rolled_back_sha"):
            obs.append({"at": r["rolled_back_at"], "sha": r["rolled_back_sha"], "kind": "rolled_back",
                        "run_id": r["run_id"], "_k": (order, 2)})
    # ordered by the INSTANT each point was observed, never by its text (Codex phase-1 r4 #9: an
    # older-created run can execute later; r5 #7: '...00Z' sorts after '...00.5Z' as text); within one
    # run the log order breaks a tie
    obs.sort(key=lambda p: (_instant(p["at"])[0], p.pop("_k")))
    for a, b in zip(obs, obs[1:]):
        # two runs' points whose precision intervals overlap cannot be ordered; if they disagree, refuse
        # (a symmetric overlap test: it must not depend on the order it is asked to check)
        (fa, ca), (fb, cb) = _instant(a["at"]), _instant(b["at"])
        if a["run_id"] != b["run_id"] and not _same(a["sha"], b["sha"]) and fa < cb and fb < ca:
            raise ValueError(f"observations of different runs cannot be ordered at their precision and disagree: "
                             f"{a['at']} {a['sha']} vs {b['at']} {b['sha']}")
    return obs


def installed_timeline(runs: list[dict]) -> list[dict]:
    """Segments ``{"from", "to", "sha", "basis", "candidate"}`` between consecutive observation points.

    ``transition`` — inside one run (pre-deploy -> deployed, deployed -> rolled back): the box moved
    between the two SHAs at an unobserved instant; ``assumed_continuous`` — between runs whose endpoints
    agree (endpoint agreement is evidence; continuity in between is an explicit assumption, never proof
    that no out-of-band change occurred); ``drift`` — endpoints disagree with no retained run between;
    ``unknown`` — from the start of a run whose log expired (or a failed canary without a clean
    rollback) until the next observation, whose SHA is kept only as a ``candidate``."""
    points = observations(runs)
    breaks = [r["created_at"] for r in runs if r["log"] != "retained"]
    breaks += [r["canary_at"] or r["deployed_at"] for r in runs if r["log"] == "retained"
               and r.get("canary") == "failed" and r.get("rollback") != "clean" and r.get("deployed_at")]
    breaks = sorted(breaks, key=lambda t: _instant(t)[0])
    segs: list[dict] = []
    for a, b in zip(points, points[1:] + [None]):
        to = b["at"] if b else None
        brk = [t for t in breaks if _instant(a["at"])[0] < _instant(t)[0]
               and (to is None or _instant(t)[0] < _instant(to)[0])]
        if b is not None and a["run_id"] == b["run_id"]:
            segs.append({"from": a["at"], "to": to, "sha": None, "basis": "transition", "candidate": None,
                         "between": [a["sha"], b["sha"]]})
            continue
        end = brk[0] if brk else to
        if b is None and not brk:
            segs.append({"from": a["at"], "to": None, "sha": a["sha"], "basis": "assumed_continuous", "candidate": None})
            continue
        if b is not None and not brk and not _same(a["sha"], b["sha"]):
            segs.append({"from": a["at"], "to": to, "sha": None, "basis": "drift", "candidate": None})
            continue
        segs.append({"from": a["at"], "to": end, "sha": a["sha"], "basis": "assumed_continuous", "candidate": None})
        if brk:
            segs.append({"from": brk[0], "to": to, "sha": None, "basis": "unknown",
                         "candidate": b["sha"] if b else None})
    return segs


def live_at(timeline: list[dict], t: str) -> dict:
    """What the box ran at ``t``: ``observed`` exactly at an observation point, else the segment's basis."""
    for seg in timeline:
        if seg["from"] == t and seg["basis"] in ("assumed_continuous", "transition"):
            sha = seg["sha"] if seg["basis"] == "assumed_continuous" else seg["between"][0]
            return {"sha": sha, "basis": "observed", "candidate": None}
    inside = [s for s in timeline if _instant(s["from"])[0] <= _instant(t)[0]
              and (s["to"] is None or _instant(t)[0] < _instant(s["to"])[0])]
    if not inside:
        first = next((s["sha"] or (s.get("between") or [None])[0] for s in timeline), None)
        return {"sha": None, "basis": "unknown", "candidate": first}
    seg = inside[-1]
    if seg["basis"] == "assumed_continuous":
        return {"sha": seg["sha"], "basis": "assumed_continuous", "candidate": None}
    return {"sha": None, "basis": seg["basis"], "candidate": seg.get("candidate")}


def _is_ancestor(fix: str, sha: str, repo) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", fix, sha], cwd=repo,
                          capture_output=True).returncode == 0


def first_live(runs_or_points, fix: str, *, repo) -> dict | None:
    """Earliest log OBSERVATION of an install containing ``fix`` and the latest observation before it
    of an install without it: the first-live time lies in ``(not_live_before, live_by]``. Both ends are
    observation points (e.g. a run's pre-deploy line and its deployed line), never a segment boundary,
    so a deploy is bounded by its own log lines, not collapsed to one instant (Codex phase-1 r3 #5).
    Each end keeps its line's precision (Codex phase-1 r5 #7): ``not_live_before`` is the earlier line's
    printed instant (the true one is at or after it), ``live_by`` the END of the containing line's
    printed unit (the true one is before it), so the interval is never empty.
    ``live_by`` is the earliest OBSERVED containing install, not proof of the first-ever install when
    earlier logs expired (``not_live_before`` is then older or None)."""
    points = runs_or_points if runs_or_points and "kind" in runs_or_points[0] else observations(runs_or_points)
    repo = Path(repo)
    for i, p in enumerate(points):
        if _is_ancestor(fix, p["sha"], repo):
            prior = [q for q in points[:i] if not _is_ancestor(fix, q["sha"], repo)]
            return {"sha": p["sha"], "live_by": _iso(_instant(p["at"])[1]), "live_by_kind": p["kind"],
                    "run_id": p["run_id"], "not_live_before": _iso(_instant(prior[-1]["at"])[0]) if prior else None,
                    "basis": "log"}
    return None
