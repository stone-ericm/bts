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
from pathlib import Path

_LINE = re.compile(r"^[^\t]*\t[^\t]*\t(?P<ts>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.\d+)?Z "
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
            "rolled_back_at": rolled_at, "clean_completion": "clean_completion" in hits,
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
                        "clean_completion": False, "test_gate": None, "template_hits": {}, "anomalies": []})
        out.append(rec)
    return out


def _same(a: str | None, b: str | None) -> bool:
    return bool(a and b) and (a.startswith(b) or b.startswith(a))


def installed_timeline(runs: list[dict]) -> list[dict]:
    """Segments ``{"from", "sha", "basis", "candidate"}`` in time order (each runs until the next)."""
    segs: list[dict] = []

    def open_(t, sha, basis, candidate=None):
        segs.append({"from": t, "sha": sha, "basis": basis, "candidate": candidate})

    for r in sorted(runs, key=lambda r: r["created_at"]):
        if r["log"] != "retained":
            open_(r["created_at"], None, "unknown")
            continue
        if r.get("pre_sha"):
            prev = segs[-1] if segs else None
            if prev is not None and prev["basis"] == "log" and not _same(prev["sha"], r["pre_sha"]):
                prev.update(sha=None, basis="drift")
            elif prev is not None and prev["basis"] == "unknown" and prev["candidate"] is None:
                prev["candidate"] = r["pre_sha"]
            open_(r["pre_at"], r["pre_sha"], "log")
        if r.get("deployed_sha"):
            open_(r["deployed_at"], r["deployed_sha"], "log")
            if r.get("canary") == "failed" and r.get("rollback") == "clean":
                open_(r["rolled_back_at"], r["pre_sha"], "log")
            elif r.get("canary") == "failed":
                open_(r["canary_at"] or r["deployed_at"], None, "unknown")
    return segs


def live_at(timeline: list[dict], t: str) -> dict:
    before = [s for s in timeline if s["from"] <= t]
    if not before:
        first = next((s["sha"] for s in timeline if s["basis"] == "log"), None)
        return {"sha": None, "basis": "unknown", "candidate": first}
    seg = before[-1]
    if seg["basis"] == "log":
        return {"sha": seg["sha"], "basis": "log", "candidate": None}
    return {"sha": None, "basis": seg["basis"], "candidate": seg["candidate"]}


def _is_ancestor(fix: str, sha: str, repo) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", fix, sha], cwd=repo,
                          capture_output=True).returncode == 0


def first_live(timeline: list[dict], fix: str, *, repo) -> dict | None:
    """Earliest logged install containing ``fix`` and the latest instant it was logged absent.

    ``live_by`` is the first log-observed instant with the fix installed; ``not_live_before`` is
    the end of the last log segment whose install lacks it. The true first-live time lies in
    ``(not_live_before, live_by]``; equal values mean the deploy is pinned exactly.
    """
    repo = Path(repo)
    for i, seg in enumerate(timeline):
        if seg["basis"] == "log" and _is_ancestor(fix, seg["sha"], repo):
            not_live = None
            for j in range(i - 1, -1, -1):
                prev = timeline[j]
                if prev["basis"] == "log" and not _is_ancestor(fix, prev["sha"], repo):
                    not_live = timeline[j + 1]["from"]
                    break
            return {"sha": seg["sha"], "live_by": seg["from"], "not_live_before": not_live, "basis": "log"}
    return None
