"""The shared C1 admission gate for outcome-bearing candidate runs (rank 3 first). It applies what the deferred 4b
review found (`docs/audit/2026-10-05-c1-r4b-code-codex-r3.md`, B1-B3).

**The admission record** (`admission.json`, the only file of the executable closure that may change after review):
- `reviewed_commit`: the full commit Codex reviewed.
- `review_report`: the archived review report (repo path). At the exposure commit it must exist, record a SIGN (or
  SIGN WITH EDITS) under `## Verdict`, and name the reviewed commit in full. A self-declared commit is not a review
  anchor (B2).
- `exposure_commit`: the commit that publishes the candidate's exposure row. The row must first appear in that commit
  and be unchanged at HEAD.

**The checks:**
- the reviewed commit precedes the exposure commit, which precedes HEAD;
- no closure file differs from the reviewed commit except `admission.json`, and nothing in the closure is untracked
  or modified;
- every loaded `bts` / `scripts` module comes from this checkout.

**One claimed run:**
- **Claim:** a durable `CLAIM.json` precedes the first outcome-bearing read. Any claimed run blocks another.
- **Release:** `INVALIDATION_<run>.json` must carry the claim's sha256, and register row `C1-invalidate-<run>` must
  record exactly "**RULED <date>: INVALIDATE `<run>` claim `<sha prefix>`; correction `<full commit>`**" with Eric as
  its source. The correction commit must be in HEAD's history (B1).

**Inputs** are read once: `read_pinned` hashes and returns the same bytes, so the parsed buffer is the pinned one
(B3).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
VERDICT_SIGN = re.compile(r"^\*\*SIGN(?: WITH EDITS)?\b")
INVALIDATE = re.compile(r"^\*\*RULED \d{4}-\d{2}-\d{2}: INVALIDATE `(?P<run>[^`]+)` claim `(?P<claim>[0-9a-f]{12,64})`; "
                        r"correction `(?P<corr>[0-9a-f]{40})`\*\*$")


class ProvenanceError(RuntimeError):
    pass


def _git(repo: Path, *args, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def _ancestor(repo: Path, a: str, b: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def row_cells(text: str, row_id: str) -> list[str] | None:
    line = next((l for l in text.splitlines() if l.startswith(f"| {row_id} |")), None)
    return [c.strip() for c in line.strip().strip("|").split("|")] if line else None


def _review_signs(report: str, reviewed: str) -> list[str]:
    lines = report.splitlines()
    try:
        i = lines.index("## Verdict")
    except ValueError:
        return ["the review report has no '## Verdict' section"]
    verdict = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
    out = [] if VERDICT_SIGN.match(verdict) else [f"the review report's verdict is not a SIGN: {verdict[:60]!r}"]
    if reviewed not in report:
        out.append("the review report does not name the reviewed commit in full")
    return out


def admission_check(repo: Path, adm: dict, *, closure, admission_rel: str, register_rel: str,
                    exposure_row: str) -> tuple[str, list[str]]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    rc, xc, rep = adm.get("reviewed_commit"), adm.get("exposure_commit"), adm.get("review_report")
    reasons = [f"admission: {k} is not a full commit id" for k, v in (("reviewed_commit", rc), ("exposure_commit", xc))
               if not (isinstance(v, str) and HEX40.match(v))]
    if not (isinstance(rep, str) and rep.strip()):
        reasons.append("admission: review_report is unset")
    if reasons:
        return head, reasons
    shown = _git(repo, "show", f"{xc}:{rep}", check=False)
    if shown.returncode != 0:
        reasons.append(f"admission: the review report {rep} does not exist at the exposure commit")
    else:
        reasons += [f"admission: {x}" for x in _review_signs(shown.stdout, rc)]
    if not _ancestor(repo, rc, xc):
        reasons.append("admission: the reviewed commit is not an ancestor of the exposure commit")
    if not _ancestor(repo, xc, head):
        reasons.append(f"admission: the exposure commit is not an ancestor of HEAD {head[:7]}")
    row_at = lambda c: next((l for l in _git(repo, "show", f"{c}:{register_rel}", check=False).stdout.splitlines()  # noqa: E731
                             if l.startswith(f"| {exposure_row} |")), None)
    at_x, at_parent = row_at(xc), row_at(f"{xc}^")
    at_head = next((l for l in (repo / register_rel).read_text().splitlines() if l.startswith(f"| {exposure_row} |")), None)
    if at_x is None:
        reasons.append(f"admission: the exposure commit's register has no {exposure_row} row")
    elif at_parent is not None:
        reasons.append(f"admission: {exposure_row} was not published in the named commit")
    elif at_head != at_x:
        reasons.append(f"admission: the {exposure_row} row changed after its publication")
    changed = [f for f in _git(repo, "diff", "--name-only", rc, head, "--", *closure).stdout.split() if f != admission_rel]
    if changed:
        reasons.append(f"admission: executable files changed since the reviewed commit: {changed[:5]}")
    loose = [l for l in _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *closure).stdout.splitlines()
             if l.strip()]
    if loose:
        reasons.append(f"admission: modified or untracked files in the executable closure: {loose[:5]}")
    return head, reasons


def foreign_imports(repo: Path = REPO) -> list[str]:
    bad = []
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] in ("bts", "scripts"):
            f = getattr(mod, "__file__", None)
            if f is None or not Path(f).resolve().is_relative_to(repo):
                bad.append(f"{name}: {f}")
    return sorted(bad)


def durable_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_claim(run_dir: Path, head: str) -> str:
    rec = {"run": run_dir.name, "code": head, "pid": os.getpid(), "claimed_utc": datetime.now(timezone.utc).isoformat()}
    data = (json.dumps(rec) + "\n").encode()
    durable_write(run_dir / "CLAIM.json", data)
    return hashlib.sha256(data).hexdigest()


def invalidation_problems(rec, run: str, claim_sha: str, register_text: str, repo: Path = REPO) -> list[str]:
    if not isinstance(rec, dict) or rec.get("run") != run:
        return ["the invalidation does not name this run"]
    if rec.get("claim_sha256") != claim_sha:
        return ["the invalidation is not bound to this claim"]
    cells = row_cells(register_text, f"C1-invalidate-{run}")
    m = INVALIDATE.match(cells[2]) if cells and len(cells) >= 4 else None
    if not (m and m.group("run") == run and claim_sha.startswith(m.group("claim")) and cells[-1].startswith("Eric")):
        return [f"no register row C1-invalidate-{run} recording Eric's INVALIDATE of this claim"]
    if not _ancestor(repo, m.group("corr"), "HEAD"):
        return ["the ruling's correction commit is not in HEAD's history"]
    return []


def claimed_runs(root: Path, register_text: str) -> list[str]:
    blocked = []
    if root.exists():
        for d in sorted(x for x in root.iterdir() if x.is_dir() and (x / "CLAIM.json").exists()):
            claim_sha = hashlib.sha256((d / "CLAIM.json").read_bytes()).hexdigest()
            inv = root / f"INVALIDATION_{d.name}.json"
            try:
                problems = invalidation_problems(json.loads(inv.read_text()), d.name, claim_sha, register_text) \
                    if inv.exists() else ["no invalidation"]
            except ValueError:
                problems = ["unreadable invalidation"]
            if problems:
                blocked.append(f"{d.name} ({'; '.join(problems)})")
    return blocked


@contextlib.contextmanager
def admission_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".admission.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("refusing: another run holds the admission lock") from None
        yield


def read_pinned(path: Path, sha256: str) -> tuple[bytes, str]:
    """Read once; the returned bytes are the hashed bytes."""
    b = Path(path).read_bytes()
    h = hashlib.sha256(b).hexdigest()
    if h != sha256:
        raise ProvenanceError(f"{path}: sha256 {h[:12]} is not the pinned {sha256[:12]}")
    return b, h
