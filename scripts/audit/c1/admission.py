"""The shared C1 admission gate for outcome-bearing candidate runs (rank 3 first). It applies what the deferred 4b
review found (`docs/audit/2026-10-05-c1-r4b-code-codex-r3.md`, B1-B3).

**The admission record** (`admission.json`, the only file of the executable closure that may change after review):
- `reviewed_commit`: the full commit Codex reviewed.
- `review_report`: the archived review report (repo path). At the exposure commit it must exist; its verdict line
  must begin exactly "**SIGN.**"; and its verdict section must name the reviewed commit in full (rank-3 review r2
  N1: a hash mentioned elsewhere does not count).
  - A "**SIGN WITH EDITS.**" is conditional: it is refused until a plain SIGN of the edited commit is recorded. An
    ancestry check cannot show that the edits were applied verbatim.
- `exposure_commit`: the commit that publishes the exposure row. The row must first appear in that commit and be
  unchanged at HEAD. Its description cell must be a positive structured record (r2 N1):
  "**PREDECLARED <date>: <scope>**; review `<report path>` sha256 `<16+ hex>`; reviewed `<commit>`[; inputs
  `<16+ hex>`]". Each field must equal this admission. A DENIED row, a withheld row or a free-text mention does not
  match.
- `accepted_identity` returns the accepted report's exact bytes digest at the exposure commit, so the run's manifest
  records the report the gate accepted rather than a later working-tree copy (r2 N6).

**The checks:**
- the reviewed commit precedes the exposure commit, which precedes HEAD;
- no closure file differs from the reviewed commit except `admission.json`, and nothing in the closure is untracked
  or modified;
- every loaded `bts` / `scripts` module comes from this checkout.

**One claimed run:**
- **Claim:** a durable `CLAIM.json` precedes the first outcome-bearing read, inside a run directory whose parent entry
  is fsynced (`make_run_dir`). Any claimed run blocks another.
- **Release:** `INVALIDATION_<run>.json` must carry the claim's sha256, and register row `C1-invalidate-<run>` must
  record exactly "**RULED <date>: INVALIDATE `<run>` claim `<sha prefix>`; correction `<full commit>` reviewed
  `<report path>` `<sha prefix>`**", with the source token Eric (B1; r3b F3).
  - The correction commit must be in HEAD's history.
  - The cited report (at HEAD) must match the sha prefix, carry an exact SIGN verdict, and name the correction
    commit.

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
VERDICT_SIGN = re.compile(r"^\*\*(?P<v>SIGN|SIGN WITH EDITS)\.\*\*(?:\s|$)")
EXPOSURE = re.compile(r"^\*\*PREDECLARED (?P<date>\d{4}-\d{2}-\d{2}): (?P<scope>[^*]+)\*\*; review `(?P<rep>[^`]+)` "
                      r"sha256 `(?P<sha>[0-9a-f]{16,64})`; reviewed `(?P<ref>[0-9a-f]{40})`"
                      r"(?:; inputs `(?P<inputs>[0-9a-f]{16,64})`)?")
INVALIDATE = re.compile(r"^\*\*RULED \d{4}-\d{2}-\d{2}: INVALIDATE `(?P<run>[^`]+)` claim `(?P<claim>[0-9a-f]{12,64})`; "
                        r"correction `(?P<corr>[0-9a-f]{40})` reviewed `(?P<rep>[^`]+)` `(?P<repsha>[0-9a-f]{16,64})`\*\*$")


class ProvenanceError(RuntimeError):
    pass


def _git(repo: Path, *args, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def _ancestor(repo: Path, a: str, b: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def row_cells(text: str, row_id: str) -> list[str] | None:
    line = next((l for l in text.splitlines() if l.startswith(f"| {row_id} |")), None)
    return [c.strip() for c in line.strip().strip("|").split("|")] if line else None


def review_verdict(report: str) -> str | None:
    """'SIGN', 'SIGN WITH EDITS' or None (anything else, including a missing verdict)."""
    lines = report.splitlines()
    if "## Verdict" not in lines:
        return None
    verdict = next((l.strip() for l in lines[lines.index("## Verdict") + 1:] if l.strip()), "")
    m = VERDICT_SIGN.match(verdict)
    return m.group("v") if m else None


def _verdict_section(report: str) -> str:
    lines = report.splitlines()
    if "## Verdict" not in lines:
        return ""
    body = []
    for line in lines[lines.index("## Verdict") + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body)


def _review_signs(report: str, reviewed: str) -> list[str]:
    """A plain SIGN whose verdict section names the reviewed commit in full."""
    v = review_verdict(report)
    out = []
    if v is None:
        out.append("the review report's verdict is not exactly SIGN")
    elif v == "SIGN WITH EDITS":
        out.append("the review is a conditional SIGN WITH EDITS: record a plain SIGN of the edited commit first")
    if reviewed not in _verdict_section(report):
        out.append("the review report's verdict section does not name the reviewed commit in full")
    return out


def _show_bytes(repo: Path, rev_path: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(repo), "show", rev_path], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def accepted_identity(repo: Path, adm: dict) -> dict:
    """The exact report bytes the gate accepted (at the exposure commit) and the commits it binds."""
    b = _show_bytes(repo, f"{adm['exposure_commit']}:{adm['review_report']}")
    return {"review_report": adm["review_report"], "review_report_sha256": hashlib.sha256(b).hexdigest() if b else None,
            "reviewed_commit": adm["reviewed_commit"], "exposure_commit": adm["exposure_commit"]}


def _sign_eric(cell: str) -> bool:
    return cell.split()[:1] == ["Eric"]


def admission_check(repo: Path, adm: dict, *, closure, admission_rel: str, register_rel: str,
                    exposure_row: str, scope_phrase: str, inputs_digest: str | None = None) -> tuple[str, list[str]]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    rc, xc, rep = adm.get("reviewed_commit"), adm.get("exposure_commit"), adm.get("review_report")
    reasons = [f"admission: {k} is not a full commit id" for k, v in (("reviewed_commit", rc), ("exposure_commit", xc))
               if not (isinstance(v, str) and HEX40.match(v))]
    if not (isinstance(rep, str) and rep.strip()):
        reasons.append("admission: review_report is unset")
    if reasons:
        return head, reasons
    ref = rc
    report_bytes = _show_bytes(repo, f"{xc}:{rep}")
    report_sha = None
    if report_bytes is None:
        reasons.append(f"admission: the review report {rep} does not exist at the exposure commit")
    else:
        report_sha = hashlib.sha256(report_bytes).hexdigest()
        reasons += [f"admission: {x}" for x in _review_signs(report_bytes.decode("utf-8", "replace"), rc)]
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
    else:
        cells = [c.strip() for c in at_x.strip().strip("|").split("|")]
        m = EXPOSURE.match(cells[1]) if len(cells) > 1 else None
        ok = bool(m and m.group("scope") == scope_phrase and m.group("rep") == rep and m.group("ref") == ref
                  and report_sha is not None and report_sha.startswith(m.group("sha"))
                  and (inputs_digest is None or (m.group("inputs") and inputs_digest.startswith(m.group("inputs")))))
        if not ok:
            reasons.append(f"admission: the {exposure_row} row is not a positive structured PREDECLARED record of "
                           f"{scope_phrase!r} binding this review report, its sha256, the reviewed commit"
                           + (" and the input pins" if inputs_digest else ""))
    changed = [f for f in _git(repo, "diff", "--name-only", ref, head, "--", *closure).stdout.split() if f != admission_rel]
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


def make_run_dir(root: Path, name: str) -> Path:
    """Create the run directory and fsync its parent, so the new entry survives a crash (r3b F1)."""
    root.mkdir(parents=True, exist_ok=True)
    d = root / name
    d.mkdir()
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return d


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
    if not (m and m.group("run") == run and claim_sha.startswith(m.group("claim")) and _sign_eric(cells[-1])):
        return [f"no register row C1-invalidate-{run} recording Eric's INVALIDATE of this claim"]
    if not _ancestor(repo, m.group("corr"), "HEAD"):
        return ["the ruling's correction commit is not in HEAD's history"]
    report = _show_bytes(repo, f"HEAD:{m.group('rep')}")
    if report is None or not hashlib.sha256(report).hexdigest().startswith(m.group("repsha")):
        return ["the cited correction review report is missing or does not match its sha prefix"]
    if _review_signs(report.decode("utf-8", "replace"), m.group("corr")):
        return ["the cited correction review is not a plain SIGN naming that correction commit in its verdict"]
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
