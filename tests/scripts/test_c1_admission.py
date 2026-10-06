"""The shared C1 admission gate (used by rank 3 and later candidates; the deferred 4b review's B1/B2 lessons)."""
import hashlib
import json
import subprocess

import pytest

from scripts.audit.c1 import admission as A


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, files, msg):
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


SCOPE = "historical count build"


def review(sha, verdict="**SIGN.**", extra=""):
    return f"# Review\n\n## Verdict\n\n{verdict} Reviewed `{sha}`.\n\n## Findings\n{extra}"


def x34_row(report_path, report_text, ref, scope=SCOPE, *, prefix="PREDECLARED 2026-10-07", inputs=None):
    sha = hashlib.sha256(report_text.encode()).hexdigest()
    tail = f"; inputs `{inputs[:16]}`" if inputs else ""
    return f"| X-34 | **{prefix}: {scope}**; review `{report_path}` sha256 `{sha[:16]}`; reviewed `{ref}`{tail} | ... |\n"


def make_repo(tmp_path, *, verdict="**SIGN.**", row=None, report_extra=""):
    """R reviewed; S archives the review; X publishes X-34 citing it; Y sets the exposure commit."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    R = commit(r, {"pkg/a.py": "x = 1\n", "pkg/admission.json": "{}", "reg.md": "| X-33 | y |\n"}, "reviewed")
    rep = review(R, verdict, report_extra)
    commit(r, {"docs/review-r1.md": rep}, "archive the review")
    adm = {"reviewed_commit": R, "review_report": "docs/review-r1.md"}
    X = commit(r, {"reg.md": "| X-33 | y |\n" + (row(rep, R) if row else x34_row("docs/review-r1.md", rep, R)),
                   "pkg/admission.json": json.dumps(adm)}, "publish X-34")
    adm = {**adm, "exposure_commit": X}
    commit(r, {"pkg/admission.json": json.dumps(adm)}, "admission")
    return r, adm


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


def check(r, adm, **kw):
    return A.admission_check(r, adm, closure=("pkg",), admission_rel="pkg/admission.json", register_rel="reg.md",
                             exposure_row="X-34", scope_phrase=SCOPE, **kw)[1]


def test_admission_passes_with_a_recorded_sign_and_metadata_only_edits(repo):
    r, adm = repo
    assert check(r, adm) == []
    ident = A.accepted_identity(r, adm)
    assert ident["review_report_sha256"] == hashlib.sha256((r / "docs/review-r1.md").read_bytes()).hexdigest()


def test_admission_refuses_unset_fields(repo):
    r, adm = repo
    for k in ("reviewed_commit", "exposure_commit", "review_report"):
        assert check(r, {**adm, k: None}), k


def test_only_a_plain_sign_naming_the_commit_in_its_verdict_counts():
    """r3b r1 F2 and r2 N1: no prefix tricks, no conditional acceptance, and no hash mentioned only elsewhere."""
    sha = "a" * 40
    assert A._review_signs(review(sha), sha) == []
    assert any("conditional" in x for x in A._review_signs(review(sha, "**SIGN WITH EDITS.**"), sha))
    for v in ("**SIGN OFF REFUSED.**", "**SIGN WITH EDITS pending application.**", "**SIGNATURE PENDING**",
              "**BLOCK.**", "SIGN."):
        assert A._review_signs(review(sha, v), sha), v
    other = "b" * 40
    historical = review(other, extra=f"\nPrevious review of `{sha}` was BLOCK.\n")
    assert any("verdict section" in x for x in A._review_signs(historical, sha))


def test_sign_with_edits_is_refused_until_a_plain_sign_is_recorded(tmp_path):
    r, adm = make_repo(tmp_path, verdict="**SIGN WITH EDITS.**")
    assert any("conditional" in x for x in check(r, adm))


def test_the_review_report_must_exist_at_the_exposure_commit(repo):
    r, adm = repo
    commit(r, {"docs/late.md": review(adm["reviewed_commit"])}, "a report archived after the exposure commit")
    assert any("does not exist at the exposure commit" in x for x in check(r, {**adm, "review_report": "docs/late.md"}))


@pytest.mark.parametrize("row", [
    lambda rep, R: "| X-34 | unrelated diagnostic; no historical fitting approved | withheld | Codex |\n",
    lambda rep, R: x34_row("docs/review-r1.md", rep, R, prefix="DENIED 2026-10-07"),
    lambda rep, R: x34_row("docs/review-r1.md", rep, R, scope="historical count build DENIED; do not read inputs"),
    lambda rep, R: x34_row("docs/review-r1.md", rep, R, scope="diagnostic only"),
    lambda rep, R: x34_row("docs/review-r1.md", rep + "tampered", R),
    lambda rep, R: x34_row("docs/other.md", rep, R),
    lambda rep, R: x34_row("docs/review-r1.md", rep, "0" * 40),
])
def test_the_exposure_row_must_be_a_positive_structured_record(tmp_path, row):
    r, adm = make_repo(tmp_path, row=row)
    assert any("positive structured PREDECLARED" in x for x in check(r, adm))


def test_the_exposure_row_must_bind_the_input_pins_when_required(tmp_path):
    pins = "c" * 64
    r, adm = make_repo(tmp_path, row=lambda rep, R: x34_row("docs/review-r1.md", rep, R, inputs=pins))
    assert check(r, adm, inputs_digest=pins) == []
    assert check(r, adm, inputs_digest="d" * 64)
    (tmp_path / "w2").mkdir()
    r2, adm2 = make_repo(tmp_path / "w2")
    assert check(r2, adm2, inputs_digest=pins)


def test_a_replacement_commit_with_a_new_report_is_refused(repo):
    r, adm = repo
    R2 = commit(r, {"pkg/a.py": "x = 2\n"}, "replacement code")
    commit(r, {"docs/review-r2.md": review(R2)}, "a new report")
    assert check(r, {**adm, "reviewed_commit": R2, "review_report": "docs/review-r2.md"})


def test_a_commit_that_changes_code_cannot_declare_itself_reviewed(repo):
    r, adm = repo
    X2 = commit(r, {"reg.md": "| X-33 | y |\n| X-34 | edited | ... |\n", "pkg/a.py": "x = 2\n"}, "row edit and code")
    assert check(r, {**adm, "reviewed_commit": X2, "exposure_commit": X2})


def test_executable_changes_untracked_sources_and_row_edits_refuse(repo):
    r, adm = repo
    (r / "pkg" / "shadow.py").write_text("y = 1\n")
    assert any("untracked" in x for x in check(r, adm))
    (r / "pkg" / "shadow.py").unlink()
    commit(r, {"pkg/a.py": "x = 3\n"}, "a later code change")
    assert any("changed since the reviewed commit" in x for x in check(r, adm))


def test_foreign_imports(monkeypatch, tmp_path):
    import sys
    import types
    m = types.ModuleType("bts.shadowed")
    m.__file__ = "/elsewhere/bts/shadowed.py"
    monkeypatch.setitem(sys.modules, "bts.shadowed", m)
    assert "bts.shadowed: /elsewhere/bts/shadowed.py" in A.foreign_imports(A.REPO)


def inv_row(run, claim_sha, correction, report="docs/fix-review.md", report_sha="0" * 16, *, verb="INVALIDATE",
            source="Eric 2026-10-06"):
    return (f"| C1-invalidate-{run} | x | **RULED 2026-10-06: {verb} `{run}` claim `{claim_sha[:16]}`; "
            f"correction `{correction}` reviewed `{report}` `{report_sha[:16]}`** | {source} |\n")


def test_a_claimed_run_is_released_only_by_eric_s_exact_invalidation_with_a_reviewed_correction(tmp_path):
    """4b r3 B1, r3b r1 F3, r2 N1: the ruling names this run and claim, a correction in HEAD's history, and a plain
    SIGN (at HEAD, sha-bound) whose verdict names that correction; the source token is exactly Eric."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    C = commit(r, {"a.py": "fixed\n"}, "the correction")
    good_rep, cond_rep = review(C), review(C, "**SIGN WITH EDITS.**")
    commit(r, {"docs/fix.md": good_rep, "docs/cond.md": cond_rep}, "reviews")
    root = tmp_path / "runs"
    d = A.make_run_dir(root, "abc1234-20261006T000000Z")
    A.write_claim(d, "f" * 40)
    claim = hashlib.sha256((d / "CLAIM.json").read_bytes()).hexdigest()
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    (root / f"INVALIDATION_{d.name}.json").write_text(json.dumps({"run": d.name, "claim_sha256": claim}))
    ok = inv_row(d.name, claim, C, "docs/fix.md", sha(good_rep))
    assert A.invalidation_problems({"run": d.name, "claim_sha256": claim}, d.name, claim, ok, repo=r) == []
    for label, reg in {"none": "",
                       "denial": inv_row(d.name, claim, C, "docs/fix.md", sha(good_rep), verb="DO NOT INVALIDATE"),
                       "Erica": inv_row(d.name, claim, C, "docs/fix.md", sha(good_rep), source="Erica 2026-10-06"),
                       "other claim": inv_row(d.name, "e" * 64, C, "docs/fix.md", sha(good_rep)),
                       "correction not in history": inv_row(d.name, claim, "0" * 40, "docs/fix.md", sha(good_rep)),
                       "report sha mismatch": inv_row(d.name, claim, C, "docs/fix.md", "0" * 16),
                       "conditional correction review": inv_row(d.name, claim, C, "docs/cond.md", sha(cond_rep)),
                       "unrelated ruling": f"| C1-invalidate-{d.name} | x | **RULED 2026-10-06: A** | Eric |\n"}.items():
        assert A.invalidation_problems({"run": d.name, "claim_sha256": claim}, d.name, claim, reg, repo=r), label
    assert A.invalidation_problems({}, d.name, claim, ok, repo=r)


def test_make_run_dir_creates_and_refuses_reuse(tmp_path):
    d = A.make_run_dir(tmp_path / "runs", "r1")
    assert d.is_dir()
    with pytest.raises(FileExistsError):
        A.make_run_dir(tmp_path / "runs", "r1")


def test_the_admission_lock_refuses_a_second_holder(tmp_path):
    with A.admission_lock(tmp_path):
        with pytest.raises(SystemExit, match="lock"):
            with A.admission_lock(tmp_path):
                pass


def test_read_once_hashes_and_returns_the_same_bytes(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    b, h = A.read_pinned(f, hashlib.sha256(b"abc").hexdigest())
    assert b == b"abc"
    with pytest.raises(A.ProvenanceError):
        A.read_pinned(f, "0" * 64)
