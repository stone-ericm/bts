"""Replay both reviewers' original triggers against the current shell literals."""
from pathlib import Path
from subprocess import run
import sys
from tempfile import TemporaryDirectory

from shared import HERE, ROOT, DEPLOY, RECOVER, box, group, invoke, posts, record
import counterexamples as codex

CARBON = HERE.parent / "main-docs/docs/reviews/2026-11-18-whole-range/review-fix-d3b-carbon"


def carbon(name):
    if name == "groups":
        source = (CARBON / "probe_counterexamples_d3b_v3.py").read_text()
        # Keep the original inputs and replace the former failure expectations.
        source = source[:source.index("    fixed_deploy = apply_smallest_fix(DEPLOY, is_deploy=True)")]
        source += '    print(json.dumps(report, indent=2))\n\nmain()\n'
        source = source.replace('assert "Scheduler stopped: verified."', 'assert "Scheduler stopped: unverified."')
        source = source.replace('assert "/legacy-orphan" not in dep_checked', 'assert "/legacy-orphan" in dep_checked')
        source = source.replace('assert "/legacy-orphan" not in cgroup_checks(calls_1b)',
                                'assert "/legacy-orphan" in cgroup_checks(calls_1b)')
        source = source.replace('assert "DEPLOY RECOVERY — containment re-checked." in',
                                'assert "DEPLOY RECOVERY — containment re-checked." not in')
    else:
        source = (CARBON / "probe_ce3_dangling_fragment_fallthrough.py").read_text()
        source = source[:source.index("    # Verify smallest fix for CE3")]
        source += '    print(json.dumps(out, indent=2))\n\nmain()\n'
        source = source.replace('assert "Scheduler stopped: verified."', 'assert "Scheduler stopped: unverified."')
    source = source.replace("REPO = Path(__file__).resolve().parents[2]", f"REPO = Path({str(ROOT)!r})")
    source = source.replace('SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"',
                            'SCRATCH_PARENT = Path(__file__).resolve().parent')
    with TemporaryDirectory(prefix="carbon-replay-", dir=HERE) as name:
        replay = Path(name) / "replay.py"
        replay.write_text(source)
        run([sys.executable, str(replay)], cwd=ROOT, check=True, timeout=54)


def own_groups():
    for recovery in (False, True):
        for mode in ("not-found-with-group", "legacy-fragment-present"):
            with box(record() if recovery else None, shim=codex.MANAGER) as path:
                if mode == "legacy-fragment-present":
                    fragment = path / "etc/systemd/system/bts-scheduler.service"
                    fragment.parent.mkdir(parents=True)
                    fragment.write_text("[Service]\nType=oneshot\nExecStart=/bin/false\n")
                    group(path, "/legacy-pre", 0)
                    group(path, "/legacy-post", 0)
                else:
                    group(path, "/legacy-group", 1)
                state, _ = invoke(path, f"Codex B2 {mode} {recovery=}", RECOVER if recovery else DEPLOY,
                                  mode, fault="" if recovery else "reset:17")
                print((path / "observations").read_text(), end="")
                assert state["rc"] == 1 and "Scheduler stopped: unverified." in state["out"]
                assert ("legacy-group" in " ".join(state["groups"])) == (mode == "not-found-with-group")


def urls():
    cases = (
        ("https://checks.invalid/configured/../different-check", False),
        ("https://checks.invalid/configured/./same-check", False),
        ("https://checks.invalid/configured/%2e%2e/different-check", False),
        ("https://hc-ping.com/123e4567-e89b-12d3-a456-426614174000", True),
        ("https://hc-ping.com/project-key/my-job/", True),
        ("https://hc-ping.com/project-key/my%2Djob", False),
    )
    for recovery in (False, True):
        for url, allowed in cases:
            with box(record() if recovery else None, url=url) as path:
                state, calls = invoke(path, f"Codex B1 {url=} {recovery=}", RECOVER if recovery else DEPLOY,
                                      fault="" if recovery else "reset:17")
                assert bool(posts(calls)) == allowed
                expected = "verified" if allowed else "unverified"
                assert f"Failure ping received HTTP 2xx: {expected}." in state["out"]
                if allowed:
                    argv = posts(calls)[0]
                    assert "--path-as-is" in argv and "--globoff" in argv
                    assert argv[-1] == url.rstrip("/") + "/fail"
    escaped = cases[-1][0]
    print("Escaping an unreserved hyphen preserves the decoded slug; R1 explicitly rejects the escaped spelling.")
    from urllib.parse import unquote
    assert unquote(escaped) == cases[-2][0].rstrip("/")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
