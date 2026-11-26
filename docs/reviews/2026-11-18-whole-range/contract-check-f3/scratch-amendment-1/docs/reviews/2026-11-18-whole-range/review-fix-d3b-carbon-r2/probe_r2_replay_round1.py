#!/usr/bin/env python3
"""Round 2 Item 1: Replay all Round-1 blocking probes of BOTH reviewers (Codex + Carbon) on 428cfa93."""
from __future__ import annotations

from ctypes import CDLL, POINTER, byref, c_char_p, c_int, c_uint, c_void_p
import importlib.util
from itertools import product
import json
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[3]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/r2/tmp"
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
assert SPEC and SPEC.loader
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def replace_one(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"expected 1 occurrence of {old!r}, got {source.count(old)}"
    return source.replace(old, new)


def inject_shim(fragment: str) -> str:
    return replace_one(tests.SHIM, 'if command == "sudo":', fragment + '\nif command == "sudo":')


def make_box(tmp: Path, *, initial: str | None = None, shim: str | None = None, url: str = "https://checks.invalid/review") -> Path:
    path = tests.box.__wrapped__(tmp)
    if shim is not None:
        for executable in (path / "shims").iterdir():
            executable.write_text(executable.read_text().split("\n", 1)[0] + "\n" + shim)
    runtime = path / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    if initial is not None:
        (runtime / "attempt").write_text(initial)
    (path / "enabled").write_text("enabled")
    (path / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (path / "home/projects/bts/.env").write_text(f"BTS_SCHEDULER_HEARTBEAT_PING_URL={url}\n")
    return path


def add_group(root: Path, group_path: str, population: int) -> None:
    loc = root / "sys/fs/cgroup" / group_path.lstrip("/")
    loc.mkdir(parents=True, exist_ok=True)
    (loc / "cgroup.procs").write_text("9999\n" if population else "")
    (loc / "cgroup.events").write_text(f"populated {population}\n")


def cgroups(calls: list[list[str]]) -> list[str]:
    return [c[-1] for c in calls if c[0] == "python3" and c[3:4] == ["cgroup"]]


def posts(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[0] == "curl" and "--data-raw" in c]


def writes(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[0] == "systemctl" and set(c[1:]) & {"disable", "enable", "stop", "restart", "daemon-reload"}]


def record(state: str = "begun", tail: str = "- - - -") -> str:
    return f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} {state} {tail}\n"


def curl_path(url: str, preserve: bool = False) -> str:
    lib = CDLL("/usr/lib/libcurl.4.dylib")
    lib.curl_url.restype = c_void_p
    lib.curl_url_cleanup.argtypes = [c_void_p]
    lib.curl_free.argtypes = [c_void_p]
    lib.curl_url_set.argtypes = [c_void_p, c_int, c_char_p, c_uint]
    lib.curl_url_get.argtypes = [c_void_p, c_int, POINTER(c_char_p), c_uint]
    handle = lib.curl_url()
    val = c_char_p()
    try:
        assert lib.curl_url_set(handle, 0, url.encode(), (1 << 4) if preserve else 0) == 0
        assert lib.curl_url_get(handle, 7, byref(val), 0) == 0
        return val.value.decode()
    finally:
        if val:
            lib.curl_free(val)
        lib.curl_url_cleanup(handle)


# 1. Codex Finding 1: dot_path + escapes + --path-as-is
def replay_codex_1_dot_path() -> dict:
    rejected = []
    for recovery, bad_url in product(
        (False, True),
        (
            "https://checks.invalid/configured/../different-check",
            "https://checks.invalid/a/./b",
            "https://checks.invalid/a/..",
            "https://checks.invalid/%2e%2e/other",
            "https://checks.invalid/check/%2fother",
        ),
    ):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), initial=record() if recovery else None, url=bad_url)
            if recovery:
                (b / "head").write_text(tests.TARGET)
            res, calls = tests.run_box_script(
                b, RECOVER if recovery else DEPLOY, "active", fault="" if recovery else "reset:17"
            )
            assert res.returncode == 1
            assert not posts(calls), f"unexpected POST for {bad_url}"
            assert "Failure ping received HTTP 2xx: unverified." in res.stdout
            assert (b / "run/bts-deploy/attempt").read_text().endswith(" unverified\n")
            rejected.append({"recovery": recovery, "url": bad_url, "posts": 0, "page": "unverified"})

    # Also verify valid URL carries --path-as-is and --globoff in both scripts, and libcurl preserves path
    sent_argvs = []
    for recovery in (False, True):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), initial=record() if recovery else None, url="https://checks.invalid/configured/check")
            if recovery:
                (b / "head").write_text(tests.TARGET)
            res, calls = tests.run_box_script(
                b, RECOVER if recovery else DEPLOY, "active", fault="" if recovery else "reset:17"
            )
            (argv,) = posts(calls)
            assert "--path-as-is" in argv and "--globoff" in argv
            assert argv[-1] == "https://checks.invalid/configured/check/fail"
            assert "Failure ping received HTTP 2xx: verified." in res.stdout
            sent_argvs.append({"recovery": recovery, "has_path_as_is": "--path-as-is" in argv, "url": argv[-1]})

    # Confirm libcurl behavior with --path-as-is vs without on Codex's original dot URL
    sample_dot = "https://checks.invalid/configured/../different-check/fail"
    assert curl_path(sample_dot, preserve=False) == "/different-check/fail"
    assert curl_path(sample_dot, preserve=True) == "/configured/../different-check/fail"

    return {
        "status": "FIXED",
        "rejected_count": len(rejected),
        "valid_posts": sent_argvs,
        "libcurl_preserve_verified": True,
    }


# 2. Codex Finding 2a & 2b (exact MANAGER shim from review-fix-d3b-codex/counterexamples.py)
CODEX_MANAGER = inject_shim(r'''
if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"]:
    old_stop = (root / "legacy-stopped").exists()
    group = "/legacy-group" if scenario == "not-found-with-group" else "/legacy-pre"
    if scenario == "legacy-fragment-present" and old_stop:
        group = "/legacy-post"
    fields = dict(Id=args[1], LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup=group)
    if scenario == "legacy-fragment-present" and not old_stop:
        fields.update(LoadState="loaded", ActiveState="active", MainPID="5042")
    for flag in args:
        if flag in fields:
            print(f"{flag}={fields[flag]}")
    raise SystemExit(0)
if command == "systemctl" and args == ["is-enabled", "bts-scheduler.service"]:
    print("disabled")
    raise SystemExit(1)
''')


def replay_codex_2_absent() -> dict:
    cases = []
    for recovery, mode in product((False, True), ("not-found-with-group", "legacy-fragment-present")):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), initial=record() if recovery else None, shim=CODEX_MANAGER)
            if recovery:
                (b / "head").write_text(tests.TARGET)
            if mode == "legacy-fragment-present":
                frag = b / "etc/systemd/system/bts-scheduler.service"
                frag.parent.mkdir(parents=True)
                frag.write_text("[Service]\nType=oneshot\nExecStart=/bin/false\n")
                add_group(b, "/legacy-pre", 0)
                add_group(b, "/legacy-post", 0)
            else:
                add_group(b, "/legacy-group", 1)
            res, calls = tests.run_box_script(
                b, RECOVER if recovery else DEPLOY, mode, fault="" if recovery else "reset:17"
            )
            assert res.returncode == 1
            assert "Scheduler stopped: unverified." in res.stdout
            att = (b / "run/bts-deploy/attempt").read_text()
            assert " unverified verified verified\n" in att
            if mode == "not-found-with-group":
                assert "/legacy-group" in cgroups(calls)
                assert ["systemctl", "stop", "bts-scheduler.service"] in writes(calls)
            cases.append({
                "mode": mode,
                "recovery": recovery,
                "stop_line": "unverified",
                "checked_cgroups": cgroups(calls),
            })
    return {"status": "FIXED", "cases": cases}


# 3. Carbon B1, B2, B3 (exact scenarios from probe_counterexamples_d3b_v3.py & probe_ce3_dangling_fragment_fallthrough.py)
def build_carbon_shim() -> str:
    daemon_reload_hook = '''    if not user and args[0] == "daemon-reload":
        (root / "daemon-reloaded").touch()
    if user and args[0] in {"disable", "enable"}:'''
    shim = replace_one(tests.SHIM, '    if user and args[0] in {"disable", "enable"}:', daemon_reload_hook)

    show_hook = '''        if not user and args[1] == "bts-scheduler.service":
            if scenario == "ce1-absent-with-cgroup":
                states.update(LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup="/legacy-orphan")
            elif scenario == "ce1b-absent-then-post-cgroup":
                seen = root / "seen-first-legacy-show"
                cg = "/legacy-orphan" if seen.exists() else ""
                if args[2:] == ["-p", "ControlGroup"]:
                    seen.touch()
                states.update(LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup=cg)
            elif scenario in {"ce2-phase2-unload-populated", "ce2b-phase2-unload-nogroup"}:
                frag = root / "etc/systemd/system/bts-scheduler.service"
                unloaded = (root / "legacy-stopped").exists() and not frag.exists() and (root / "daemon-reloaded").exists()
                if not unloaded:
                    cg = "" if scenario.endswith("nogroup") else "/legacy-orphan"
                    active_st = "inactive" if (root / "legacy-stopped").exists() else "active"
                    pid_st = "0" if (root / "legacy-stopped").exists() else "1234"
                    states.update(LoadState="loaded", ActiveState=active_st, MainPID=pid_st, ControlGroup=cg)
                else:
                    states.update(LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup="")
        if "ControlGroup" in args and user and scenario == "main-pid":'''
    return replace_one(shim, '        if "ControlGroup" in args and user and scenario == "main-pid":', show_hook)


def replay_carbon_b1_b2_b3() -> dict:
    shim = build_carbon_shim()
    results = {}

    # Carbon B1 (CE1 + CE1b)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), shim=shim)
        add_group(b, "/legacy-orphan", 1)
        dep_res, dep_calls = tests.run_box_script(b, DEPLOY, "ce1-absent-with-cgroup", fault="reset:17")
        assert "Scheduler stopped: unverified." in dep_res.stdout
        assert "/legacy-orphan" in cgroups(dep_calls)
        (b / "calls").unlink()
        rec_res, rec_calls = tests.run_box_script(b, RECOVER, "ce1-absent-with-cgroup")
        assert "Scheduler stopped: unverified." in rec_res.stdout
        assert "containment re-checked" not in rec_res.stdout
        assert "/legacy-orphan" in cgroups(rec_calls)
        (b / "run/bts-deploy/attempt").write_text(record())
        (b / "calls").unlink()
        rec_begun, rec_begun_calls = tests.run_box_script(b, RECOVER, "ce1-absent-with-cgroup")
        assert "Scheduler stopped: unverified." in rec_begun.stdout
        assert "/legacy-orphan" in cgroups(rec_begun_calls)
        results["Carbon_B1_CE1"] = "FIXED (deploy + contained recovery + begun recovery all unverified & inspect /legacy-orphan)"

    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), shim=shim)
        add_group(b, "/legacy-orphan", 1)
        res_1b, calls_1b = tests.run_box_script(b, DEPLOY, "ce1b-absent-then-post-cgroup", fault="reset:17")
        assert "Scheduler stopped: unverified." in res_1b.stdout
        assert "/legacy-orphan" in cgroups(calls_1b)
        results["Carbon_B1_CE1b"] = "FIXED (blank first sample then populated post-sample cgroup -> unverified)"

    # Carbon B2 (CE2 + CE2b: Phase-2 legacy cleanup unloads unit before contest-state failure)
    for scen, pop in (("ce2-phase2-unload-populated", 1), ("ce2b-phase2-unload-nogroup", None)):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), shim=shim)
            if pop is not None:
                add_group(b, "/legacy-orphan", pop)
            frags = b / "etc/systemd/system"
            frags.mkdir(parents=True)
            (frags / "bts-scheduler.service").write_text("[Service]\nExecStart=/bin/true\n")
            dep_res, dep_calls = tests.run_box_script(b, DEPLOY, scen, fault="contest-state:17")
            assert dep_res.returncode == 1
            assert ["systemctl", "daemon-reload"] in dep_calls
            assert (b / "run/bts-deploy/legacy.123.1").is_file()
            assert "Scheduler stopped: unverified." in dep_res.stdout
            (b / "calls").unlink()
            rec_res, _ = tests.run_box_script(b, RECOVER, scen)
            assert "Scheduler stopped: unverified." in rec_res.stdout
            assert "containment re-checked" not in rec_res.stdout
            results[f"Carbon_B2_{scen}"] = "FIXED (legacy.123.1 mark written; deploy + recovery both unverified)"

    # Carbon B3 (CE3: legacy-vanishing with dangling fragment symlink)
    for mode, is_recovery in (("deploy", False), ("recovery", True)):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = tests.box.__wrapped__(Path(td))
            tests.d3b_legacy_group(b, 0)
            units = b / "etc/systemd/system"
            units.mkdir(parents=True)
            (units / "bts-scheduler.service").symlink_to(b / "missing-unit-file")
            res, _ = tests.d3b_run_contained(b, is_recovery, "legacy-vanishing")
            assert res.returncode == 1
            assert "Scheduler stopped: unverified." in res.stdout
            results[f"Carbon_B3_CE3_{mode}"] = "FIXED (not-found with dangling symlink -> unverified)"

    return results


# 4. Codex Finding 3 / Carbon N2: exact timing_model.py from review-fix-d3b-codex/timing_model.py
TIMING_SHIM = inject_shim(r'''
in_handler = root / "in-handler"
virtual_time = root / "home/projects/bts/data/virtual-elapsed"
if command == "python3" and args[2] == "clock" and virtual_time.exists():
    in_handler.touch()
    raise SystemExit(124)

if command == "timeout" and in_handler.exists():
    grace, term = map(int, args[1:3])
    delta = virtual_time
    elapsed = int(delta.read_text())
    prog = Path(args[3]).name
    words = args[4:]
    if prog == "sudo":
        prog = "systemctl" if "systemctl" in words else Path(words[2]).name
    if prog == "systemctl":
        scope = "user" if "--user" in words else "legacy"
        action = next(w for w in words if w in ("disable", "stop", "show", "list-jobs", "is-enabled"))
        label = f"{scope}-{action}"
        if action == "show" and words[-2:] == ["-p", "ControlGroup"] and "ActiveState" not in words:
            label += "-group"
    elif prog == "python3":
        label = words[words.index("-c") + 2]
    elif prog == "bash":
        label = "recheck" if any("containment re-checked" in word for word in words) else "log"
        if any("Scheduler stopped:" in word for word in words):
            label = "report"
    else:
        label = prog
    delays = {
        "id": 4, "user-show-group": 4, "user-disable": 4, "user-stop": 54,
        "legacy-show": 4, "legacy-list-jobs": 4, "legacy-show-group": 4,
        "legacy-disable": 4, "legacy-stop": 54, "user-is-enabled": 14,
        "user-show": 14, "user-list-jobs": 14, "cgroup": 14, "legacy-is-enabled": 14,
        "pgrep": 40, "marker": 40, "endpoint": 14, "curl": 10,
        "stage-record": 20,
    }
    wait = delays.get(label, 0)
    if label == "legacy-list-jobs" or label == "legacy-show" and elapsed >= 180:
        wait = 20
    if label == "endpoint":
        wait = min(wait, term - 1)
    if scenario == "early":
        wait = 0
    if scenario == "late-successful" and label in ("pgrep", "marker"):
        wait = min(wait, term - 1)
    if scenario == "late-successful" and label == "stage-record":
        wait = 1
    rounded_success = False
    if scenario == "blocked-final-recheck":
        if label.startswith("legacy-") and elapsed >= 198 or label == "cgroup" and elapsed >= 198:
            wait = 0
        if label in ("pgrep", "marker", "endpoint", "stage-record"):
            wait = term
            rounded_success = True
        if label in ("mv", "rm", "report", "recheck"):
            wait = term + grace
    timed_out = wait >= term and not rounded_success
    end = elapsed + (term + grace if timed_out else wait)
    delta.write_text(str(end))
    with (root / "timeline").open("a") as log:
        print(json.dumps(dict(label=label, start=elapsed, end=end, term=term, grace=grace,
                              timed_out=timed_out, scope=prog)), file=log)
    if timed_out:
        raise SystemExit(124)

if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"]:
    states = dict(Id=args[1], ActiveState="inactive", MainPID="0", LoadState="loaded", ControlGroup="/legacy")
    for key in args:
        if key in states:
            print(f"{key}={states[key]}")
    raise SystemExit(0)
''')


def instrument_timing(script: str) -> str:
    script = replace_one(
        script,
        "local budget_start=$SECONDS budget_origin\n",
        'SECONDS=0\n    printf 0 > "$checkout/data/virtual-elapsed"\n    local budget_start=$SECONDS budget_origin\n',
    )
    script = replace_one(
        script,
        "if [[ -n ${budget_start:-} ]]; then\n        now=",
        'if [[ -n ${budget_start:-} ]]; then\n'
        '        if [[ -e $checkout/data/virtual-elapsed ]]; then\n'
        '            IFS= read -r SECONDS < "$checkout/data/virtual-elapsed"\n'
        '        fi\n        now=',
    )
    return script


def replay_codex_3_timing_model() -> dict:
    # 1. Blocked final recheck (was ending at 281s on 90a9b6f0; must now end at 279s <= 280s)
    initial = record("contained", tail="head unverified unverified unverified")
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), initial=initial, shim=TIMING_SHIM)
        (b / "head").write_text(tests.TARGET)
        res, calls = tests.run_box_script(b, instrument_timing(RECOVER), "blocked-final-recheck")
        timeline = [json.loads(line) for line in (b / "timeline").read_text().splitlines()]
        assert res.returncode == 1
        assert (b / "run/bts-deploy/attempt").read_text() == initial
        assert len(posts(calls)) == 1 and not res.stdout
        assert timeline[-1]["label"] == "report"
        assert timeline[-1]["end"] == 279 <= 280
        assert not any(entry["label"] == "recheck" for entry in timeline)
        blocked_tail = [(x["label"], x["start"], x["end"]) for x in timeline[-3:]]

    # 2. All 6 (recovery x early/cap-stress/late-successful) window runs stay <= 280s
    window_ends = {}
    for recovery in (False, True):
        for scenario in ("early", "cap-stress", "late-successful"):
            with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                b = make_box(Path(td), initial=record() if recovery else None, shim=TIMING_SHIM)
                if recovery:
                    (b / "head").write_text(tests.TARGET)
                res, calls = tests.run_box_script(
                    b,
                    instrument_timing(RECOVER if recovery else DEPLOY),
                    scenario,
                    fault="" if recovery else "reset:17",
                )
                tl = [json.loads(line) for line in (b / "timeline").read_text().splitlines()]
                assert res.returncode == 1 and tl[-1]["end"] <= 280
                steps = {entry["label"]: entry for entry in tl}
                assert steps["pgrep"]["term"] >= 5 and steps["pgrep"]["end"] <= 220
                assert steps["marker"]["end"] <= 240 and steps["endpoint"]["end"] <= 254
                assert steps["curl"]["term"] >= 10 and steps["curl"]["end"] <= 270
                window_ends[f"{'recover' if recovery else 'deploy'}:{scenario}"] = tl[-1]["end"]

    return {
        "status": "FIXED",
        "blocked_final_recheck_tail": blocked_tail,
        "window_model_ends": window_ends,
    }


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    summary = {
        "codex_1_url_dot_and_escape": replay_codex_1_dot_path(),
        "codex_2_legacy_absence_and_fragment": replay_codex_2_absent(),
        "carbon_b1_b2_b3": replay_carbon_b1_b2_b3(),
        "codex_3_carbon_n2_single_final_print_timing": replay_codex_3_timing_model(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
