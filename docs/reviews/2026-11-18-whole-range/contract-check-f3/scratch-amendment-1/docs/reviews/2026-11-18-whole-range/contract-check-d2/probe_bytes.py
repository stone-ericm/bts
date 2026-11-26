"""Compare D2's output bytes against local command shims, with no host commands."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


BASE = "fix/wr-deploy"
SCHEDULER = "bts-scheduler.service"
DASHBOARD = "bts-dashboard.service"
DROP_PATH = f"/etc/systemd/user/{SCHEDULER}.d/50-deploy-hold.conf"
DROP = (
    b"[Service]\nExecCondition=/usr/bin/python3 -I "
    b"/usr/local/bin/bts-deploy-hold check-clear --scheduler\n"
)
INSTALLER = "scripts/install-deploy-guard-hetzner.sh"
CRON = "scripts/cron-setup-hetzner.sh"
PATHS = (INSTALLER, CRON, f"scripts/systemd/{SCHEDULER}", f"scripts/systemd/{DASHBOARD}")
OVERRIDE = b"BTS_TEST_DEPLOY_GUARD_DIR"


def git_blob(path):
    return subprocess.run(
        ["/usr/bin/git", "show", f"{BASE}:{path}"],
        stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=2,
    ).stdout


def get_rendered_cron(staging, source, intent):
    # The starting point has no render command. Its show action emits the same
    # BTS payload as install. Discard the preview heading and normalize HOME.
    home = staging / "home"
    uv = home / ".local/bin/uv"
    uv.parent.mkdir(parents=True, exist_ok=True)
    (home / "projects/bts").mkdir(parents=True, exist_ok=True)
    uv.write_text(f"#!/bin/sh\nprintf '%s\\n' {intent}\n")
    uv.chmod(0o700)
    crontab = uv.with_name("crontab")
    crontab.write_text("#!/bin/sh\nexit 1\n")
    crontab.chmod(0o700)
    result = subprocess.run(
        ["/bin/bash", "-s", "--", "show"], input=source, capture_output=True,
        env={
            "HOME": str(home), "PATH": f"{uv.parent}:/usr/bin:/bin",
            "HEALTHCHECKS_PING_URL": "https://example.invalid/ping",
        }, check=True, timeout=2,
    ).stdout
    payload = result.split(b"Would install:\n", 1)[1].replace(os.fsencode(home), b"/home/bts")
    assert payload.startswith(b"# BTS-HETZNER\n") and payload.endswith(b"\n")
    assert all(line.endswith(b"# BTS-HETZNER") for line in payload.splitlines())
    return payload


def expected_unit(unit, fragment):
    path = f"/home/bts/.config/systemd/user/{unit}"
    cat = f"# {path}\n".encode() + fragment
    drop_path = ""
    if unit == SCHEDULER:
        cat += f"\n# {DROP_PATH}\n".encode() + DROP
        drop_path = DROP_PATH
    show = f"LoadState=loaded\nFragmentPath={path}\nDropInPaths={drop_path}\nNeedDaemonReload=no\n"
    return {f"cat-{unit}": cat, f"show-{unit}": show.encode()}


def shim(argv):
    staging, command, *args = argv
    if command == "crontab":
        assert args == ["-u", "bts", "-l"]
        key = "cron"
    else:
        assert command == "systemctl" and args[0] == "--user"
        action, *rest = args[1:]
        key = "show-environment" if action == "show-environment" else f"{action}-{rest[0]}"
    try:
        with (Path(staging) / key).open("rb") as channel:
            sys.stdout.buffer.write(channel.read(65537))
    except FileNotFoundError:
        return 1
    return 0


def invoke(staging, key):
    if key == "cron":
        command = ["crontab", "-u", "bts", "-l"]
    elif key == "show-environment":
        command = ["systemctl", "--user", "show-environment"]
    else:
        verb, unit = key.split("-", 1)
        command = ["systemctl", "--user", verb, unit]
        if verb == "show":
            command.extend(["-p", "LoadState", "-p", "FragmentPath",
                            "-p", "DropInPaths", "-p", "NeedDaemonReload"])
    result = subprocess.run(
        [sys.executable, __file__, "shim", str(staging), *command],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=2,
    )
    return result.stdout if result.returncode == 0 else None


def digest(data):
    return hashlib.sha256(data).digest()


def refusal(key, expected, observed):
    if observed is None or digest(observed) != digest(expected):
        return "crontab" if key == "cron" else "guard"
    return None


def check(staging, key, original, changed, expected):
    if changed is None:
        (staging / key).unlink()
    else:
        (staging / key).write_bytes(changed)
    assert refusal(key, original, invoke(staging, key)) == expected, key
    (staging / key).write_bytes(original)


def cron_cases(cron):
    prefix = b"0 4 * * * "
    monitor = next(line for line in cron.splitlines(keepends=True) if b"check_heartbeat.py" in line)
    writers = (
        b"bts 'check-results'", b"python -m bts.cli reconcile", b"(bts reconcile)",
        b"true && bts reconcile", rb"bts re\concile", rb"b\ts check-results",
        b"cmd=reconcile && bts $cmd",
        b".venv/bin/python -c 'from bts.cli import cli; cli()' reconcile",
        b"/usr/local/bin/bts-deploy-hold run check-results -- /bin/sh -c 'echo bts check-results'",
    )
    for writer in writers:
        yield cron + prefix + writer + b"\n"
    for changed in (
        b"#" + monitor, monitor.replace(b"/var/lib/bts-deploy/hold", b"/tmp/other"),
        monitor.replace(b"*/5 * * * *", b"@reboot"),
        monitor.replace(b"cd ", b"true || cd ", 1),
        monitor.replace(b" >>", b" --deploy-hold=/tmp/other >>"),
    ):
        yield cron.replace(monitor, changed)
    header, first, second, rest = cron.split(b"\n", 3)
    yield b"\n".join((header, second, first, rest))
    yield b"\n".join((header, second, rest))
    yield cron.replace(b"0 1 *", b"0  1 *", 1)
    yield cron.replace(b"\n", b"\r\n")
    yield cron[:-1]
    yield cron + b"0 0 * * * /bin/true\n"
    yield b"no crontab for bts\n"
    yield None


def main():
    blobs = {path: git_blob(path) for path in PATHS}
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        staging = Path(directory)
        enter = get_rendered_cron(staging, blobs[CRON], "enter")
        research = get_rendered_cron(staging, blobs[CRON], "research")
        assert b"check-pick-entered" in enter and b"check-pick-entered" not in research
        sources = {"cron": enter, "show-environment": b"PATH=/usr/bin:/bin\n"}
        for unit in (SCHEDULER, DASHBOARD):
            sources.update(expected_unit(unit, blobs[f"scripts/systemd/{unit}"]))
        for key, source in sources.items():
            (staging / key).write_bytes(source)
            assert invoke(staging, key) == source
        cron_count = 0
        for changed in cron_cases(enter):
            check(staging, "cron", enter, changed, "crontab")
            cron_count += 1
        cat_key, show_key = f"cat-{SCHEDULER}", f"show-{SCHEDULER}"
        cat, show = sources[cat_key], sources[show_key]
        directives = (
            b"SuccessExitStatus=1", b"WorkingDirectory=/other", b"Environment=FOO=bar",
            b"EnvironmentFile=/tmp/other", b"RootDirectory=/tmp/root",
            b"BindPaths=/tmp", b"BindReadOnlyPaths=/tmp", b"TemporaryFileSystem=/tmp",
            b"PassEnvironment=FOO", b"UnsetEnvironment=FOO", b"ExecStartPre=/bin/true",
            b"ExecStartPost=/bin/true", b"ExecStopPost=/bin/true",
            b"Environment=BTS_TEST_DEPLOY_GUARD_DIR=/tmp/other",
        )
        cat_cases = [cat.replace(b"/usr/local/bin/bts-deploy-hold", b"/tmp/other"),
                     cat.removesuffix(DROP), None]
        for root in (b"/home/bts/.config/systemd/user", b"/run/user/1000/systemd/user"):
            cat_cases.append(cat + b"\n# " + root + b"/bts-scheduler.service.d/90-foreign.conf\n[Service]\n")
        cat_cases.extend(cat + b"\n[Service]\n" + directive + b"\n" for directive in directives)
        show_cases = [show.replace(b"LoadState=loaded", b"LoadState=masked"),
                      show.replace(b"NeedDaemonReload=no", b"NeedDaemonReload=yes")]
        for changed in cat_cases:
            check(staging, cat_key, cat, changed, "guard")
        for changed in show_cases:
            check(staging, show_key, show, changed, "guard")
        for path, content in blobs.items():
            changed = dict(blobs, **{path: content + b"\n"})
            assert any(digest(changed[key]) != digest(blobs[key]) for key in PATHS)
        environment = sources["show-environment"] + OVERRIDE + b"=/tmp/other\n"
        (staging / "show-environment").write_bytes(environment)
        assert OVERRIDE in invoke(staging, "show-environment")
        assert OVERRIDE in b"# " + OVERRIDE + b" used to select a test directory\n"
        # An ordinary config edit changes the rendered intent without changing
        # the installed crontab or either unit. No C1 hash sees that edit.
        assert enter != research and invoke(staging, "cron") == enter
        print(f"{cron_count} altered cron outputs refused; "
              f"{len(cat_cases) + len(show_cases)} altered unit outputs refused; "
              "4 blob changes differ; 2 intent renders; stable outputs pass")


if __name__ == "__main__":
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise SystemExit("run this probe at nice 19")
    if sys.argv[1:2] == ["shim"]:
        raise SystemExit(shim(sys.argv[2:]))
    main()
