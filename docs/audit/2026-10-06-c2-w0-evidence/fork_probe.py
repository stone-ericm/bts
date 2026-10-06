"""Probe: does a sandbox profile denying process-fork (SIGKILL / EPERM) stop every way Python creates a process?"""
import os, subprocess, sys, tempfile, textwrap
PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
allow = tempfile.mkdtemp()
real = os.path.realpath(allow)
DEV = '(literal "/dev/null") (literal "/dev/dtracehelper")'
def prof(kill):
    act = "(with send-signal SIGKILL)" if kill else ""
    return (f'(version 1)(allow default)(deny file-write* {act})(allow file-write* (subpath "{real}") {DEV})'
            f'(deny process-fork {act})')
SPAWNERS = {
  "none": "pass",
  "import_cli": "import bts.cli, bts.watchdog.cli, bts.dm",
  "subprocess_run": "import subprocess, sys\nr = subprocess.run([sys.executable, '-c', 'print(1)'], check=False)\nprint('PARENT-CONTINUES', r.returncode)",
  "os_fork": "import os\npid = os.fork()\nif pid == 0:\n    os._exit(0)\nos.waitpid(pid, 0)\nprint('PARENT-CONTINUES')",
  "posix_spawn": "import os, sys\npid = os.posix_spawn(sys.executable, [sys.executable, '-c', 'print(1)'], os.environ)\nprint('PARENT-CONTINUES', os.waitpid(pid, 0))",
  "os_system": "import os\nprint('PARENT-CONTINUES', os.system('true'))",
  "mp_spawn": "import multiprocessing as mp\nif __name__ == '__main__':\n    ctx = mp.get_context('spawn'); p = ctx.Process(target=print, args=(1,)); p.start(); p.join(); print('PARENT-CONTINUES', p.exitcode)",
  "mp_fork": "import multiprocessing as mp\nctx = mp.get_context('fork'); p = ctx.Process(target=print, args=(1,)); p.start(); p.join(); print('PARENT-CONTINUES', p.exitcode)",
  "popen_shell": "import subprocess\nprint('PARENT-CONTINUES', subprocess.call('true', shell=True))",
}
env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONPATH": f"{REPO}/src:{REPO}", "TZ": "America/New_York"}
for kill in (True, False):
    for name, body in SPAWNERS.items():
        code = "print('CHILD-STARTED', flush=True)\n" + "try:\n" + textwrap.indent(body, "    ") + \
               "\nexcept PermissionError as e:\n    print('EPERM', e, flush=True)\nexcept OSError as e:\n    print('OSERROR', e.errno, e, flush=True)\nprint('END', flush=True)"
        r = subprocess.run(["/usr/bin/sandbox-exec", "-p", prof(kill), PY, "-B", "-c", code], env=env, cwd=allow,
                           capture_output=True, text=True, timeout=120)
        out = " | ".join(l for l in r.stdout.splitlines() if l)[:160]
        err = (r.stderr.strip().splitlines() or [""])[-1][:120]
        print(f"kill={kill!s:5} {name:15} rc={r.returncode:4} out=[{out}] err=[{err}]")
