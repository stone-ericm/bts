"""Run unchanged, relevant main tests through the externally supplied yield wrapper."""
import sys
sys.dont_write_bytecode = True

def deny_network(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                 "socket.connect", "socket.connect_ex", "socket.sendto"}:
        raise RuntimeError("audit: network forbidden: " + event)

sys.addaudithook(deny_network)
from pathlib import Path
sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd())]
import pytest

files = sorted({str(p) for pattern in (
    "tests/test_contest_state*.py", "tests/test_contest_dated*.py", "tests/test_saver*.py",
    "tests/scripts/season_ledger/test_*.py",
) for p in Path.cwd().glob(pattern)})
files += ["tests/test_contest_fetch.py", "tests/test_contest_ledger.py", "tests/test_backup.py",
          "tests/test_web_saver.py", "tests/health/test_contest_state.py",
          "tests/test_cli_integration.py::TestFetchContestStreak"]
raise SystemExit(pytest.main([
    "-p", "no:cacheprovider", "-q", "--tb=short",
    "--basetemp=.codex-review/cs4b1-check/pytest-main",
    "-k", "not test_do_post_saver_transition_socket_wiring", *files,
]))
