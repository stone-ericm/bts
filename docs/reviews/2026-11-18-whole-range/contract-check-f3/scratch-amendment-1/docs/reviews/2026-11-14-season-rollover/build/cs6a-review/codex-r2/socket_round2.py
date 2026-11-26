"""Check the adopted fixture during its module and around real local socket tests."""

import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADOPTED = ROOT / "tests/test_live_gate_cs6a_review.py"
WEB = ROOT / "tests/test_web_saver.py"
SYSTEMD = ROOT / "tests/test_sd_notify.py"


def local_only(event, args):
    if event in {"socket.bind", "socket.connect", "socket.connect_ex", "socket.sendto", "socket.sendmsg"}:
        address = args[1]
        if isinstance(address, tuple) and address[0] == "127.0.0.1":
            return
        if isinstance(address, str):
            path = Path(address)
            if address == "/nonexistent/socket/path.sock":
                return  # Intentional failure in the systemd notification test.
            if path.parent.parent == Path(gettempdir()) and path.parent.name.startswith("sd_"):
                return
    elif event == "socket.getaddrinfo" and args[0] == "127.0.0.1":
        return
    elif event not in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                       "socket.gethostbyaddr", "socket.getnameinfo", "urllib.Request"}:
        return
    raise PermissionError(f"Review forbids external transport: {event} {args!r}")


def listen():
    server = socket.socket()
    try:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(2)
        return server
    except BaseException:
        server.close()
        raise


def verify_connection(blocked):
    import pytest

    with listen() as server, socket.socket() as client:
        client.settimeout(2)
        if blocked:
            with pytest.raises(RuntimeError, match="Network forbidden in test: socket.connect"):
                client.connect(server.getsockname())
        else:
            client.connect(server.getsockname())
            connection, _ = server.accept()
            connection.close()


def child(mode, tmp, adopted):
    sys.addaudithook(local_only)
    import pytest

    class AuditProof:
        during_count = 0
        failures = []
        order = []

        @pytest.hookimpl(tryfirst=True)
        def pytest_runtest_call(self, item):
            self.order.append(item.path.name)
            if item.path == adopted and self.during_count == 0:
                verify_connection(blocked=True)
                self.during_count += 1

        def pytest_runtest_logreport(self, report):
            if report.failed:
                self.failures.append(report)

    proof = AuditProof()
    files = (SYSTEMD, WEB, adopted) if mode == "before" else (adopted, WEB, SYSTEMD)
    # HTTPServer resolves the bound address to a name for its metadata. Keep it numeric.
    with patch.object(socket, "getfqdn", lambda host: host):
        result = pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider",
                              "--basetemp", tmp, *(str(f) for f in files)], plugins=[proof])
    assert proof.during_count == 1, "the live module did not block a real connect as expected"
    assert list(dict.fromkeys(proof.order)) == [file.name for file in files], proof.order
    print("Loopback connect was rejected during the adopted module", flush=True)
    if mode == "no-teardown":
        assert result == pytest.ExitCode.TESTS_FAILED, result
        assert len(proof.failures) == 1, proof.failures
        failure = proof.failures[0]
        assert failure.nodeid.endswith("test_do_post_saver_transition_socket_wiring"), failure.nodeid
        assert "Network forbidden in test: socket.getaddrinfo" in str(failure.longrepr), failure.longrepr
        verify_connection(blocked=True)
        print("Permanent-hook control failed the real HTTP test and still rejects connects", flush=True)
    else:
        assert result == pytest.ExitCode.OK, result
        verify_connection(blocked=False)
        print("Real local connection succeeded after module teardown", flush=True)


def main():
    with TemporaryDirectory(prefix="socket-runs-", dir=HERE) as folder:
        scratch = Path(folder)
        mutant = scratch / "test_live_gate_cs6a_review.py"
        source = ADOPTED.read_text()
        teardown = "    active[0] = False\n"
        assert source.count(teardown) == 1
        mutant.write_text(source.replace(teardown, "", 1))
        for mode in ("after", "before", "no-teardown"):
            print(f"=== {mode} ===", flush=True)
            selected = mutant if mode == "no-teardown" else ADOPTED
            command = [sys.executable, str(__file__), "child", mode, str(scratch / mode), str(selected)]
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        main()
