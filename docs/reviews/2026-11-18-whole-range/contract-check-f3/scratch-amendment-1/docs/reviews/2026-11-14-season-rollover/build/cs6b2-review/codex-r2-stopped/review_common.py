import contextlib
import importlib.util
import io
import ipaddress
import logging
import os
import signal
import socket
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = "src/bts/health/alert.py"
ET = ZoneInfo("America/New_York")


def offline():
    """Resolve only numeric loopback and the server's own name, without DNS."""
    ipv4_names = {"localhost", "localhost.", "ip6-localhost", socket.gethostname().lower()}
    actual_getaddrinfo = socket.getaddrinfo
    actual_connect = socket.socket.connect
    actual_connect_ex = socket.socket.connect_ex
    actual_bind = socket.socket.bind
    actual_sendto = socket.socket.sendto

    def local_host(host):
        if isinstance(host, bytes):
            host = host.decode("ascii")
        if isinstance(host, str) and host.lower() in ipv4_names:
            return "127.0.0.1"
        try:
            if ipaddress.ip_address(host).is_loopback:
                return host
        except ValueError:
            pass
        raise AssertionError(f"offline review rejected network address {host!r}")

    def address_for(sock, addr):
        if sock.family == socket.AF_UNIX:
            return addr
        assert sock.family in (socket.AF_INET, socket.AF_INET6), sock.family
        return (local_host(addr[0]), *addr[1:])

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if port is not None:
            assert isinstance(port, int) or (isinstance(port, str) and port.isdecimal())
        return actual_getaddrinfo(local_host(host), port, family, type, proto,
                                  flags | socket.AI_NUMERICHOST | socket.AI_NUMERICSERV)

    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = local_host
    socket.gethostbyname_ex = lambda host: (local_host(host), [], [local_host(host)])
    socket.gethostbyaddr = lambda host: ("localhost", [], [local_host(host)])
    socket.getnameinfo = lambda addr, flags: (
        ("localhost" if not flags & socket.NI_NUMERICHOST else local_host(addr[0])),
        str(addr[1]),
    ) if local_host(addr[0]) else None
    socket.socket.connect = lambda sock, addr: actual_connect(sock, address_for(sock, addr))
    socket.socket.connect_ex = lambda sock, addr: actual_connect_ex(sock, address_for(sock, addr))
    socket.socket.bind = lambda sock, addr: actual_bind(sock, address_for(sock, addr))
    socket.socket.sendto = lambda sock, data, *args: actual_sendto(
        sock, data, *args[:-1], address_for(sock, args[-1]))
    if hasattr(socket.socket, "sendmsg"):
        actual_sendmsg = socket.socket.sendmsg

        def sendmsg(sock, buffers, ancillary=(), flags=0, address=None):
            if address is None:
                return actual_sendmsg(sock, buffers, ancillary, flags)
            return actual_sendmsg(sock, buffers, ancillary, flags, address_for(sock, address))

        socket.socket.sendmsg = sendmsg


def archive(revision, target=None):
    if target is None:
        target = HERE / ("src-" + revision)
    target.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(["git", "archive", revision, "src"], cwd=ROOT,
                          stdout=subprocess.PIPE) as git:
        with tarfile.open(fileobj=git.stdout, mode="r|") as source:
            source.extractall(target, filter="data")
        assert git.wait() == 0
    return target / "src"


def archived_alert(revision):
    archive_root = HERE / ("src-" + revision)
    path = archive_root / SOURCE
    if not path.exists():
        archive(revision, archive_root)
    name = "_review_" + revision
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Deadline(BaseException):
    pass


@contextlib.contextmanager
def bounded(label, seconds=2):
    def timeout(*_):
        raise Deadline(label)

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        self.rows.append((record.levelname, record.getMessage(),
                          record.exc_info[0].__name__ if record.exc_info else None))


@contextlib.contextmanager
def captured(module, *, day=14, fail=False):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            instant = cls(2027, 7, day, 23, 59, 59, tzinfo=ET)
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    calls = []
    records = Records()

    def send(*args):
        calls.append(args)
        if fail:
            raise RuntimeError("offline transport failed — é")

    with patch.object(module, "send_dm", send), patch.object(module, "datetime", Clock):
        level, propagate = module.log.level, module.log.propagate
        module.log.setLevel(logging.INFO)
        module.log.propagate = False
        module.log.addHandler(records)
        try:
            yield calls, records.rows
        finally:
            module.log.removeHandler(records)
            module.log.setLevel(level)
            module.log.propagate = propagate


def issue(module, path, mode, *, day=14, explicit=True):
    critical = module.Alert("CRITICAL", "probe", "critical probe")
    warning = module.Alert("WARN", "probe", "warn probe")
    return module.dispatch_dm_for_health_alerts(
        [critical] if mode in ("critical", "no_recipient") else [],
        None if mode == "no_recipient" else "offline.test",
        warn_attention=[warning] if mode == "warn" else [],
        status_path=path,
        now_et_date=datetime(2027, 7, day).date() if explicit else None,
    )
