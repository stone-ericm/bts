"""Keep review Python processes offline, including local HTTP tests."""
import ipaddress
import socket
import sys


def loopback(host):
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except (AttributeError, ValueError):
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or (mapped is not None and mapped.is_loopback)


def audit(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                 "socket.gethostbyaddr", "socket.getnameinfo"}:
        address = args[0]
        if isinstance(address, tuple):
            address = address[0]
        if event != "socket.getaddrinfo" or not loopback(address):
            raise AssertionError(f"DNS forbidden: {event} {address!r}")
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
        sock, address = args[:2]
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            if not isinstance(address, tuple) or not loopback(address[0]):
                raise AssertionError(f"Outward network forbidden: {event} {address!r}")


def reverse_lookup(host):
    if loopback(host):
        return "localhost", [], [host]
    raise AssertionError(f"DNS forbidden: reverse lookup {host!r}")


getaddrinfo = socket.getaddrinfo


def numeric_address(host, port, family=0, type=0, proto=0, flags=0):
    if host == "localhost":
        host = "127.0.0.1"
    return getaddrinfo(host, port, family, type, proto, flags | socket.AI_NUMERICHOST)


socket.gethostbyaddr = reverse_lookup
socket.getaddrinfo = numeric_address
sys.addaudithook(audit)


def check():
    with socket.socket() as sock:
        for attempt in (lambda: socket.getaddrinfo("dns-probe.invalid", 443),
                        lambda: sock.connect(("192.0.2.1", 443))):
            try:
                attempt()
            except AssertionError:
                continue
            raise AssertionError("Offline guard failed")
    assert socket.gethostbyaddr("127.0.0.1") == ("localhost", [], ["127.0.0.1"])
