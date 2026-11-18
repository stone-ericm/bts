import ipaddress
import socket
import sys


def _loopback(host):
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
        return address.is_loopback or getattr(address, "ipv4_mapped", None) in (
            ipaddress.IPv4Address("127.0.0.1"),
        )
    except (AttributeError, ValueError):
        return False


def _network(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                 "socket.gethostbyaddr", "socket.getnameinfo"}:
        value = args[0]
        host = value[0] if isinstance(value, tuple) else value
        if not _loopback(host) or event in {"socket.gethostbyaddr", "socket.getnameinfo"}:
            raise AssertionError(f"DNS forbidden: {event} {host!r}")
    if event in {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind"}:
        sock, address = args[:2]
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            if isinstance(address, tuple) and _loopback(address[0]):
                return
            raise AssertionError(f"Non-loopback network forbidden: {event} {address!r}")


def install():
    sys.addaudithook(_network)
