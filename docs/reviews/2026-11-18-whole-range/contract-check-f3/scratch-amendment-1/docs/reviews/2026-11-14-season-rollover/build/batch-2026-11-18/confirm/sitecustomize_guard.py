"""Keep all Python review processes, including test children, offline."""
import ipaddress
import socket
import sys


def loopback(host):
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
        return ip.is_loopback or bool(getattr(ip, "ipv4_mapped", None) and ip.ipv4_mapped.is_loopback)
    except (AttributeError, ValueError):
        return False


def audit(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                 "socket.gethostbyaddr", "socket.getnameinfo"}:
        address = args[0]
        host = address[0] if isinstance(address, tuple) else address
        if event == "socket.getaddrinfo" and loopback(host):
            return
        raise AssertionError(f"DNS forbidden: {event} {host!r}")
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
        sock, address = args[:2]
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            if not isinstance(address, tuple) or not loopback(address[0]):
                raise AssertionError(f"External network forbidden: {event} {address!r}")


def reverse_lookup(host):
    if loopback(host):
        return "localhost", [], [host]
    raise AssertionError(f"DNS forbidden: reverse lookup {host!r}")


original_getaddrinfo = socket.getaddrinfo


def numeric_address(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, family, type, proto, flags | socket.AI_NUMERICHOST)


socket.gethostbyaddr = reverse_lookup
socket.getaddrinfo = numeric_address
sys.addaudithook(audit)
