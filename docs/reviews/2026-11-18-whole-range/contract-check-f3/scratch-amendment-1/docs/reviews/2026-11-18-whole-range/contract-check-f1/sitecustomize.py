"""Deny real IP transports for offline probes and tests."""
import socket


def deny_network(*args, **kwargs):
    raise AssertionError("offline contract check: IP transport attempted")


def guarded(method):
    def call(sock, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            deny_network()
        return method(sock, *args, **kwargs)
    return call


for name in ("connect", "connect_ex", "sendto", "sendmsg"):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, guarded(getattr(socket.socket, name)))
socket.getaddrinfo = deny_network
socket.gethostbyname = deny_network
socket.gethostbyname_ex = deny_network
