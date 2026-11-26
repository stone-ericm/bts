"""Keep tests and their Python subprocesses offline."""

import socket


def no_network(*args, **kwargs):
    raise OSError("K1 check: network disabled")


socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.socket.sendto = no_network
socket.getaddrinfo = no_network
