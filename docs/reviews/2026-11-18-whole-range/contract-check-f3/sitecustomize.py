import socket
import sys


def block_network(event, args):
    if event == "socket.connect" and args[0].family == socket.AF_UNIX:
        return
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto"}:
        raise AssertionError(f"unexpected network operation: {event}")


sys.addaudithook(block_network)
