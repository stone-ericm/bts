"""Install the offline guard in each review process, including pytest children."""

import ipaddress
import atexit
import sys


blocked = []


def loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def offline(event, args):
    if event in {"socket.gethostbyaddr", "socket.gethostbyname", "socket.getnameinfo"}:
        address = args[0]
    elif event == "socket.getaddrinfo" and args[0] is not None and not loopback(args[0]):
        address = args[0]
    elif event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
        address = args[1]
        if isinstance(address, tuple) and loopback(address[0]):
            return
        if address is None and event == "socket.sendmsg":
            return
    else:
        return
    blocked.append((event, address))
    raise AssertionError(f"Review blocked {event} to {address!r}")


def verify_offline():
    if blocked:
        print(f"Blocked network attempts: {blocked!r}", file=sys.stderr)


sys.addaudithook(offline)
atexit.register(verify_offline)
