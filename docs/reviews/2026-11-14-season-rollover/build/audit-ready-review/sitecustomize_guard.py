import ipaddress
import socket
import sys


def loopback(host):
    try:
        addr = ipaddress.ip_address(host.split('%', 1)[0])
    except (AttributeError, ValueError):
        return False
    return addr.is_loopback or bool(getattr(addr, 'ipv4_mapped', None) and addr.ipv4_mapped.is_loopback)


def audit(event, args):
    if event in {'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyname_ex',
                 'socket.gethostbyaddr', 'socket.getnameinfo'}:
        host = args[0][0] if isinstance(args[0], tuple) else args[0]
        if not loopback(host) or event in {'socket.gethostbyaddr', 'socket.getnameinfo'}:
            raise AssertionError(f'DNS blocked: {event} {host!r}')
    if event in {'socket.connect', 'socket.sendto', 'socket.sendmsg', 'socket.bind'}:
        sock, address = args[:2]
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            if not isinstance(address, tuple) or not loopback(address[0]):
                raise AssertionError(f'Outward socket blocked: {event} {address!r}')


def gethostbyaddr(host):
    if not loopback(host):
        raise AssertionError(f'DNS blocked: reverse lookup of {host!r}')
    return 'localhost', [], [host]


original_getaddrinfo = socket.getaddrinfo


def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if not loopback(host):
        raise AssertionError(f'DNS blocked: lookup of {host!r}')
    return original_getaddrinfo(host, port, family, type, proto, flags | socket.AI_NUMERICHOST)


sys.addaudithook(audit)
socket.gethostbyaddr = gethostbyaddr
socket.getaddrinfo = getaddrinfo


def verify():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with socket.create_connection(listener.getsockname()):
            pass
        for call in (lambda: socket.getaddrinfo('example.invalid', 80),
                     lambda: listener.connect(('192.0.2.1', 80))):
            try:
                call()
            except AssertionError:
                continue
            raise AssertionError('network guard did not prevent operation')
