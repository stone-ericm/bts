import sys


def refuse_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise OSError(f'contract check: offline ({event})')


sys.addaudithook(refuse_network)
