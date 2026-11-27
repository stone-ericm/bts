import os
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def reject_network(event, args):
    if event == "socket.getaddrinfo" or (
        event in {"socket.connect", "socket.sendto"}
        and args[0].family in {socket.AF_INET, socket.AF_INET6}
    ):
        raise OSError("K2 review has no network")


def install():
    sys.addaudithook(reject_network)
    sys.path.append(str(ROOT))
    import bts

    print(f"bts.__file__ = {bts.__file__}", file=sys.stderr, flush=True)
    expected = os.environ.get("K2_REVIEW_EXPECT_SRC")
    if expected:
        assert Path(bts.__file__).resolve().is_relative_to(Path(expected).resolve())


if __name__ == "__main__":
    install()
    import pytest

    archived_scripts = os.environ.get("K2_REVIEW_ARCHIVED_SCRIPTS")
    if archived_scripts:
        import scripts

        scripts.__path__.insert(0, archived_scripts)
    sys.exit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *sys.argv[1:]]))
