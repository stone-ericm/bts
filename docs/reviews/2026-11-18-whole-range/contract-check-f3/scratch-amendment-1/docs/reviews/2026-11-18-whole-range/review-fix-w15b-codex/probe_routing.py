import contextlib
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
import probe_health


class StderrProxy:
    encoding = "utf-8"

    def __init__(self, stream):
        self.stream = stream

    def fileno(self):
        return self.stream.fileno()

    def flush(self):
        self.stream.flush()


def stderr_proxy(source, home):
    target = home / "captured-utf8"
    with target.open("wb") as stream:
        with contextlib.redirect_stderr(StderrProxy(stream)):
            probe_health.producer.warn("unchanged ASCII wording")
    actual = target.read_bytes()
    if source == probe_health.HERE / "r2/src":
        assert actual == b"WARNING: unchanged ASCII wording\n"
    else:
        assert actual == b""
    print("UTF-8 descriptor proxy without an errors attribute", repr(actual))


def custom_routing(home):
    producer_root = home / "producer"
    health_root = home / "health"
    producer_root.mkdir()
    health_root.mkdir()
    os.chdir(producer_root)
    cache, raw = probe_health.CACHE, Path("data/raw")
    cache.parent.mkdir(parents=True)
    cache.write_text('{"100":null}')
    feed = raw / "2026/100.json"
    feed.parent.mkdir(parents=True)
    feed.write_text("{}")
    probe_health.Clock.at = datetime.fromisoformat(probe_health.AT)
    with patch.object(probe_health.producer, "datetime", probe_health.Clock):
        with contextlib.redirect_stderr(io.StringIO()):
            probe_health.producer.build(cache, raw)
    evidence, = (cache.parent / "rejected").iterdir()
    os.chdir(health_root)
    with probe_health.captured() as (packets, rejected, logs):
        result, = probe_health.run(picks=producer_root / probe_health.PICKS)
        assert len(packets) == 1 and not rejected
        assert str(evidence) in logs.getvalue() and str(producer_root) not in logs.getvalue()
        assert not evidence.exists() and (producer_root / evidence).exists()
        assert "beside the picks directory" in result.message
    print("Absolute custom picks can read the record but evidence location is relative to the producer cwd", evidence)


def main():
    source = Path(sys.argv[1]).resolve()
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    start = Path.cwd()
    with TemporaryDirectory(dir=probe_health.HERE, prefix="routing-") as work:
        try:
            home = Path(work)
            stderr_proxy(source, home)
            if source != probe_health.HERE / "r2/src":
                custom_routing(home)
        finally:
            os.chdir(start)


if __name__ == "__main__":
    main()
