import os
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import bts

import probe_health


def main():
    source = Path(sys.argv[1]).resolve()
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    assert source == probe_health.HERE / "r2/src"
    capture = probe_health.captured

    @contextmanager
    def observe(*args, **kwargs):
        with capture(*args, **kwargs) as result:
            try:
                yield result
            finally:
                sent, blocked, _ = result
                print("Round 2 journal replay transport", "accepted", list(map(len, sent)),
                      "rejected", list(map(len, blocked)))
                for body in sent:
                    if "scheduler death evidence:" in body:
                        print(body)

    probe_health.captured = observe
    initial = Path.cwd()
    with TemporaryDirectory(dir=probe_health.HERE, prefix="prior-followup-") as work:
        os.chdir(work)
        try:
            probe_health.policy_and_bound(True)
            try:
                probe_health.journal_feedback(True)
            except AssertionError as error:
                assert Path("replay-receipt.json").exists()
                import json
                receipt = json.loads(Path("replay-receipt.json").read_bytes())
                assert receipt["status"] == "sent" and receipt["critical_count"] == 1, receipt
                print("Round 2 asserted its false journal CRITICAL and sent it below the cap;", repr(error))
        finally:
            os.chdir(initial)


if __name__ == "__main__":
    main()
