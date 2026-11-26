"""Check that failures in superseded Carbon drafts already fail on round one."""

import os
from pathlib import Path
import subprocess
import sys

from replay_round1 import CARBON, REPLAY, REVIEW, ROOT


def main():
    checkout = REVIEW / "round1-source"
    checkout.mkdir(exist_ok=True)
    archive = subprocess.run(["git", "archive", "dfe774df", "src"], cwd=ROOT,
                             capture_output=True, check=True, timeout=10)
    subprocess.run(["tar", "-x", "-C", str(checkout)], input=archive.stdout,
                   check=True, timeout=10)

    original = REPLAY / "carbon"
    destination = REPLAY / "carbon-round1"
    destination.mkdir(exist_ok=True)
    prefix = os.pathsep.join(map(str, (checkout / "src", ROOT)))
    env = os.environ | {"PYTHONPATH": prefix, "TZ": "America/New_York",
                        "PYTHONDONTWRITEBYTECODE": "1"}
    for source in sorted(original.glob("probe_*.py")):
        if source.name in CARBON:
            continue
        adapted = source.read_text().replace('ROOT / "src"', 'HERE.parents[1] / "round1-source" / "src"')
        script = destination / source.name
        script.write_text(adapted)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=45)
        output = result.stdout + result.stderr
        (REVIEW / f"replay-carbon-{source.stem}-round1-head.log").write_text(output)
        stock = (REVIEW / f"replay-carbon-{source.stem}-head.log").read_text()
        signature = output.strip().splitlines()[-1].replace(str(destination), str(original))
        assert result.returncode == 1 and signature == stock.strip().splitlines()[-1], source.name
        print(f"{source.name}: same exit and exception on dfe774df and e897c766", flush=True)


if __name__ == "__main__":
    main()
