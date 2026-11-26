"""Exercise the three non-pytest files matched by the requested search without live I/O."""
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=9, check=False)
    assert result.returncode == 0, (args, result.stderr)
    return result.stdout


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "cron":
        run(["bash", "-n", "scripts/cron-setup-hetzner.sh"])
        print("Cron syntax passed. The real show, install, and remove paths ran with stubs in test_entry_intent.py.")
    elif command == "leaderboard":
        output = run([sys.executable, "scripts/final_leaderboard_grab.py", "--help"])
        assert "--final-round-date" in output
        print(output)
    elif command == "pvr":
        with TemporaryDirectory() as temp:
            csv = Path(temp) / "synthetic_slots.csv"
            first = date(2027, 3, 24)
            rows = [f"{first + timedelta(days=i)},{'miss' if i % 3 == 0 else 'hit'},0.7" for i in range(21)]
            csv.write_text("date,outcome,p\n" + "\n".join(rows) + "\n")
            output = run([sys.executable, "scripts/audit/pvr_threshold_bootstrap.py", str(csv)])
            assert "21 resolved days" in output and "calibrated-null" in output
            print(output)
    else:
        raise ValueError(command)
