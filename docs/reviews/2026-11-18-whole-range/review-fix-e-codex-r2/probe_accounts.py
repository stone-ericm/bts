"""Offline exercise of the Mac capture, box cookie lookup and copied-picks drill."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from fixtures import HERE, ROOT
from click.testing import CliRunner
from bts import dm, scheduler
from bts.cli import cli
from bts.leaderboard import auth


def capture(root):
    uv = shutil.which("uv")
    assert uv
    result = subprocess.run(
        [uv, "--offline", "--no-cache", "run", "--no-sync", "python",
         "scripts/capture_bts_cookies.py", "--help"],
        cwd=ROOT, text=True, capture_output=True, timeout=6,
    )
    print("clean synced dependencies:", result.returncode, result.stderr.strip())
    assert result.returncode == 1 and "No module named 'playwright'" in result.stderr

    package = ModuleType("playwright")
    package.sync_api = ModuleType("playwright.sync_api")
    play = package.sync_api.sync_playwright = MagicMock()
    with patch.dict(sys.modules, {"playwright": package, "playwright.sync_api": package.sync_api}):
        from scripts import capture_bts_cookies as script

    browser = play.return_value.__enter__.return_value.chromium.launch.return_value
    fresh = [{"name": "synthetic", "value": "fresh"}]
    browser.new_context.return_value.cookies.return_value = fresh
    for flags in ([], ["--print"]):
        stdout = io.StringIO()
        with patch.object(sys, "platform", "darwin"), patch.object(sys, "argv", ["capture", *flags]), \
             patch("builtins.input", return_value=""), patch.object(subprocess, "run") as keychain, \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            script.main()
            assert keychain.call_count == 2
        assert (json.loads(stdout.getvalue()) == fresh) if flags else not stdout.getvalue()
        print("capture", flags, "stdout bytes", len(stdout.getvalue()), "Keychain actions", keychain.call_count)

    default = root / ".bts-leaderboard-cookies.json"
    default.write_text(json.dumps(fresh))
    other = root / "configured.json"
    old_raw = json.dumps([{"name": "synthetic", "value": "expired"}])
    other.write_text(old_raw)
    with patch.object(sys, "platform", "linux"), patch.dict(os.environ, {"HOME": str(root)}, clear=True):
        with patch.object(subprocess, "check_output", return_value=old_raw.encode()):
            assert auth.load_session_cookies()["synthetic"] == "expired"
            print("fresh default file, usable pass entry -> loaded expired pass entry")
        with patch.object(subprocess, "check_output", side_effect=FileNotFoundError):
            assert auth.load_session_cookies()["synthetic"] == "fresh"
            with patch.dict(os.environ, {"BTS_LEADERBOARD_COOKIE_FILE": str(other)}):
                assert auth.load_session_cookies()["synthetic"] == "expired"
                print("fresh default file, configured override -> loaded expired override")


def drill(root):
    dir_live = root / "data/picks"
    day = "2027-07-15"
    decision = dir_live / day / "decision.json"
    decision.parent.mkdir(parents=True)
    decision.write_text('{"source":"synthetic"}')
    copied = root / "drill/picks"
    copied.parent.mkdir()
    subprocess.run(["/bin/cp", "-a", str(dir_live), str(copied)], timeout=3, check=True)
    (copied / day / decision.name).unlink()
    config = root / ".bts-orchestrator.toml"
    config.write_text('[scheduler]\npick_delivery="dm"\nentry_intent="enter"\n')
    (root / ".env").write_text("BTS_BLUESKY_APP_PASSWORD=synthetic-drill\n")
    shell = subprocess.run(["/bin/bash", "-c", "set -a && . ./.env && set +a && env"],
                           env={}, cwd=root, timeout=3, check=True, capture_output=True, text=True)
    exported = dict(line.split("=", 1) for line in shell.stdout.splitlines())
    replies = [
        {"accessJwt": "offline"}, {"did": "did:fixture"},
        {"convo": {"id": "fixture"}}, {"id": "receipt"},
    ]
    args = ["day-outcome", "--date", day, "--config", str(config), "--picks-dir", str(copied),
            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social"]
    with patch.object(scheduler, "fetch_schedule", return_value=[{"gamePk": 1, "status": {"detailedState": "Final"}}]), \
         patch.object(subprocess, "run", side_effect=FileNotFoundError), \
         patch.object(dm, "retry_urlopen", side_effect=[io.BytesIO(json.dumps(p).encode()) for p in replies]) as transport:
        with patch.dict(os.environ, {}, clear=True):
            old = CliRunner().invoke(cli, args)
        assert old.exit_code == 1 and "DM failed: RuntimeError" in old.output and transport.call_count == 0
        with patch.dict(os.environ, exported, clear=True):
            current = CliRunner().invoke(cli, args)
        assert current.exit_code == 1 and "no_decision" in current.output and transport.call_count == 4
        assert b"BTS day check" in transport.call_args.args[0].data
    assert decision.read_text() == '{"source":"synthetic"}'
    print("B4", old.output, current.output, "DRILL DM captured; live decision bytes retained.")


with tempfile.TemporaryDirectory(dir=HERE) as name:
    {"capture": capture, "drill": drill}[sys.argv[1]](Path(name))
