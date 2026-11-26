"""Select an isolated production package before importing the current tests."""
import datetime as dt
import shutil
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
mode = sys.argv[1]
if mode == "r2_scheduler":
    folder = ROOT / "sources" / "head_r2_scheduler"
    if not folder.exists():
        shutil.copytree(ROOT / "sources" / "head", folder)
        shutil.copyfile(ROOT / "sources" / "r2" / "src" / "bts" / "scheduler.py",
                        folder / "src" / "bts" / "scheduler.py")
elif mode in ("base_guard", "base_old_guard"):
    folder = ROOT / "sources" / "base"
else:
    folder = REPO
sys.path.insert(0, str(folder / "src"))
sys.path.insert(1, str(REPO))
import pytest
from bts import cli as c, scheduler as s
print("PRODUCTION", s.__file__, flush=True)
(ROOT / "pytest-tmp").mkdir(exist_ok=True)
args = ["-p", "no:cacheprovider", "-q", "--basetemp", str(ROOT / "pytest-tmp" / mode)]
if mode == "r2_scheduler":
    assert s.__file__ == str(folder / "src" / "bts" / "scheduler.py")
    args += [str(REPO / "tests/test_streak_season_cs2.py"), "-k",
             "test_row6_scheduler_display_reads_scored_season or test_row8_shadow_selection_uses_own_date_season"]
elif mode in ("base_guard", "base_old_guard"):
    target = REPO / "tests/test_streak_season_cs2.py"
    if mode == "base_old_guard":
        text = target.read_text()
        old = 'pytest.raises(TypeError, match="season must be an int or None")'
        assert text.count(old) == 1
        target = ROOT / "test_old_guard.py"
        target.write_text(text.replace(old, 'pytest.raises(TypeError)'))
    args += [str(target), "-k", "test_a_season_argument_that_is_not_an_int_is_a_type_error"]
elif mode in ("clock2028", "clock2028_old_pin", "clock2028_dateless_void"):
    class FutureDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2028, 1, 1, 12, tzinfo=ZoneInfo("America/New_York"))
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    c.datetime = FutureDatetime
    assert c._now_et().year == 2028
    print("UNPINNED CLI CLOCK", c._now_et().isoformat(), flush=True)
    target = REPO / "tests/test_streak_season_cs2.py"
    if mode == "clock2028_old_pin":
        text = target.read_text()
        start = text.index('    # Pin the command\'s clock')
        end = text.index('    result = CliRunner().invoke', start)
        target = ROOT / "test_old_clock.py"
        target.write_text(text[:start]+text[end:])
    elif mode == "clock2028_dateless_void":
        # Mutation checks that the pin doesn't make a dateless display pass.
        import inspect
        callback = c.check_results.callback
        source = inspect.getsource(callback)
        old = 'else load_streak(picks_path, season=date_type.fromisoformat(date).year)'
        assert source.count(old) == 1
        # Click decorators need not be re-executed: replace the function body
        # in a namespace copied from the production module.
        source = source[source.index('def check_results('):]
        command = c.check_results
        exec(compile(source.replace(old, 'else load_streak(picks_path)'), '<dateless-void-mutation>', 'exec'), vars(c))
        command.callback = c.check_results
        c.check_results = command
    args += [str(target), "-k", "test_row6_cli_void_display_reads_scored_season"]
else:
    raise ValueError(mode)
sys.exit(pytest.main(args))
