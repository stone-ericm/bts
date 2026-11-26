"""Show whether the fallback evaluates the saver read after display diagnostics."""
from pathlib import Path
from tempfile import TemporaryDirectory

from probe_support import NOW, REVIEW, REV, auto, dated, invoke, offline, put, switch
from bts.contest_state import load_decision_streak_state

with TemporaryDirectory(dir=REVIEW, prefix="saver-value-") as directory:
    picks = Path(directory)
    put(picks, "season_contract.json", switch())
    put(picks, "contest_streak.json", auto())
    manual = put(picks, "contest_streak.manual.json", dated())
    put(picks, "saver_state.json", b'{"season":' + b"9" * 5000 + b',"state":"used"}')
    with offline():
        result = invoke(picks, "season-contract", "status")
    assert result.exit_code == 0, result.exception
    assert "live decision now: accepted" in result.output
    assert "saver: unavailable (invalid: Exceeds the limit" in result.output, result.output
    expected = "accepted" if REV == "no-saver" else "refused"
    automatic, = [line for line in result.output.splitlines() if "automatic path alone:" in line]
    assert f"alone: {expected}" in automatic, automatic
    manual.unlink()
    try:
        load_decision_streak_state(picks, now=NOW, require_contest_state=True)
    except ValueError as exc:
        assert "Exceeds the limit" in str(exc)
        print(f"{REV}: {automatic} | live raises {type(exc).__name__}")
    else:
        raise AssertionError("the ordinary live loader skipped the saver read")
