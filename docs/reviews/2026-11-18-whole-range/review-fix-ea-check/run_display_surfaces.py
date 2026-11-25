"""Keep the real legacy bar when predicting BASE summary text."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe_display as probe
from scripts.audit.season_walk import walk


real_render = probe.render_loaded
fallback = ". No probability comparison is available for this check."


def render_with_legacy_bar(path, summary, expected, journal=None):
    if fallback in expected and "bar" in summary:
        streak = summary.get("streak")
        bar = f"the streak-{streak} bar" if type(streak) is int and streak >= 0 else "the bar"
        assert summary["best_p"] < summary["bar"]
        expected = expected.replace(fallback, f", below {bar} ({summary['bar']:.1%}) using unrounded values.")
    real_render(path, summary, expected, journal)


legacy = probe.load_legacy()
with TemporaryDirectory(dir=probe.WORK) as temp, walk.offline(probe.NOW), \
        patch.object(probe, "render_loaded", render_with_legacy_bar):
    probe.skip_checks(Path(temp), legacy)
