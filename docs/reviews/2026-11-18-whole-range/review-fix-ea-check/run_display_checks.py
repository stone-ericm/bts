"""Read the closing Markdown pipe explicitly, then run the skip probe."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe_display as probe
from scripts.audit.season_walk import walk


original = Path.read_text


def read_with_trailing_cell_space(path, *args, **kwargs):
    contents = original(path, *args, **kwargs)
    if path.name == "kickoff-fix-ea.md":
        contents = "\n".join(line + " " if line.endswith(" |") else line for line in contents.split("\n"))
    return contents


with patch.object(Path, "read_text", read_with_trailing_cell_space):
    probe.b8a()
legacy = probe.load_legacy()
with TemporaryDirectory(dir=probe.WORK) as temp, walk.offline(probe.NOW):
    probe.skip_checks(Path(temp), legacy)
