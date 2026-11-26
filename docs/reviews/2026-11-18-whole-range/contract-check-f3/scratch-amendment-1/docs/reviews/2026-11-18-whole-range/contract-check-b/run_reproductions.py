"""Supply the slice harness's offline legacy-writer archive in the review directory."""

import io
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import reproduce
import opening_rehearsal
import offline


with TemporaryDirectory(prefix="legacy-writer-", dir=reproduce.HERE) as folder:
    root = Path(folder)
    archive = subprocess.check_output(["git", "archive", "619abda7", "src"], cwd=reproduce.HERE)
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        contents.extractall(root / "base", filter="data")
    with patch.object(opening_rehearsal, "HERE", root), patch.object(offline, "HERE", reproduce.HERE):
        reproduce.main()
