from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from parity_probe import schedules
from review_support import ROOT, assert_source, test_entry_intent


assert_source(ROOT / 'src')
with TemporaryDirectory() as dirname:
    root = Path(dirname)
    installer = test_entry_intent.box.__wrapped__(root)
    with patch.object(test_entry_intent, 'box', SimpleNamespace(__wrapped__=lambda _: installer)):
        schedules(root, Path(sys.argv[1]) / 'base')
