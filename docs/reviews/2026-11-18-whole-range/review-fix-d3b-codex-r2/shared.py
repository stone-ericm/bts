"""Use the existing shim harness with all new scratch paths under r2."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import review_box

review_box.HERE = HERE
from review_box import (ROOT, DEPLOY, RECOVER, OLD, TARGET, SIGNALS, box, cgroups, contained,
                        group, inject_shim, invoke, posts, record, replace, tests)
