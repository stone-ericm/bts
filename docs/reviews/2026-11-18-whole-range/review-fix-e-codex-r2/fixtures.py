"""Load the checked-in synthetic fixtures against this checkout."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVIEWS = ROOT / "docs/reviews/2026-11-18-whole-range"
sys.path[:0] = [str(ROOT), str(REVIEWS / "contract-check-e")]


def fixture(name, directory="review-fix-e-codex"):
    spec = importlib.util.spec_from_file_location(name, REVIEWS / directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    if directory == "review-fix-e-codex":
        module.__file__ = str(HERE / (name + ".py"))
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.HERE, module.ROOT = HERE, ROOT
    return module
