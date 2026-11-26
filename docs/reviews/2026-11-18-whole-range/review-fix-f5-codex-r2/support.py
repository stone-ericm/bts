"""Bind the captured transports to this review's freshly generated cron lines."""
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-f5-codex"
sys.path[:0] = [str(ROOT), str(LEGACY)]

import bts
import probe_support

DAY = probe_support.DAY
ET = probe_support.ET
Replay = probe_support.Replay
clock_slots = probe_support.clock_slots
no_network = probe_support.no_network


def load_line(version):
    generated = json.loads((HERE / "generated.json").read_text())
    line, = [line for line in generated[f"{version}_enter"] if "bts check-pick-entered" in line]
    return line


probe_support.load_line = load_line


def source(version):
    module = Path(bts.__file__).resolve()
    expected = HERE / "base/src" if version == "base" else ROOT / "src"
    assert module.is_relative_to(expected), (module, expected)
    print("bts.__file__:", module, flush=True)


def legacy(name):
    return importlib.import_module(name)


def outcome(run):
    return run["receipt"]["outcome"]


def messages(run):
    return [text.rsplit(" (", 1)[0] for text in run["sent"]]
