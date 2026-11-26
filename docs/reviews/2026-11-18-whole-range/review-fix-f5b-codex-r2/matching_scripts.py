"""Repeat the archived installer, backup and offline audit script checks."""

from runpy import run_path

from review_support import ROOT

archived = ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex/readers_probe.py'
run_path(str(archived), run_name='__main__')
