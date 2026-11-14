#!/usr/bin/env python3
"""Scan src/bts/, scripts/cron-setup-hetzner.sh, and deploy/ for season-bound literals."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(".")
TARGETS = (
    sorted(Path("src/bts").rglob("*.py"))
    + [Path("scripts/cron-setup-hetzner.sh")]
    + sorted(p for p in Path("deploy").rglob("*") if p.is_file())
)

PATTERNS = [
    ("YEAR_25_27", re.compile(r"\b202[567]\b")),
    ("OTHER_YEAR", re.compile(r"\b20(?:1[5-9]|2[0-489]|3\d)\b")),
    ("MMDD", re.compile(r"(?:\b0[1-9]|\b1[0-2])-(?:0[1-9]|[12]\d|3[01])\b")),
    (
        "SEASON_WORD",
        re.compile(
            r"\b(?:SEASON_END|OFFSEASON|MAX_TAIL_DAYS|late_phase_days|season_days"
            r"|days_remaining|season_length|180|186)\b"
        ),
    ),
]

for p in TARGETS:
    text = p.read_text(errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        tags = [name for name, rx in PATTERNS if rx.search(line)]
        if tags:
            tag_str = ",".join(tags)
            print(f"{p}:{lineno} [{tag_str}]: {line.strip()[:140]}")
