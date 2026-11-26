"""Use the round-one helpers with round-two artifact paths."""

from pathlib import Path

checkout = Path(__file__).resolve().parents[2]
archived = checkout / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex/review_support.py'
exec(compile(archived.read_bytes(), str(archived), 'exec'))
