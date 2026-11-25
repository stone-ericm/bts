#!/usr/bin/env bash
# Run one bounded process per scratch change. The Python runner restores the copy
# after each individual suite.
set -euo pipefail

for variant in \
    top-prediction below-on-equality drop-unverified-label swap-K1-K5 \
    old-K3-journal-sentence unescaped-banner-name drop-delivery-saver \
    old-late-refusal string-render-raises
do
    timeout 110s nice -n 19 env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH=.codex-review/wr-e .venv/bin/python -c '
import sys
import run_review

name = sys.argv[1]
selected = {name: run_review.variations()[name]}
run_review.variations = lambda: selected
sys.argv[1:] = ["mutations"]
raise SystemExit(run_review.main())
' "$variant"
done
