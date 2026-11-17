#!/bin/zsh
set -u
export TZ=America/New_York PYTHONDONTWRITEBYTECODE=1
root=$PWD
export PYTHONPATH="$root/.codex-review/cs3-review:$root/src:$root"
export REVIEW_SOURCE="$root/src"
for year in 2027 2028; do
  REVIEW_YEAR=$year .venv/bin/python -m pytest -p no:cacheprovider -p review_clock tests/test_saver_cs3.py tests/test_decision_saver_fallback.py tests/health/test_contest_state.py tests/test_web_saver.py tests/test_cli_integration.py tests/test_contest_fetch.py -q > ".codex-review/cs3-review/clock-$year.log" 2>&1
  echo "clock $year exit=$?"
  tail -3 ".codex-review/cs3-review/clock-$year.log"
done
export PYTHONPATH="$root/.codex-review/cs3-review:$root/.codex-review/cs3-review/base/src:$root"
export REVIEW_SOURCE="$root/.codex-review/cs3-review/base/src"
REVIEW_YEAR=2027 .venv/bin/python -m pytest -p no:cacheprovider -p review_clock tests/test_saver_cs3.py -q > .codex-review/cs3-review/base-new-tests.log 2>&1
echo "base new tests exit=$?"
tail -3 .codex-review/cs3-review/base-new-tests.log
export PYTHONPATH="$root/.codex-review/cs3-review:$root/src:$root"
export REVIEW_SOURCE="$root/src"
REVIEW_YEAR=2027 .venv/bin/python -m pytest -p no:cacheprovider -p review_clock tests/test_saver_state.py tests/health/test_attention.py tests/health/test_alert.py tests/health/test_runner.py -q > .codex-review/cs3-review/unchanged-consumers.log 2>&1
echo "unchanged consumers exit=$?"
tail -3 .codex-review/cs3-review/unchanged-consumers.log
