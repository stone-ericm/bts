"""Offline subset: exclude the test requiring an actual HTTP socket connection."""
import sys
sys.dont_write_bytecode=True
def deny(event,args):
    if event in {'socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.connect','socket.connect_ex','socket.sendto'}:
        raise RuntimeError('audit: network forbidden: '+event)
sys.addaudithook(deny)
import pytest
raise SystemExit(pytest.main(['-p','no:cacheprovider','-q','--tb=short',
    '--basetemp=.codex-review/cs4b-check/pytest-offline-temp',
    '-k','not test_do_post_saver_transition_socket_wiring',
    'tests/test_contest_fetch.py','tests/test_contest_state_cs1_review.py',
    'tests/test_contest_ledger.py','tests/test_backup.py','tests/test_web_saver.py',
    'tests/scripts/season_ledger/test_contest.py','tests/health/test_contest_state.py']))
