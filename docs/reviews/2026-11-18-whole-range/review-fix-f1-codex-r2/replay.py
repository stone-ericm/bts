"""Replay the archived reviewer script unchanged, using a relocated fixture."""
import runpy

from support import ARCHIVED_REVIEW

runpy.run_path(str(ARCHIVED_REVIEW / 'failed_observation.py'), run_name='__main__')
