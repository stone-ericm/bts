"""Rerun the archived standing rows without loading their archived paths as fixtures."""
import runpy

from support import ARCHIVED_REVIEW

runpy.run_path(str(ARCHIVED_REVIEW / 'standing.py'), run_name='__main__')
