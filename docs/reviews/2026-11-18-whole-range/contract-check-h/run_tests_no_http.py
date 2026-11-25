from pathlib import Path
import runpy
from urllib.error import HTTPError

from bts import util


def missing(req, **kwargs):
    raise HTTPError(getattr(req, 'full_url', str(req)), 404, 'network disabled for probe', {}, None)


util.urlopen = missing
runpy.run_path(str(Path(__file__).with_name('run_tests.py')), run_name='__main__')
