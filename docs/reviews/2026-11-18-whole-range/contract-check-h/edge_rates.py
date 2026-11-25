from pathlib import Path
import runpy

from bts.simulate import tail_policy


original = tail_policy.save_tail_policy


def save_for_probe(artifact, path, *, expected_base_sha):
    tail_policy.validate_tail_policy(artifact, expected_base_sha=expected_base_sha)
    return original(artifact, path)


tail_policy.save_tail_policy = save_for_probe
runpy.run_path(str(Path(__file__).with_name('arithmetic.py')), run_name='__main__')
