import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import review_mutations as prior


PRODUCER = prior.PRODUCER
HEALTH = prior.HEALTH
READER = prior.READER
INCIDENTS = prior.INCIDENTS
HEALTH_TESTS = prior.HEALTH_TESTS
prior.MUTANTS = HERE / 'mutants'


def restore_fd_leak(text):
    start = text.index('    fd = os.open(path, flags)')
    end = text.index('\n\n\ndef directory_exists', start)
    return text[:start] + (
        '    with os.fdopen(os.open(path, flags), "rb") as stream:\n'
        '        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):\n'
        '            raise NonRegularFile(path)\n'
        '        yield stream\n'
    ) + text[end:]


def restore_buffered_stderr(text):
    start = text.index('def warn(message):')
    end = text.index('\n\n\ndef decode', start)
    return text[:start] + (
        'def warn(message):\n'
        '    try:\n'
        '        print(f"WARNING: {message}", file=sys.stderr)\n'
        '    except Exception:\n'
        '        pass\n'
    ) + text[end:]


prior.MUTATIONS = [
    (name, filename, transform, tests) for name, filename, transform, tests in prior.MUTATIONS
    if name != 'key-by-reason'
] + [
    ('key-by-reason', HEALTH,
     ('json.dumps(sorted(identities), separators=(",", ":"))', 'json.dumps(sorted(kinds), separators=(",", ":"))'),
     [f'{HEALTH_TESTS}::test_same_day_changed_incident_remains_visible']),
    ('skip-existing-record-comparison', PRODUCER,
     ('    existing = read_record(path)\n'
      '    if any(existing[key] != value for key, value in row.items() if key != "recorded_at"):\n'
      '        raise FileExistsError(path)', '    read_record(path)'),
     [f'{INCIDENTS}::test_publish_rejects_valid_conflicting_record_at_same_destination']),
    ('raw-root-errno-as-absent', PRODUCER,
     ('except OSError:\n        return scanned, rejected, True', 'except OSError:\n        return scanned, rejected, False'),
     [f'{INCIDENTS}::test_inaccessible_raw_root_marks_unverified_feeds_unusable']),
    ('skip-size-check', PRODUCER,
     ('    if len(payload) > MAX_BYTES:\n        raise ValueError("status exceeds size limit")\n', ''),
     [f'{INCIDENTS}::test_publish_rejects_oversize_serialization_before_writing']),
    ('restore-directory-fd-leak', READER, restore_fd_leak,
     [f'{INCIDENTS}::test_directory_feed_candidates_keep_descriptors_and_bullpen_value']),
    ('restore-buffered-stderr', PRODUCER, restore_buffered_stderr,
     [f'{INCIDENTS}::test_feature_child_exits_zero_with_closed_stderr_reader']),
]


if __name__ == '__main__':
    prior.main()
