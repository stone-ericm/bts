from pathlib import Path


runner = Path(__file__).with_name('run_tests.py')
source = runner.read_text().replace('os.environ.update(GIT_DIR=git_dir, GIT_WORK_TREE=str(root), PYTHONDONTWRITEBYTECODE=\'1\')',
                                    'os.environ[\'PYTHONDONTWRITEBYTECODE\'] = \'1\'')
exec(compile(source, str(runner), 'exec'), {'__file__': str(runner), '__name__': '__main__'})
