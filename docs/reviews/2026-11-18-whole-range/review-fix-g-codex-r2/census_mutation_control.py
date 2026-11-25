"""Run the independent Gate B control against the surviving scratch change."""
from pathlib import Path
import runpy

import scripts.audit

here = Path(__file__).resolve().parent
target = here / "scratch/census_unguarded/scripts/audit"
scripts.audit.__path__.insert(0, str(target))
from scripts.audit import boundary_shadow_census

assert Path(boundary_shadow_census.__file__).is_relative_to(target)
print(boundary_shadow_census.__file__, flush=True)
runpy.run_path(str(here / "audit_controls.py"), run_name="__main__")
