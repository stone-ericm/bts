"""Re-run the matrix after making its fixture filenames portable on APFS."""
from pathlib import Path


probe = Path(__file__).with_name("probe_text.py")
source = probe.read_text().replace('"code=killed status=9/KILL"', '"code=killed signal=kill"')
source = source.replace("'A' * 12", "'B' * 12")
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
