import hashlib, io, resource, sys, time
from pathlib import Path
import pandas as pd, pyarrow as pa
mode, d = sys.argv[1], Path(sys.argv[2])
files = sorted(d.glob("pa_*.parquet"))
t0 = time.perf_counter()
dfs = []
for p in files:
    if mode == "base":
        dfs.append(pd.read_parquet(p))
    else:
        raw = p.read_bytes(); hashlib.sha256(raw).hexdigest()
        dfs.append(pd.read_parquet(pa.BufferReader(raw) if mode == "native" else io.BytesIO(raw)))
df = pd.concat(dfs, ignore_index=True)
print(mode, round(time.perf_counter() - t0, 2), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10**6)
if mode != "base":
    ref = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    pd.testing.assert_frame_equal(df, ref); print("  equal to path parse")
