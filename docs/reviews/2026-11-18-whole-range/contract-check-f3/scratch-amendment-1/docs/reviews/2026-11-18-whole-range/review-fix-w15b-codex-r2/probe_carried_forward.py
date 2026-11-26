"""Rerun the earlier feature and restart checks on the new HEAD in the new review folder."""
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import probe_features
import probe_restarts


source = Path(sys.argv[1]).resolve()
probe_features.HERE = HERE
probe_restarts.HERE = HERE
probe_features.main()
probe_restarts.main(source)
