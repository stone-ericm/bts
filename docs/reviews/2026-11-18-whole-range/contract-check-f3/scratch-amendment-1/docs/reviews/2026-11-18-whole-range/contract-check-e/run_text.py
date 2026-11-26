"""Choose a shipped MDP streak that skips the runner-up and run the text probe."""
import socket
import tempfile
from pathlib import Path

from pytest import MonkeyPatch

import probe_text as probe
from bts import strategy

original_seed = probe.seed_automatic
mdp = strategy._load_mdp()
streak = next(s for s in range(10, 56)
              if strategy._mdp_action_from(mdp, .75, s, probe.DAY, False) == "skip")
with tempfile.TemporaryDirectory() as directory, MonkeyPatch.context() as mp:
    mp.setattr(probe, "seed_automatic", lambda d: original_seed(d, streak=streak))
    mp.setattr(socket.socket, "connect", probe.offline)
    mp.setattr(socket, "create_connection", probe.offline)
    probe.probe(Path(directory), mp)
