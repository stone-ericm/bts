"""Give the refusal-key probe an offline pregame status and watchdog."""

from contextlib import nullcontext
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch

from contract_edges import HERE, refusal_keys
from bts import picks, scheduler


def deny_network(*args, **kwargs):
    raise AssertionError('Network is forbidden during a contract probe')


with TemporaryDirectory(dir=HERE) as scratch, \
     patch.object(socket, 'socket', deny_network), \
     patch.object(socket, 'create_connection', deny_network), \
     patch.object(scheduler, 'heartbeat_watchdog', return_value=nullcontext()), \
     patch.object(picks, 'get_game_statuses_detailed', return_value={101: {'abstract': 'P', 'detailed': 'Scheduled'}}):
    refusal_keys(Path(scratch) / 'picks')
