from pathlib import Path
from tempfile import TemporaryDirectory

from bts.picks import save_pick
from review_support import ROOT, World, assert_source, health


assert_source(ROOT / 'src')
with TemporaryDirectory() as dirname:
    world = World(Path(dirname))
    pick = world.commit_at('06:10', at='04:55')
    result = world.cron('05:00', dm='failed')
    assert result['receipt']['marker_status'] == 'dm_failed'
    assert health(world)[0][0] == 'CRITICAL'
    pick.pick.game_pk = 102
    save_pick(pick, world.picks)
    assert world.early_state()['slots'][0]['game_pk'] == 101
    assert health(world) == []
    print('the failed warning for game 101 does not page after the recorded selection changes to game 102')
