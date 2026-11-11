import json
import torch
import pytest
from qrokkun_ai.v24 import core as c
from qrokkun_ai.v24 import run as r


def test_real_update_and_contract(tmp_path):
    torch.set_num_threads(1)
    a = r.parser().parse_args(['train', '--out-dir', str(tmp_path)])
    assert a.updates == 1000 and a.deadline_unix is None and a.episodes_per_update == 16
    r.validate_args(a)
    a.updates = 999
    with pytest.raises(ValueError):
        r.validate_args(a)
    p, _ = c.load_player(c.PLAYER_PATH)
    before = {k: v.clone() for k, v in p.state_dict().items()}
    net = c.Spawner()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    records, data, stats = r.episodes(
        net, p, [2000000, 2000001], 120, 'cpu', r.Guard(), train_rng=torch.Generator().manual_seed(17)
    )
    metrics = r.ppo(net, opt, data, r.Guard(), 2)
    assert metrics['parameter_delta'] > 0 and metrics['grad_norm'] > 0
    assert all(torch.equal(v, before[k]) for k, v in p.state_dict().items())
    assert all(v.grad is None and not v.requires_grad for v in p.parameters())
    assert stats['decisions'] >= 2 and sum(stats['birth_histogram']) == stats['decisions']
    assert all(x['cap_frames'] == 120 for x in records)
    assert data[0][3] < 0
    with pytest.raises(ValueError):
        c.load_spawner(c.COMPARATOR_PATH)
    r.atomic(tmp_path / 'test.pt', dict(model=net.state_dict(), **net.architecture_metadata()))
    loaded, _ = c.load_spawner(tmp_path / 'test.pt')
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in net.state_dict().items())


def test_lifecycle_and_final_lock(tmp_path):
    a = r.parser().parse_args(
        [
            'train',
            '--out-dir',
            str(tmp_path / 'run'),
            '--source-commit',
            '0123456789abcdef0123456789abcdef01234567',
            '--quick',
            '--updates',
            '1',
            '--episodes-per-update',
            '2',
            '--eval-seeds',
            '1',
        ]
    )
    r.main(a)
    out = tmp_path / 'run'
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['source_commit'] == a.source_commit == manifest['config']['source_commit']
    ck = torch.load(out / 'latest.pt', weights_only=False)
    assert ck['completed_updates'] == 1 and ck['attempted_updates'] == 1 and not ck['partial_update']
    assert ck['optimizer']['state'] and ck['architecture'] == c.ARCHITECTURE
    status = json.loads((out / 'status.json').read_text())
    assert status['status'] == 'updates_complete' and status['last_error'] is None
    assert not (out / 'final-claimed.json').exists()
    a.command = 'final'
    with pytest.raises(ValueError, match='1000'):
        r.main(a)
