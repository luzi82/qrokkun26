from pathlib import Path
import pytest
import torch
from qrokkun_ai.v24 import core, run
from qrokkun_ai.v24.render_demo import load_spawner, rollout

S = Path('artifacts/sv24-media-inputs/sv24-u100.pt')
P = Path('artifacts/sv22-media-inputs/pv21-u400.pt')
SHA = '3088cf10f960892637a4016741f98427fbe2969f9c0a906632ed75de510df5cf'


def test_loader_identity(tmp_path):
    net, ck = load_spawner(S, expected_update=100, expected_spawner_sha256=SHA)
    assert ck['completed_updates'] == 100
    with pytest.raises(ValueError):
        load_spawner(S, expected_spawner_sha256='0'*64)
    with pytest.raises(ValueError):
        load_spawner(S, expected_update=99, expected_spawner_sha256=SHA)
    for key, value in [('partial_update', True), ('completed_updates', 99),
                       ('player_sha', 'bad'), ('player_state_sha', 'bad'),
                       ('architecture', 'bad'), ('model', {})]:
        bad = dict(ck); bad[key] = value
        path = tmp_path / 'bad.pt'; torch.save(bad, path)
        with pytest.raises((ValueError, RuntimeError)):
            load_spawner(path, expected_spawner_sha256=core.file_hash(path))


def test_rollout_matches_production():
    torch.set_num_threads(1)
    net, _ = load_spawner(S, expected_spawner_sha256=SHA)
    player, _ = core.load_player(P)
    seen = []
    actual = rollout(net, player, 3000, cap=120, on_frame=lambda e,a: seen.append(e.frame))
    expected, _, _ = run.episodes(net, player, [3000], 120, 'cpu', run.Guard(),
                                mode='stochastic', train_rng=None)
    assert actual == expected[0]
    assert seen == list(range(1, actual['frames']+1))


def test_synthetic_assets_use_existing_helpers(tmp_path):
    from PIL import Image
    from qrokkun_env.env import ACTIONS
    from qrokkun_ai.v21.render_player_v21_demo import load_bullet_sprites, render_frame
    for name in ['bullet_med.png', 'player.png', *[f'player_{a}.png' for a in ACTIONS if a != 'idle']]:
        Image.new('RGBA', (8, 8), 'white').save(tmp_path / name)
    bullets = load_bullet_sprites(tmp_path)
    sprites = {a: Image.open(tmp_path / ('player.png' if a == 'idle' else f'player_{a}.png')).convert('RGBA') for a in ACTIONS}
    net, _ = load_spawner(S, expected_spawner_sha256=SHA)
    player, _ = core.load_player(P)
    images = []
    rollout(net, player, 3000, cap=1, on_frame=lambda e,a: images.append(
        render_frame(e, a, tmp_path, 3, bullet_sprites=bullets, player_sprites=sprites)))
    assert images[0].size == (960, 720)


def test_loader_u980_identity(tmp_path):
    path = Path('artifacts/sv24-media-inputs/sv24-u980.pt')
    sha = '554290b6c17a280dbfb8bb4076583a1230b278bf73a24a701f66fca76e5bfcc0'
    _, ck = load_spawner(path, expected_update=980, expected_spawner_sha256=sha)
    assert ck['completed_updates'] == 980 and ck['partial_update'] is False
    with pytest.raises(ValueError):
        load_spawner(path, expected_update=980, expected_spawner_sha256='0'*64)
    for update in (100, 979, 1000):
        with pytest.raises(ValueError):
            load_spawner(path, expected_update=update, expected_spawner_sha256=sha)
    with pytest.raises(ValueError):
        load_spawner(path, expected_spawner_sha256=sha)
    for key, value in [('completed_updates', 1000), ('partial_update', True),
                       ('partial_update', None), ('player_sha', 'bad'),
                       ('player_state_sha', 'bad'), ('architecture', 'bad'), ('model', {})]:
        bad = dict(ck); bad[key] = value
        bad_path = tmp_path / 'bad980.pt'; torch.save(bad, bad_path)
        with pytest.raises((ValueError, RuntimeError)):
            load_spawner(bad_path, expected_update=980,
                         expected_spawner_sha256=core.file_hash(bad_path))
    _, old = load_spawner(S, expected_spawner_sha256=SHA)
    assert old['completed_updates'] == 100
