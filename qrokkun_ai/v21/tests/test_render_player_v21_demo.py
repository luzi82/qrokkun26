import json
from pathlib import Path

import pytest
from PIL import Image

from qrokkun_ai.v21 import render_player_v21_demo as demo
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, save_v21_checkpoint
from qrokkun_env.gv20 import Bullet, Gv20Env


def test_kindless_bullet_uses_medium_sprite(tmp_path):
    Image.new('RGBA', (3, 3), (17, 239, 81, 255)).save(tmp_path / 'bullet_med.png')
    env = Gv20Env(seed=3)
    env.reset(seed=3)
    env.bullets = [Bullet(50, 50, 0, 0)]
    assert not hasattr(env.bullets[0], 'kind')
    frame = demo.render_frame(
        env, 'idle', tmp_path, 1, bullet_sprites=demo.load_bullet_sprites(tmp_path)
    )
    assert frame.getpixel((50, 50)) == (17, 239, 81)


def test_missing_medium_sprite_fails_before_encoding(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('checkpoint loading or ffmpeg reached without medium sprite')

    monkeypatch.setattr(demo, 'load_demo_player', unexpected)
    monkeypatch.setattr(demo.subprocess, 'run', unexpected)
    with pytest.raises(demo.DemoRenderError, match='bullet_med.png'):
        demo.render_demo(tmp_path / 'unused.pt', tmp_path / 'demo.mp4', assets=tmp_path)


def test_short_hit_tail_and_sidecar(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'player.pt'
    payload = save_v21_checkpoint(PlayerV21(hidden=8), checkpoint, extra={'update': 42})
    Image.new('RGBA', (3, 3), 'lime').save(tmp_path / 'bullet_med.png')
    monkeypatch.setattr(demo, 'deterministic_player_action', lambda *args: 0)
    monkeypatch.setattr(Gv20Env, '_check_hit', lambda self: True)
    frame_dirs = []

    def fake_encode(command, **kwargs):
        frame_dir = Path(command[command.index('-i') + 1]).parent
        frame_dirs.append(frame_dir)
        assert len(list(frame_dir.glob('*.png'))) == 1 + demo.HIT_ANIM_FRAMES
        assert command[command.index('-c:v') + 1] == 'libx264'
        assert command[command.index('-pix_fmt') + 1] == 'yuv420p'
        assert command[command.index('-crf') + 1] == '18'
        assert command[command.index('-framerate') + 1] == '60'
        Path(command[-1]).touch()

    monkeypatch.setattr(demo.subprocess, 'run', fake_encode)
    output = tmp_path / 'demo.mp4'
    metadata = demo.render_demo(checkpoint, output, assets=tmp_path, scale=1, seed=17)
    assert metadata['result']['frames'] == 1
    assert metadata['result']['elapsed'] == pytest.approx(1 / 60)
    assert metadata['result']['seed'] == 17
    assert metadata['kind'] == 'player_v21_gv20_scripted_diagnostic_demo'
    assert metadata['experimental'] is True
    assert metadata['promotion'] == 'forbidden'
    assert metadata['checkpoint_selection'] == 'none'
    assert metadata['checkpoint'] == {
        'path': str(checkpoint.resolve()),
        'file_sha256': demo.file_sha256(checkpoint),
        'state_dict_sha256': payload['state_dict_sha256'],
        'extra': {'update': 42},
    }
    assert json.loads(demo._sidecar_path(output).read_text()) == metadata
    assert all(not path.exists() for path in frame_dirs)


@pytest.mark.parametrize('existing', ['mp4', 'sidecar', 'frames'])
def test_refuses_existing_destination(tmp_path, existing):
    output = tmp_path / 'demo.mp4'
    path = {'mp4': output, 'sidecar': demo._sidecar_path(output),
            'frames': tmp_path / '.demo.player_v21_frames'}[existing]
    if existing == 'frames':
        path.mkdir()
    else:
        path.touch()
    with pytest.raises(demo.DemoRenderError, match='refusing to overwrite'):
        demo.render_demo(tmp_path / 'unused.pt', output, assets=tmp_path)
    assert path.exists()
