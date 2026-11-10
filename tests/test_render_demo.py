import torch
from qrokkun_ai.v22.render_demo import rollout
from qrokkun_ai.v22.core import Spawner
from qrokkun_ai.v22.run import episodes

class Player:
    def forward_policy(self,p,b,m):
        return type('Policy',(),{'logits':torch.zeros((p.shape[0],9))})()
class Guard:
    def check(self): pass

def test_rollout_matches_reference():
    torch.set_num_threads(1)
    net=Spawner().eval(); player=Player()
    expected,data=episodes(net,player,[3000],40,'cpu',Guard())
    frames=[]
    result=rollout(net,player,3000,cap=40,on_frame=lambda e,a:frames.append(e.frame))
    assert result==expected[0]
    assert frames==list(range(1,41))

def test_checkpoint_identity_rejected(tmp_path):
    import pytest
    from qrokkun_ai.v22.render_demo import load_spawner
    path=tmp_path/'wrong.pt'; torch.save({'update':320},path)
    with pytest.raises(ValueError,match='identity'): load_spawner(path)

def test_required_bullet_sprite_missing(tmp_path):
    import pytest
    from qrokkun_ai.v21.render_player_v21_demo import load_bullet_sprites,DemoRenderError
    with pytest.raises(DemoRenderError): load_bullet_sprites(tmp_path)

def test_hit_sprite_draw_changes_pixels(tmp_path):
    from PIL import Image
    from qrokkun_env.env import ACTIONS
    from qrokkun_ai.v22.core import LearnedEnv
    from qrokkun_ai.v21.render_player_v21_demo import render_frame
    import numpy as np
    e=LearnedEnv(lambda obs:np.zeros(5)); e.reset(3000)
    sprites={a:Image.new('RGBA',(8,8),'white') for a in ACTIONS}
    kw=dict(bullet_sprites={1:Image.new('RGBA',(8,8),'red')},player_sprites=sprites)
    normal=render_frame(e,'idle',tmp_path,3,**kw)
    hit=render_frame(e,'idle',tmp_path,3,hit_step=0,**kw)
    assert normal.size==(960,720)
    assert normal.tobytes()!=hit.tobytes()

def test_spawn_trajectory_equivalence():
    import numpy as np
    class Recording(Spawner):
        def distribution(self,x):
            self.inputs.append(x.detach().clone())
            d=super().distribution(x)
            self.means.append(d.mean.detach().clone())
            return d
    torch.set_num_threads(1)
    net=Recording().eval(); player=Player()
    for seed in [3000,3001]:
        net.inputs=[]; net.means=[]
        expected,data=episodes(net,player,[seed],160,'cpu',Guard())
        inputs=net.inputs; means=net.means
        net.inputs=[]; net.means=[]
        result=rollout(net,player,seed,cap=160)
        assert result==expected[0]
        assert len(inputs)>1
        assert all(torch.equal(a,b) for a,b in zip(inputs,net.inputs))
        assert all(torch.equal(a,b) for a,b in zip(means,net.means))
        # Inputs after later spawns encode the bullets produced by prior sampled actions.
        assert len(inputs)==len(net.inputs)

def test_checkpoint_metadata_rejected(tmp_path,monkeypatch):
    import pytest
    from qrokkun_ai.v22 import render_demo as r
    monkeypatch.setattr(r,'file_hash',lambda p:r.SV_SHA)
    path=tmp_path/'bad.pt'
    for update,partial in [(319,False),(320,True)]:
        torch.save(dict(update=update,partial_update=partial,model=Spawner().state_dict()),path)
        with pytest.raises(ValueError,match='update'): r.load_spawner(path)

def test_main_hit_tail_and_cleanup(tmp_path,monkeypatch):
    import sys,json
    from PIL import Image
    from qrokkun_ai.v22 import render_demo as r
    from qrokkun_env.env import ACTIONS
    assets=tmp_path/'assets'; assets.mkdir()
    for name in ['bullet_med.png']+['player.png' if a=='idle' else f'player_{a}.png' for a in ACTIONS]:
        Image.new('RGBA',(8,8),'white').save(assets/name)
    monkeypatch.setattr(r,'load_spawner',lambda p:object())
    monkeypatch.setattr(r,'load_player',lambda *a,**kw:(object(),{}))
    def fake_rollout(*a,on_frame,**kw):
        on_frame(object(),'idle')
        return dict(seed=3000,frames=1,seconds=1/60,death=True,censored=False,cap_frames=7200)
    monkeypatch.setattr(r,'rollout',fake_rollout)
    steps=[]
    def fake_frame(*a,hit_step=None,**kw):
        steps.append(hit_step); return Image.new('RGB',(960,720))
    monkeypatch.setattr(r,'render_frame',fake_frame)
    work=[]
    def encode(cmd,check):
        png=__import__('pathlib').Path(cmd[cmd.index('-i')+1]).parent
        work.append(png)
        assert len(list(png.glob('*.png')))==31
    monkeypatch.setattr(r.subprocess,'run',encode)
    output=tmp_path/'out.mp4'
    monkeypatch.setattr(sys,'argv',['render','--spawner-checkpoint','s','--player-checkpoint','p',
        '--assets',str(assets),'--out',str(output),'--seed','3000'])
    r.main()
    assert steps==[None]+list(range(30))
    assert not work[0].exists()
    assert json.loads(output.with_suffix('.mp4.json').read_text())['video_frames']==31

def test_missing_direction_art_fails_before_checkpoint_or_encoding(tmp_path,monkeypatch):
    import sys, pytest
    from PIL import Image
    from qrokkun_ai.v22 import render_demo as r
    from qrokkun_env.env import ACTIONS
    def forbidden(*args,**kwargs): pytest.fail('checkpoint or encoding reached with missing art')
    monkeypatch.setattr(r,'load_spawner',forbidden)
    monkeypatch.setattr(r.subprocess,'run',forbidden)
    for missing in ACTIONS:
        art=tmp_path/missing; art.mkdir()
        Image.new('RGBA',(3,3),'lime').save(art/'bullet_med.png')
        for action in ACTIONS:
            if action!=missing:
                Image.new('RGBA',(3,3),'red').save(art/('player.png' if action=='idle' else f'player_{action}.png'))
        monkeypatch.setattr(sys,'argv',['render','--spawner-checkpoint','s','--player-checkpoint','p','--assets',str(art),'--out',str(tmp_path/'out.mp4'),'--seed','3000'])
        with pytest.raises(FileNotFoundError): r.main()
