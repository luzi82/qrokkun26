import random
import numpy as np
import torch
from qrokkun_ai.v24.core import Spawner
from qrokkun_ai.v26.kernel import sample_player, sample_spawner, PythonRNG


def test_episode_rng_global_isolation():
    probs = torch.tensor([[.1,.2,.7],[.4,.3,.3]])
    before = torch.get_rng_state().clone()
    gs = [torch.Generator().manual_seed(s+26000000) for s in [100000,100001]]
    a, events = sample_player(probs, gs)
    assert torch.equal(before, torch.get_rng_state())
    expected = [torch.multinomial(probs[i],1,generator=torch.Generator().manual_seed(s+26000000)).item()
                for i,s in enumerate([100000,100001])]
    assert a.tolist() == expected
    assert all(set(e) >= {'source','counter','pre','post','output'} for e in events)


def test_s_and_python_draw_parity():
    torch.manual_seed(7)
    s = Spawner().eval()
    x = torch.zeros(1,325)
    a = torch.Generator().manual_seed(99)
    b = torch.Generator().manual_seed(99)
    events = []
    with torch.no_grad():
        expected = s.sample(x,a)
        actual = sample_spawner(s,x,b,events)
    assert all(torch.equal(x,y) for x,y in zip(expected, actual))
    assert torch.equal(a.get_state(),b.get_state())
    assert len([e for e in events if e['type']=='rng']) == 3
    original = random.Random('field')
    logs = []
    traced = PythonRNG(random.Random('field'),'field',logs)
    assert [original.uniform(0,6) for _ in range(5)] == [traced.uniform(0,6) for _ in range(5)]
    assert original.getstate() == traced.getstate()
    assert len(logs) == 5


def test_reset_ids_collision_cull_and_original_env():
    from qrokkun_ai.v26.kernel import Env
    from qrokkun_ai.v24.core import LearnedEnv
    from qrokkun_env.gv20 import Bullet
    s=Spawner().requires_grad_(False).eval()
    e=Env(s,100000)
    g=torch.Generator().manual_seed(25100000)
    def cb(obs):
        with torch.no_grad():return s.sample(torch.from_numpy(obs).unsqueeze(0),g)[0][0].numpy()
    original=LearnedEnv(cb,seed=100000);original.reset(seed=100000)
    for _ in range(80):
        assert e.step(0)==original.step(0)
    e.reset(seed=100000)
    assert e.birth==1
    assert e.bullet(e.bullets[0])['id']==0
    e.bullets.append(Bullet(-100,-100,0,0));e.tag('field')
    e.bullets.append(Bullet(e.px,e.py,0,0));e.tag('S')
    result,data,audit=e.traced_step(0)
    assert result[2] and e.dead
    assert any(x['type']=='removal' and x['reason']=='offscreen' for x in data['events'])
    assert any(x['type']=='collision' and x['ids']==[2] for x in data['events'])
