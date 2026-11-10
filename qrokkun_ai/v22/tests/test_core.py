import numpy as np
from qrokkun_ai.v22.core import LearnedEnv, mapping, timed_returns

def test_frame_zero_mapping():
    seen=[]
    e=LearnedEnv(lambda x: seen.append(x.copy()) or np.array([0,0,-3,0,0.]))
    e.reset(17)
    assert len(seen)==1 and np.isfinite(seen[0]).all()
    assert e.s_history==[(0,115.)]
    assert len(e.bullets)==1 and not e.bullets[0].moved
    p,d,s=mapping(np.array([0,100,-3,0,0.]))
    assert s==200 and np.isfinite(d).all()

def test_timed_returns():
    r,g=timed_returns([0,20,70],120)
    assert np.isclose(sum(r),-120/7200)
    assert np.allclose(g,[-120/7200,-100/7200,-50/7200])

import torch
import time
import pytest
from qrokkun_ai.v22.core import Spawner
from qrokkun_ai.v22.run import Guard, Stopped, episodes, ppo
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256

def fixture_player():
    # Synthetic tiny REAL architecture; explicitly NOT the experiment opponent.
    p=PlayerV21(top_k=2,hidden=8,lookahead_horizon=1)
    return p.eval().requires_grad_(False)

def test_raw_ratio_and_finite_gradient():
    torch.set_num_threads(1); n=Spawner(); x=torch.zeros(8,325)
    d=n.distribution(x); raw=d.sample(); old=d.log_prob(raw).sum(-1).detach()
    assert torch.equal((n.distribution(x).log_prob(raw).sum(-1)-old).exp(),torch.ones(8))
    data=list(zip(x.numpy(),raw.numpy(),old.numpy(),np.linspace(-1,0,8)))
    m=ppo(n,torch.optim.Adam(n.parameters()),data,Guard(time.time()+30),1,.001)
    assert np.isfinite(list(m.values())).all() and m['parameter_delta']>0

def test_replay_eval_isolation_and_frozen_player():
    torch.set_num_threads(1); torch.manual_seed(17)
    n=Spawner(); p=fixture_player(); before=state_dict_sha256(p.state_dict())
    rng=torch.Generator().manual_seed(17); r=rng.get_state().clone(); global_r=torch.get_rng_state().clone()
    weights=state_dict_sha256(n.state_dict())
    for mode in ('scripted','stochastic'):
        a,_=episodes(n,p,[400000,400001],40,'cpu',Guard(time.time()+30),mode)
        b,_=episodes(n,p,[400000,400001],40,'cpu',Guard(time.time()+30),mode)
        assert a==b
    assert torch.equal(r,rng.get_state()) and torch.equal(global_r,torch.get_rng_state())
    assert before==state_dict_sha256(p.state_dict()) and weights==state_dict_sha256(n.state_dict())
    assert all(v.grad is None and not v.requires_grad for v in p.parameters())

def test_deadline_interrupts_without_death():
    class TickGuard:
        ticks=0
        def check(self):
            self.ticks+=1
            if self.ticks>5: raise Stopped('test-deadline')
    with pytest.raises(Stopped): episodes(Spawner(),fixture_player(),[1000000],7200,'cpu',TickGuard())

def test_schedule_and_public_observation():
    from qrokkun_env.gv20 import spawn_frames
    seen=[]
    e=LearnedEnv(lambda o: seen.append((e.frame,o.copy())) or np.array([0,0,-3,1,0.]))
    e.reset(17)
    # Isolate schedule from collisions without altering production implementation.
    e._check_hit=lambda: False
    for _ in range(120): e.step(0)
    assert [f for f,_ in seen]==list(spawn_frames(121))
    assert all(o.shape==(325,) and np.isfinite(o).all() for _,o in seen)
    assert seen[0][1][-64:].sum()==0
    assert len(e.field_history)==2


def test_zero_direction_and_extremes():
    from qrokkun_env.gv20 import aimed_velocity
    for raw in ([100,-100,100,0,0],[-100,100,-100,1,1]):
        p,d,s=mapping(raw)
        assert 30<=s<=200
        assert np.isfinite(aimed_velocity(p,(160,126),d,s)).all()
    with pytest.raises(ValueError): mapping([0]*4)
    with pytest.raises(ValueError): mapping([float('nan')]*5)

def test_checkpoint_on_interrupted_rollout_and_no_final_access(tmp_path,monkeypatch):
    import sys
    import qrokkun_ai.v22.run as run
    monkeypatch.setattr(sys,'argv',['sv22','train','--out-dir',str(tmp_path/'run')])
    monkeypatch.setattr(run,'load_player',lambda *args:(fixture_player(),{}))
    calls=[]
    def interrupted(*args,**kwargs):
        calls.extend(args[2]); raise Stopped('test-deadline-no-death')
    monkeypatch.setattr(run,'episodes',interrupted)
    run.main()
    c=torch.load(tmp_path/'run/latest.pt',weights_only=False)
    assert c['status']=='stopped:test-deadline-no-death'
    assert 'optimizer' in c and 'train_rng' in c and 'torch_rng' in c
    assert calls and all(400000<=s<400016 for s in calls)
    assert not (tmp_path/'run/training-closed.json').exists()

def test_no_launch_after_hard_deadline(tmp_path,monkeypatch,capsys):
    import sys
    import qrokkun_ai.v22.run as run
    monkeypatch.setattr(sys,'argv',['sv22','train','--out-dir',str(tmp_path/'never')])
    monkeypatch.setattr(run.time,'time',lambda:run.HARD+1)
    run.main()
    assert 'not_started' in capsys.readouterr().out
    assert not (tmp_path/'never').exists()
