import torch
import pytest
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v24.core import Spawner
from qrokkun_ai.v24.run import Guard
from qrokkun_ai.v25.training import collect, optimizers, evaluate, check_frozen
from qrokkun_ai.v21.train.train_player_v21 import rollouts_to_batch_v21, train_step_v21


def test_real_env_on_policy_bootstrap_and_frozen_after_step():
    torch.manual_seed(17)
    p = PlayerV21(hidden=8)
    s = Spawner().requires_grad_(False).eval()
    digest = state_dict_sha256(s.state_dict())
    a, c = optimizers(p, Guard())
    assert not a.state and not c.state
    ro = collect(p, s, torch.device('cpu'), [100000, 100001], 3, Guard())
    batch = rollouts_to_batch_v21(ro, .9995, .999, torch.device('cpu'))
    with torch.no_grad():
        dist, _ = p(batch['player'], batch['bullets'], batch['pad'])
        assert torch.allclose(dist.log_prob(batch['actions']), batch['old_log_probs'])
    assert all(r.censored and not r.terminated and r.bootstrap_value != 0 for r in ro)
    train_step_v21(p, a, c, batch)
    check_frozen(s, digest)
    assert a.state and c.state


def test_eval_preserves_training_rng():
    p = PlayerV21(hidden=8)
    s = Spawner().requires_grad_(False).eval()
    before = torch.get_rng_state().clone()
    result = evaluate(p, s, [600000, 600001], 2, 'cpu', Guard())
    assert torch.equal(before, torch.get_rng_state())
    assert len(result['per_seed']) == 2


def test_physical_mapping_and_sampling_are_original_sv24():
    import numpy as np
    from qrokkun_ai.v24.core import LearnedEnv
    from qrokkun_ai.v25.training import environment
    s=Spawner().requires_grad_(False).eval()
    seed=100000
    rng=torch.Generator().manual_seed(seed+25000000)
    def original(obs):
        with torch.no_grad(): return s.sample(torch.from_numpy(obs).unsqueeze(0),rng)[0][0].numpy()
    a=environment(s,seed,Guard())
    b=LearnedEnv(original,seed=seed)
    a.reset(seed); b.reset(seed)
    for _ in range(20):
        a.step(0); b.step(0)
        assert [(v.x,v.y,v.vx,v.vy) for v in a.bullets]==[(v.x,v.y,v.vx,v.vy) for v in b.bullets]
    assert a.s_history==b.s_history and a.s_history


def test_death_on_exact_cap_has_no_bootstrap(monkeypatch):
    import qrokkun_ai.v25.training as training
    from qrokkun_env.gv20 import Gv20Env
    class Death(Gv20Env):
        def step(self,action):
            obs,base,done,info=super().step(action)
            self.dead=True
            return obs,base,True,info
    monkeypatch.setattr(training,'environment',lambda s,seed,g:Death(seed=seed))
    p=PlayerV21(hidden=8)
    ro=collect(p,None,torch.device('cpu'),[1],1,Guard())[0]
    assert ro.terminated and not ro.censored and not ro.truncated and ro.bootstrap_value==0


def test_optimizer_guard_prevents_step():
    from qrokkun_ai.v24.run import Stopped
    p=PlayerV21(hidden=8)
    g=Guard()
    opt,_=optimizers(p,g)
    before=state_dict_sha256(p.state_dict())
    for x in p.actor_parameters(): x.grad=torch.ones_like(x)
    g.signal(15)
    with pytest.raises(Stopped): opt.step()
    assert state_dict_sha256(p.state_dict())==before and not opt.state
