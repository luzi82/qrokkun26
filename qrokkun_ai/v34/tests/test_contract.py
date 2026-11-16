import copy
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from qrokkun_env.gv20 import Bullet, Gv20Env
from qrokkun_ai.v34.filter import harmless
from qrokkun_ai.v34.adapter import encode_obs, runtime, EpisodePool
from qrokkun_ai.v34.initialization import random_player
from qrokkun_ai.v34 import run
from qrokkun_ai.v31 import core
from qrokkun_ai.v24.core import Spawner
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256


def scene(px=160.,py=126.):
    return NS(px=px,py=py,dt=1/60)


def test_cull_before_catch():
    assert harmless(scene(200),Bullet(330,126,50,0,moved=True))


def test_slow_departing_catchable():
    assert not harmless(scene(),Bullet(180,126,10,0,moved=True))


def test_inbound_outside_visual_field():
    assert not harmless(scene(),Bullet(330,126,-200,0,moved=True))


def test_tangent_retained():
    assert not harmless(scene(),Bullet(166.6,126,200,0,moved=False))


def test_birth_stationary_can_hit():
    assert not harmless(scene(),Bullet(167,126,200,0,moved=False))


def test_stationary_nearzero_nonfinite():
    for vx in (0,1e-15,float('nan'),float('inf')):
        assert not harmless(scene(),Bullet(180,126,vx,0,moved=True))


def test_intercept_after_twelve_frames():
    assert not harmless(scene(),Bullet(230,126,-40,0,moved=True))


def test_turning_reachable_intercept():
    # A turn after moving right can reach this slowly departing diagonal target.
    assert not harmless(scene(),Bullet(190,156,10,10,moved=True))


def test_radius_is_actual_not_fixed():
    assert not harmless(scene(),Bullet(200,126,200,0,radius=50,moved=True))


def test_float_quantization_uncertainty():
    # Drift-dominated motion cannot establish a finite lifetime.
    assert not harmless(scene(),Bullet(200,126,0.00001,0,moved=True))


def test_top64_replacement_stability_padding_immutability():
    env=Gv20Env(); env.px=160.; env.py=126.
    near=[Bullet(200,126,200,0,moved=True) for _ in range(70)]
    danger=[Bullet(280,126,-200,0,moved=True),Bullet(40,126,200,0,moved=True)]
    env.bullets=near+danger
    before=copy.deepcopy(env.__dict__)
    player,bullets,pad=encode_obs(env)
    assert (~pad).sum()==2
    assert bullets[0,0]>0 and bullets[1,0]<0
    assert np.all(bullets[2:]==core._PAD_ROW)
    assert env.bullets==before['bullets'] and env.px==before['px']
    assert len(player)==6 and bullets.shape==(64,4)


def test_scoped_restore_after_failure():
    old=core.encode_obs_fast
    with pytest.raises(RuntimeError):
        with runtime():
            assert core.encode_obs_fast is encode_obs
            raise RuntimeError('primary')
    assert core.encode_obs_fast is old


def test_init_rng_weights_fresh_adam():
    before=torch.get_rng_state().clone()
    p=random_player()
    assert torch.equal(before,torch.get_rng_state())
    assert state_dict_sha256(p.state_dict())=='49a7c9426a1054fc000caf359fc1a0a3c90db49adfe6a53c3275350a9b5a7942'
    a,c=core.optimizers(p)
    assert not a.state and not c.state


def test_protocol_boundaries_and_ties():
    from qrokkun_ai.v34.protocol import schedule,better,training_seeds
    assert schedule()==list(range(0,1001,10))
    assert not better(100,100) and better(101,100)
    assert training_seeds(2)==list(range(100020,100040))


def args(*extra):
    return run.parser().parse_args(['train','--out-dir','/tmp/unused-pv34','--source-commit','a'*40,*extra])


def test_formal_config_and_source():
    assert len(run.validate(args())[1])==101
    a=args(); a.source_commit='HEAD'
    with pytest.raises(ValueError): run.validate(a)


def test_nonformal_no_final():
    a=args('--nonformal'); a.command='final'
    with pytest.raises(ValueError): run.validate(a)


def test_control_mismatch_memory(tmp_path):
    from qrokkun_ai.v34.reference import load_reference
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'): load_reference(tmp_path)


def test_cleanup_preserves_primary_and_observed_failures():
    pool=EpisodePool.__new__(EpisodePool)
    pool._closed=False
    pool.stop_event=NS(set=lambda:None)
    pool.children=[]
    def fail(): raise OSError('unlink observed')
    pool.w_shm=NS(close=lambda:None,unlink=fail); pool.t_shm=None
    with pytest.raises(BaseExceptionGroup) as caught:
        try: raise ValueError('primary observed')
        except ValueError: pool.close()
    assert 'primary observed' in repr(caught.value) and 'unlink observed' in repr(caught.value)


def test_spawn_pool_filter_and_parent_restored():
    old=core.encode_obs_fast; target=core._pool_worker
    p=random_player(8); s=Spawner().eval()
    pool=EpisodePool(p,s,1,3,1,'off',lambda:False)
    try:
        assert core._pool_worker is target and core.encode_obs_fast is old
        pool.sync_weights(p.state_dict(),0,lambda:False)
        rows=pool.collect([100000],3,validation=True,stop_check=lambda:False)
        assert rows[0].summary()['filter_stats']['considered']>0
        assert rows[0].bullets.shape==(3,64,4)
        stats=pool.shutdown_and_stats()
        assert stats and all(not r['cuda_initialized'] for r in stats)
    finally: pool.close()
    assert core.encode_obs_fast is old


def test_filtered_trajectory_teacher_features(monkeypatch):
    import numpy as np
    from qrokkun_ai.v24.core import LearnedEnv
    from qrokkun_env.gv20 import Bullet
    from qrokkun_ai.v34.adapter import run_episode
    from qrokkun_ai.v31 import core
    class Scene(LearnedEnv):
        def reset(self,*a,**k):
            result=super().reset(*a,**k)
            self.bullets=[Bullet(self.px+40,self.py,200,0,moved=True)]*70+[Bullet(self.px-80,self.py,100,0,moved=True)]
            return result
    monkeypatch.setattr(core,'LearnedEnv',Scene)
    with runtime():
        meta,buf=run_episode(random_player(8),Spawner().eval(),100000,1)
    assert meta['filter_stats']['discarded']>=70
    assert (~buf['pad'][0]).sum()==1
    ep=core._slice_trajectory(meta,buf)
    net=random_player(8)
    batch=core.assemble_batch([ep],net,'cpu')
    assert np.array_equal(batch['bullets'].numpy(),ep.bullets)
    # Production train computes both clearance features and static teacher from
    # these exact tensors; retain the unchanged hoisted numerical backend.
    import inspect
    code=inspect.getsource(core.train)
    assert 'net.features(player, bullets, pad)' in code


def test_final_once_claim_rejection_is_terminal(monkeypatch):
    from qrokkun_ai.v34 import final
    a=args(); a.command='final'
    monkeypatch.setattr(final,'preflight',lambda *a: ({},{}))
    def claimed(*a): raise FileExistsError('already claimed')
    monkeypatch.setattr(final,'claim',claimed)
    monkeypatch.setattr(final,'json_write' if hasattr(final,'json_write') else 'EpisodePool', lambda *a: (_ for _ in ()).throw(AssertionError('unexpected downstream')))
    with pytest.raises(FileExistsError): final.execute(a)
