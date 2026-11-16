import torch
from qrokkun_ai.v31 import core
from qrokkun_env.gv36 import Gv36Env

def test_runtime_installs_real_integer_environment():
    from qrokkun_ai.v37.adapter import runtime
    old = core.LearnedEnv
    with runtime():
        env = core.LearnedEnv(lambda obs: (0., 30., 0.), seed=17)
        env.reset(17)
        env.step(3)
        assert isinstance(env, Gv36Env)
        assert env.x1 == 1
    assert core.LearnedEnv is old

import copy
import pickle
import numpy as np
import pytest
from qrokkun_ai.v37.world import LearnedEnv, snapshot, restore
from qrokkun_ai.v37.teaching import Capture, CaptureEnv, Sampler, fork, auxiliary, validate, finish
from qrokkun_ai.v37.planner import search, predict
from qrokkun_ai.v35.initialization import random_player
from qrokkun_ai.v24.core import Spawner

def model(hidden=4):
    with torch.random.fork_rng(devices=[]): return random_player(hidden)

def world():
    with torch.random.fork_rng(devices=[]): s = Spawner().eval().requires_grad_(False)
    env = LearnedEnv(Sampler(s, torch.Generator().manual_seed(33).get_state()), seed=17)
    env.reset(17)
    return env

def test_snapshot_multiple_events_and_private_rng(monkeypatch):
    monkeypatch.setattr(LearnedEnv, "_check_hit", lambda self: False)
    env = world(); env.reset(17)
    state = snapshot(env); rng = env.callback.generator.get_state().clone()
    branch = fork(env)
    for _ in range(140):
        env.step(0); branch.step(0)
        assert snapshot(env)['_player'] == snapshot(branch)['_player']
        assert env.bullets == branch.bullets
        assert env._next_event == branch._next_event
    assert len(env.s_history)>1 and env.field_history
    assert not torch.equal(rng, env.callback.generator.get_state())
    assert state['frame'] == 0

def test_fork_does_not_advance_original():
    env = world(); before = pickle.dumps(snapshot(env)); rng=env.callback.generator.get_state().clone()
    fork(env).step(3)
    assert pickle.dumps(snapshot(env)) == before
    assert torch.equal(rng,env.callback.generator.get_state())

def test_spawn_method_reused():
    from qrokkun_ai.v24.core import LearnedEnv as Old
    env=world(); assert env._spawn_one.__func__ is LearnedEnv._spawn_one
    assert isinstance(env,Gv36Env)

def test_p_cache_and_rebase_replay():
    env=world()
    for _ in range(20): env.step(3)
    branch=fork(env)
    assert branch.get_player_state()==env.get_player_state()
    assert branch.position_key==env.position_key and 0<=env.x1<10

def test_planner_positions_and_wall():
    env=world(); env.set_player_position(306,226)
    result=search(env.get_player_state(), [], horizon=3,width=4,top=2)
    assert len(result['candidates'])==2
    for candidate in result['candidates']:
        for action,pos in zip(candidate['actions'],candidate['positions'][1:]):
            env._integrate_player(*env._parse_action(action)); assert [env.px,env.py]==pos
        env.set_player_position(306,226)

def test_thirteenth_bullet_is_included():
    env=world(); bullets=[[0,0,0,0,1,True]]*12+[[160,126,0,0,1000,True]]
    assert search(env.get_player_state(),bullets,horizon=1)['candidateN']==0

def test_prediction_birth_delay():
    rows=predict([[10,20,60,0,2,False]],2)
    assert rows[0][0,0]==10 and rows[1][0,0]==11

def test_planner_order_deterministic():
    env=world()
    assert search(env.get_player_state(),[],horizon=2)['candidates']==search(env.get_player_state(),[],horizon=2)['candidates']

def test_first_safe_path_has_sixty_observations():
    env=world(); env.bullets=[]
    # Frozen deterministic callback makes this a safe actual full-world fixture.
    env.callback.model.actor[-1].bias.data[0]=0
    result=search(env.get_player_state(),[],width=4,top=2)
    rows,n=validate(env,result['candidates'])
    assert len(rows)==60 and n==1
    assert rows[0][1].shape==(64,4)

def test_censored_never_searches():
    env=world(); assert finish(env)[1]['searched']==0

def test_early_death_skips():
    env=world();env.dead=True;env.frame=11
    assert finish(env)[1]['skippedearly']==1

def test_capture_bounded_and_replays(monkeypatch):
    monkeypatch.setattr(LearnedEnv, "_check_hit", lambda self: False)
    env=CaptureEnv(lambda obs:(0.,30.,0.),seed=17)
    # Use a private model sampler for replay.
    template=world();env.callback=template.callback
    env.reset(17)
    cap=Capture(env.callback.model,env.callback.generator);env.capture=cap
    for _ in range(90): env.step(0)
    anchor=cap.anchor(78)
    assert anchor.frame==78 and len(cap.checkpoints)<=2 and len(cap.actions)<=128

def demo_rows(n=3):
    from qrokkun_ai.v35.adapter import encode_obs
    obs=encode_obs(world())
    return [(*obs,0)]*n

def test_aux_actor_changes_critic_unchanged():
    net=model();oa,oc=core.optimizers(net)
    before=copy.deepcopy(net.state_dict()); critic=pickle.dumps(oc.state_dict())
    result=auxiliary(net,oa,demo_rows(),1)
    assert result['auxsteps']==1 and result['samples']==3
    assert any(not torch.equal(before[k],v) for k,v in net.state_dict().items() if k.startswith('policy'))
    assert all(torch.equal(before[k],v) for k,v in net.state_dict().items() if k.startswith('value'))
    assert pickle.dumps(oc.state_dict())==critic

def test_zero_weight_no_mutation_or_rng():
    net=model();oa,_=core.optimizers(net); before=pickle.dumps(oa.state_dict()); rng=torch.get_rng_state()
    auxiliary(net,oa,demo_rows(),1,weight=0)
    assert pickle.dumps(oa.state_dict())==before and torch.equal(rng,torch.get_rng_state())

def test_empty_aux_skips_optimizer():
    net=model();oa,_=core.optimizers(net)
    assert auxiliary(net,oa,[],1)['auxsteps']==0 and not oa.state

def test_aux_sampling_bounded_rng_isolated():
    net=model();oa,_=core.optimizers(net);rng=torch.get_rng_state()
    assert auxiliary(net,oa,demo_rows(513),1)['samples']==512
    assert torch.equal(rng,torch.get_rng_state())

def test_u0_isolated_and_identical():
    rng=torch.get_rng_state();a=model();b=model()
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

def test_sidecar_small_header_and_consumption(tmp_path):
    from qrokkun_ai.v37 import episode
    from qrokkun_ai.v37.adapter import slice_trajectory
    episode.ROOT=str(tmp_path)
    handle=episode.save_sidecar(demo_rows(),{'demoframes':3})
    assert len(pickle.dumps(handle))<256
    assert (tmp_path/handle).exists()
    meta=dict(seed=17,frames=1,terminated=False,censored=True,truncated=True,bootstrap_value=0.,elapsed=0.,pure_rollout_s=0.,teaching=handle)
    buf=core._alloc_local_slot(1)
    ep=slice_trajectory(meta,buf)
    assert len(ep.demos)==3 and not (tmp_path/handle).exists()

def test_trial_gate_rejects_formal():
    from qrokkun_ai.v37.run import _train_main
    with pytest.raises(ValueError,match='formal gate'):
        _train_main(['train','--role','control','--out-dir','unused','--source-commit','a'*40])

def test_fingerprint_bound_to_new_env():
    from qrokkun_ai.v37.fingerprint import source_digest
    assert len(source_digest())==64

def test_spawn_pipeline_more_episodes_than_workers(tmp_path):
    from qrokkun_ai.v37.adapter import EpisodePool, runtime
    net=model(); s=world().callback.model
    with runtime():
        pool=EpisodePool(p_template=net,s_cpu=s,max_slots=16,max_cap=2,workers=1,pin_cores='off',root=tmp_path/'cache',search=True)
        try:
            pool.sync_weights(net.state_dict(),policy_version=0)
            eps=pool.collect([17,18,19],2,validation=False)
            assert [e.seed for e in eps]==[17,18,19]
            assert all(not e.demos for e in eps)
            vals=pool.collect([20,21],2,validation=True)
            assert all(not e.teaching for e in vals)
            assert not list((tmp_path/'cache').glob('*.npz'))
            pool.shutdown_and_stats()
        finally: pool.close()

def test_ordinary_rollout_differential(tmp_path):
    from qrokkun_ai.v37 import episode
    from qrokkun_ai.v35.adapter import runtime as old_runtime
    net=model();s=world().callback.model
    original=core.run_episode
    with torch.no_grad(), old_runtime():
        previous=core.LearnedEnv
        try:
            core.LearnedEnv=LearnedEnv
            ma,ba=original(net,s,17,4)
        finally: core.LearnedEnv=previous
    episode.ROOT=str(tmp_path);episode.SEARCH=False
    with torch.no_grad(): mb,bb=episode.run_episode(net,s,17,4)
    for key in ba: np.testing.assert_array_equal(ba[key],bb[key])
    for key in ma:
        if key!='pure_rollout_s': assert ma[key]==mb[key]
    assert len(pickle.dumps(mb))<256
    (tmp_path/mb['teaching']).unlink()

def test_validation_never_captures_or_finishes(monkeypatch):
    from qrokkun_ai.v37 import episode
    def forbidden(*a): raise AssertionError('validation search hook')
    monkeypatch.setattr(episode,'finish',forbidden)
    monkeypatch.setattr(episode,'Capture',forbidden)
    monkeypatch.setattr(episode,'SEARCH',True)
    with torch.no_grad(): meta,_=episode.run_episode(model(),world().callback.model,17,3,validation=True)
    assert 'teaching' not in meta

def test_counterfactual_branches_have_independent_s_rng():
    env=world(); a=fork(env); b=fork(env)
    a.set_player_position(14,26); b.set_player_position(306,226)
    a._spawn_one(); b._spawn_one()
    assert torch.equal(a.callback.generator.get_state(),b.callback.generator.get_state())
    assert a.bullets[-1] != b.bullets[-1]

def test_search_does_not_consume_any_rng():
    env=world(); before=pickle.dumps(snapshot(env)); rng=torch.get_rng_state()
    s_rng=env.callback.generator.get_state().clone()
    search(env.get_player_state(),[],horizon=2,width=4)
    assert pickle.dumps(snapshot(env))==before
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(s_rng,env.callback.generator.get_state())

def test_aux_existing_critic_adam_unchanged():
    net=model();oa,oc=core.optimizers(net)
    sum(p.sum() for p in net.critic_parameters()).backward();oc.step()
    before=copy.deepcopy(oc.state_dict()); values=[p.detach().clone() for p in net.critic_parameters()]
    auxiliary(net,oa,demo_rows(),1)
    for a,b in zip(values,net.critic_parameters()): assert torch.equal(a,b)
    for k,state in before['state'].items():
        for key,val in state.items(): assert torch.equal(val,oc.state_dict()['state'][k][key])

def test_fingerprint_files_include_environment_and_required_json():
    from qrokkun_ai.v37.fingerprint import source_files
    names={str(p) for p in source_files()}
    assert any(p.endswith('/qrokkun_env/gv36.py') for p in names)
    assert any(p.endswith('/qrokkun_ai/v37/contract.json') for p in names)
    assert any(p.endswith('/qrokkun_ai/v25/reference.json') for p in names)
    assert not any('/tests/' in p or p.endswith('.md') for p in names)

def test_validate_rejects_incomplete_candidate_before_world_replay():
    env=world()
    candidate=search(env.get_player_state(),[],horizon=1,width=1)['candidates'][0]
    with pytest.raises(ValueError,match='60'):
        validate(env,[candidate])

def test_full_snapshot_replay_preserves_all_world_state(monkeypatch):
    monkeypatch.setattr(LearnedEnv,'_check_hit',lambda self:False)
    env=world()
    for action in [3,4,5,6]*20: env.step(action)
    branch=fork(env)
    before=pickle.dumps(snapshot(env)); private=env.callback.generator.get_state().clone()
    for action in [0,1,2,3]*35: branch.step(action)
    assert pickle.dumps(snapshot(env))==before
    assert torch.equal(private,env.callback.generator.get_state())
    for action in [0,1,2,3]*35: env.step(action)
    assert pickle.dumps(snapshot(env))==pickle.dumps(snapshot(branch))
    assert torch.equal(env.callback.generator.get_state(),branch.callback.generator.get_state())

def test_fatal_sixtieth_step_rejected_then_first_success(monkeypatch):
    from qrokkun_ai.v37 import teaching
    env=world()
    candidates=search(env.get_player_state(),[],width=2,top=2)['candidates']
    original=teaching.fork; calls=[]
    def make_branch(anchor):
        branch=original(anchor); calls.append(branch)
        step=branch.step
        first=len(calls)==1
        def checked(action):
            result=step(action)
            if first and branch.frame==60: return result[0],result[1],True,result[3]
            return result
        branch.step=checked
        return branch
    monkeypatch.setattr(teaching,'fork',make_branch)
    rows,n=validate(env,candidates)
    assert n==2 and len(rows)==60
    assert [r[-1] for r in rows]==candidates[1]['actions']

def test_training_death_search_once_and_anchor(monkeypatch):
    from qrokkun_ai.v37 import teaching
    env=world();env.dead=True;env.frame=40
    calls=[]
    class Checkpoints:
        def anchor(self,frame):
            calls.append(frame);return world()
    env.capture=Checkpoints()
    def planned(state,bullets):
        calls.append('search')
        return dict(candidates=[],candidateN=0,modelcompletecaseN=0,search_s=0.)
    monkeypatch.setattr(teaching,'search',planned)
    rows,ledger=finish(env)
    assert calls==[28,'search'] and not rows
    assert ledger['searched']==1 and ledger['validatedpathN']==0
