import os
from pathlib import Path

import pytest
from qrokkun_ai.v25 import run


SYNTHETIC_SOURCE_COMMIT = '0123456789abcdef0123456789abcdef01234567'


def test_formal_config_and_nonformal_final_refused():
    a = run.parser().parse_args(['train', '--out-dir', 'unused'])
    run.validate_args(a)
    a.episodes_per_update = 1
    with pytest.raises(ValueError): run.validate_args(a)
    a.nonformal = True
    run.validate_args(a)
    a.command = 'final'
    with pytest.raises(ValueError): run.validate_args(a)


def test_input_wrong_hash_fails_before_load(tmp_path):
    p = tmp_path / 'bad.pt'
    p.write_bytes(b'bad')
    with pytest.raises(ValueError): run.inputs(str(p), str(p), 'cpu')


def test_direct_final_needs_clean_closure(tmp_path):
    a = run.parser().parse_args(['final', '--out-dir', str(tmp_path)])
    with pytest.raises((ValueError, FileNotFoundError)): run.preflight_final(a)


def test_canonical_mean_does_not_cross_at_u0():
    from qrokkun_ai.v25.core import REFERENCE,THRESHOLD
    assert run.validation_mean(REFERENCE)==THRESHOLD


def test_real_locked_input_metadata_only():
    # Explicit integration opt-in; still no episodes or optimizer steps.
    if os.environ.get('PV25_TEST_REAL_INPUTS') != '1':
        pytest.skip('integration: set PV25_TEST_REAL_INPUTS=1 for locked artifact checks')
    player = os.environ.get('PV25_TEST_PLAYER_CHECKPOINT', run.P_PATH)
    spawner = os.environ.get('PV25_TEST_SPAWNER_CHECKPOINT', run.S_PATH)
    if not Path(player).is_file() or not Path(spawner).is_file():
        pytest.skip('integration: locked player/spawner artifact unavailable')
    p,s,m=run.inputs(player,spawner,'cpu')
    assert m['extra']['update']==400
    assert run.state_dict_sha256(p.state_dict())==run.STATE_SHA
    assert all(x.requires_grad for x in p.parameters())
    assert not s.training and all(not x.requires_grad for x in s.parameters())


def synthetic_inputs(*args):
    from qrokkun_ai.v21.agents.player_v21 import PlayerV21
    from qrokkun_ai.v24.core import Spawner
    p=PlayerV21(hidden=8)
    return p,Spawner().requires_grad_(False).eval(),{}


def fake_eval(model,s,seeds,cap,device,guard,on_complete=None):
    from qrokkun_ai.v24.run import summary
    rows=[dict(seed=x,frames=cap,seconds=cap/60,death=False,censored=True,cap_frames=cap) for x in seeds]
    for r in rows:
        guard.check()
        if on_complete: on_complete(r)
    return summary(rows)


def test_synthetic_nonformal_training_has_no_final_or_milestone(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'inputs',synthetic_inputs)
    monkeypatch.setattr(run,'evaluate',fake_eval)
    a=run.parser().parse_args(['train','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path),'--nonformal','--updates','1',
                              '--episodes-per-update','1','--max-steps','2','--eval-seeds','2'])
    run.main(a)
    import json
    assert json.loads((tmp_path/'status.json').read_text())['status']=='nonformal_complete'
    import torch
    ck=torch.load(tmp_path/'latest.pt',map_location='cpu',weights_only=False)
    assert ck['formal'] is False
    assert not (tmp_path/'update500.pt').exists()
    a.command='final'
    with pytest.raises(ValueError): run.main(a)
    assert not (tmp_path/'final-claimed.json').exists()


def test_historical_gate_before_any_training(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'inputs',synthetic_inputs)
    monkeypatch.setattr(run,'evaluate',fake_eval)
    def forbidden(*args,**kwargs): raise AssertionError('training reached')
    monkeypatch.setattr(run,'collect',forbidden)
    a=run.parser().parse_args(['train','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    with pytest.raises(ValueError,match='historical'): run.main(a)
    assert not (tmp_path/'best.pt').exists()


def test_paired_exact_window_and_ci_zero():
    rows=[dict(seed=x,seconds=10) for x in run.HELDOUT]
    assert run.paired(rows,rows)['interpretation']=='indistinguishable'
    with pytest.raises(ValueError): run.paired(rows[:-1],rows[:-1])


def test_exclusive_claim(tmp_path):
    path=tmp_path/'claim'
    run.claim(path,{'x':1})
    with pytest.raises(FileExistsError): run.claim(path,{'x':2})


def closed_synthetic(tmp_path,a):
    from qrokkun_ai.v21.agents.player_v21 import save_v21_checkpoint
    p,_,_=synthetic_inputs()
    names=['best.pt','latest.pt','initial.pt','update500.pt','update1000.pt']
    for name in names:
        c=save_v21_checkpoint(p,tmp_path/name)
        c.update(checkpoint_type='pv25_warm_start_v1',formal=True,partial_update=False,selection=run.Selection().report())
        run.atomic(tmp_path/name,c)
    run.json_write(tmp_path/'status.json',dict(status='complete'))
    run.claim(tmp_path/'train-claimed.json',{'id':'synthetic'})
    run.json_write(tmp_path/'manifest.json',dict(formal=True,config=run.config(a),run_id='synthetic',
        source_commit=run.v24.resolve_source_commit(a.source_commit),player_sha=run.PLAYER_SHA,
        player_state_sha=run.STATE_SHA,spawner_sha=run.S_SHA,heldout=run.HELDOUT))
    run.json_write(tmp_path/'training-closed.json',dict(status='complete',run_id='synthetic',completed_updates=1000,
        last_complete_validation=1000,partial_update=False,u0_reference_sha=run.REFERENCE_SHA,
        manifest_sha=run.file_hash(tmp_path/'manifest.json'),checkpoint_hashes={n:run.file_hash(tmp_path/n) for n in names}))


def test_final_one_shot_64_synthetic_completion(tmp_path,monkeypatch):
    a=run.parser().parse_args(['final','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    closed_synthetic(tmp_path,a)
    monkeypatch.setattr(run,'inputs',synthetic_inputs)
    monkeypatch.setattr(run,'evaluate',fake_eval)
    run.main(a)
    import json
    seal=json.loads((tmp_path/'final-original-completion.json').read_text())
    assert seal['status']=='complete' and seal['completed_seeds']==run.HELDOUT and not seal['unfinished_seeds']
    assert (tmp_path/'final.json').exists()
    with pytest.raises(ValueError,match='claimed'): run.main(a)


@pytest.mark.parametrize('failure,state',[(run.Stopped,'stopped'),(RuntimeError,'failed')])
def test_failed_final_claim_cannot_repeat(tmp_path,monkeypatch,failure,state):
    import json
    a=run.parser().parse_args(['final','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    closed_synthetic(tmp_path,a)
    monkeypatch.setattr(run,'inputs',synthetic_inputs)
    def fail(*args,**kwargs): raise failure('synthetic failure')
    monkeypatch.setattr(run,'evaluate',fail)
    with pytest.raises(failure): run.main(a)
    assert (tmp_path/'final-claimed.json').exists() and not (tmp_path/'final.json').exists()
    claim_bytes=(tmp_path/'final-claimed.json').read_bytes()
    status_bytes=(tmp_path/'final-status.json').read_bytes()
    status=json.loads(status_bytes)
    assert status['status']==state and status['error']==f'{failure.__name__}: synthetic failure'
    with pytest.raises(ValueError,match='claimed'): run.main(a)
    assert (tmp_path/'final-claimed.json').read_bytes()==claim_bytes
    assert (tmp_path/'final-status.json').read_bytes()==status_bytes


@pytest.mark.parametrize('seam',['second_preflight','exclusive_claim'])
def test_unclaimed_final_loser_preserves_owner_artifacts(tmp_path,monkeypatch,seam):
    a=run.parser().parse_args(['final','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    closed_synthetic(tmp_path,a)
    monkeypatch.setattr(run,'inputs',synthetic_inputs)
    preflight,exclusive_claim=run.preflight_final,run.claim
    in_owner=False
    preflights={'owner':0,'loser':0}
    episodes={'owner':[],'loser':[]}
    loser_writes=[]
    owner_artifacts={}

    def actor(): return 'owner' if in_owner else 'loser'

    def evaluate(*args,**kwargs):
        episodes[actor()].extend(args[2])
        return fake_eval(*args,**kwargs)

    monkeypatch.setattr(run,'evaluate',evaluate)
    def no_collection(*args,**kwargs): raise AssertionError('final must not collect training episodes')
    monkeypatch.setattr(run,'collect',no_collection)
    for name in ('json_write','append','atomic'):
        original=getattr(run,name)
        def write(path,*args,_original=original,**kwargs):
            if not in_owner: loser_writes.append(Path(path).name)
            return _original(path,*args,**kwargs)
        monkeypatch.setattr(run,name,write)

    def finish_owner():
        nonlocal in_owner
        # Reenter at the seam to run the competing invocation to completion.
        # Both first preflights pass before the owner acquires its claim.
        in_owner=True
        try:
            run.main(a)
        finally:
            in_owner=False
        owner_artifacts.update({p.name:p.read_bytes() for p in tmp_path.iterdir()})

    def checked_preflight(args):
        preflights[actor()]+=1
        if not in_owner and preflights['loser']==2 and seam=='second_preflight':
            finish_owner()
        return preflight(args)

    def racing_claim(path,value):
        if not in_owner and Path(path).name=='final-claimed.json':
            assert seam=='exclusive_claim' and preflights['loser']==2
            finish_owner()
        return exclusive_claim(path,value)

    monkeypatch.setattr(run,'preflight_final',checked_preflight)
    monkeypatch.setattr(run,'claim',racing_claim)
    error=ValueError if seam=='second_preflight' else FileExistsError
    with pytest.raises(error,match='claimed' if seam=='second_preflight' else 'final-claimed'):
        run.main(a)
    assert preflights=={'owner':2,'loser':2}
    assert episodes['owner']==run.HELDOUT+run.HELDOUT and episodes['loser']==[]
    assert (tmp_path/'final-status.json').read_bytes()==owner_artifacts['final-status.json']
    assert (tmp_path/'final-claimed.json').read_bytes()==owner_artifacts['final-claimed.json']
    assert loser_writes==[]
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==owner_artifacts


def test_closed_checkpoint_tamper_refused_before_claim(tmp_path):
    a=run.parser().parse_args(['final','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    closed_synthetic(tmp_path,a)
    (tmp_path/'best.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='hash'): run.preflight_final(a)
    assert not (tmp_path/'final-claimed.json').exists()


def test_final_rejects_foreign_occupied_output(tmp_path):
    a=run.parser().parse_args(['final','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    closed_synthetic(tmp_path,a)
    (tmp_path/'foreign.txt').write_text('occupied')
    with pytest.raises(ValueError,match='occupied'): run.preflight_final(a)


@pytest.mark.parametrize('field,value',[(None,None),('completed_updates',979),('partial_update',True),
    ('player_state_sha','wrong'),('architecture','wrong')])
def test_input_metadata_with_explicit_synthetic_injection(monkeypatch,field,value):
    p,s,_=synthetic_inputs()
    sm=dict(completed_updates=980,attempted_updates=980,partial_update=False,
            player_sha=run.PLAYER_SHA,player_state_sha=run.STATE_SHA,architecture=run.ARCHITECTURE)
    if field is not None:
        sm[field]=value
    monkeypatch.setattr(run,'file_hash',lambda path:run.PLAYER_SHA if path=='p' else run.S_SHA)
    pm = p.architecture_metadata()
    monkeypatch.setattr(run,'load_player',lambda *args,**kwargs:(p,pm))
    monkeypatch.setattr(run,'load_spawner',lambda *args:(s,sm))
    if field is not None:
        with pytest.raises(ValueError,match='metadata'): run.inputs('p','s','cpu')
    else:
        # Exercise the accepted inputs branch without external files; loaders
        # and file hashes alone are explicitly synthetic, inputs() stays real.
        p.requires_grad_(False)
        s.requires_grad_(True).train()
        actual_p, actual_s, metadata = run.inputs('p','s','cpu')
        assert actual_p is p and actual_s is s
        assert metadata == pm
        assert all(x.requires_grad for x in p.parameters())
        assert not s.training and all(not x.requires_grad for x in s.parameters())


def test_failed_u0_has_no_complete_validation(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(run,'inputs',synthetic_inputs)
    monkeypatch.setattr(run,'evaluate',fake_eval)
    a=run.parser().parse_args(['train','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    with pytest.raises(ValueError): run.main(a)
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['last_complete_validation'] is None and state['validation_pending']


def test_stopped_status_cannot_use_complete_closure(tmp_path):
    a=run.parser().parse_args(['final','--source-commit',SYNTHETIC_SOURCE_COMMIT,'--out-dir',str(tmp_path)])
    closed_synthetic(tmp_path,a)
    run.json_write(tmp_path/'status.json',dict(status='stopped'))
    with pytest.raises(ValueError,match='status'): run.preflight_final(a)
