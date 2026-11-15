from qrokkun_ai.v30 import run


def test_formal_u10_e20_split_contract():
    a = run.parser().parse_args(['--arm', 'Pv30c', '--out-dir', 'unused', '--source-commit', 'a'*40, '--updates', '10'])
    assert a.episodes == 20
    assert a.workers == 10
    assert a.pin_cores == 'on'
    assert run.validate(a) == (1, list(range(11)))
    assert run.validation_schedule(a) == (10, [0, 10])

import json
import pytest
import torch
from qrokkun_ai.v30 import compare, core, supervise


def test_linear_reconvergence_and_terminal_verdict(tmp_path, monkeypatch):
    sem = {'updates': 10, 'checkpoint_boundaries': list(range(11))}
    for arm in compare.ARMS:
        d = tmp_path / arm
        d.mkdir()
        (d/'manifest.json').write_text(json.dumps({'arm':arm,'semantic':sem,'workers':0}))
        (d/'status.json').write_text(json.dumps({'status':'complete','completed_updates':10}))
        (d/'closed.json').write_text(json.dumps({'timings':{}}))
        for u in range(11): (d/run.ckpt_name(u)).touch()
    seen=[]
    terminal=[False]
    def load(d,u):
        seen.append((d.name,u))
        return {'actor':torch.tensor([1. if d.name=='Pv30m' and (u in (1,3) or terminal[0] and u==10) else 0.])}
    monkeypatch.setattr(compare,'_load_model_state',load)
    assert compare.compare(tmp_path)==0
    r=json.loads((tmp_path/'comparison.json').read_text())
    assert r['checked_updates']==list(range(11))
    assert r['first_diverged_update']==1
    assert r['differences_by_update']['3']==['actor']
    assert len(seen)==22
    terminal[0]=True
    assert compare.compare(tmp_path)==2


def test_mixed_cadence_rng_and_no_fake_eval(tmp_path):
    common=['--arm','Pv30c','--source-commit','a'*40,'--nonformal','--synthetic','--device','cpu','--updates','3','--episodes','1','--cap','2','--validation','1','--hidden','8','--pin-cores','off']
    for cadence in (1,3):
        d=tmp_path/str(cadence)
        assert run.main([*common,'--out-dir',str(d),'--validation-every',str(cadence)])==0
    d=tmp_path/'3'
    assert sorted(p.name for p in d.glob('evaluation*'))==['evaluation-000000.json','evaluation-000003.json']
    assert len(list(d.glob('checkpoint-*.pt')))==4
    a,b=[torch.load(tmp_path/str(c)/run.ckpt_name(3),weights_only=False) for c in (1,3)]
    assert core.weight_diff(a['model'],b['model'])==[]
    assert torch.equal(a['training_rng'],b['training_rng'])
    z=torch.load(d/run.ckpt_name(0),weights_only=False)
    assert not z['actor']['state'] and not z['critic']['state']


def test_formal_rejects_changed_budget():
    base=['--arm','Pv30m','--out-dir','unused','--source-commit','a'*40]
    for change in (['--updates','5'],['--episodes','12'],['--workers','20'],['--pin-cores','auto'],['--validation-every','1'],['--checkpoint-every','10'],['--resume']):
        with pytest.raises(ValueError): run.validate(run.parser().parse_args(base+change))


def test_serial_resolution_and_child_identity():
    a=supervise.parser().parse_args(['--out-dir','unused','--source-commit','a'*40])
    cmd=supervise._build_child_cmd(a,'Pv30m',__import__('pathlib').Path('unused'))
    assert cmd[cmd.index('--workers')+1]=='10'
    assert 'qrokkun_ai.v30.run' in cmd
    assert run.resolve_workers('Pv30c',20,16,10)==0


def test_strict_pin_rejects_small_core_spill(monkeypatch):
    from qrokkun_ai.v21.agents.player_v21 import PlayerV21
    from qrokkun_ai.v24.core import Spawner
    monkeypatch.setattr(core,'sorted_affinity_cpus',lambda:[(0,200),(1,100)])
    monkeypatch.setattr(core,'cpu_budget',lambda:(2,{}))
    with pytest.raises(ValueError,match='big cores'):
        core.EpisodePool(PlayerV21(hidden=8),Spawner(),3,2,2,'on')


def test_six_digit_checkpoint_schedule():
    assert core.schedule_for(10)==(1,list(range(11)))
    assert run.ckpt_name(10)=='checkpoint-000010.pt'


def test_supervisor_runtime_failure_is_three(tmp_path, monkeypatch):
    class FailedChild:
        def __init__(self,*args,**kwargs): pass
        def poll(self): return -9
        def wait(self): return -9
    monkeypatch.setattr(supervise.subprocess,'Popen',FailedChild)
    root=tmp_path/'pair'
    assert supervise.cli(['--out-dir',str(root),'--source-commit','a'*40])==3
    status=json.loads((root/'pair-status.json').read_text())
    assert status['status']=='failed'
    assert status['exitcodes']=={'Pv30c':-9}


def test_supervisor_sigterm_during_comparison(tmp_path, monkeypatch):
    import os
    import signal

    sem = {'updates': 1, 'checkpoint_boundaries': [0, 1]}
    for arm in compare.ARMS:
        d = tmp_path / arm
        d.mkdir()
        (d / 'manifest.json').write_text(json.dumps({'arm': arm, 'semantic': sem, 'workers': 0}))
        (d / 'status.json').write_text(json.dumps({'status': 'complete', 'completed_updates': 1}))
        (d / 'closed.json').write_text(json.dumps({'timings': {}}))
        for u in (0, 1):
            (d / run.ckpt_name(u)).touch()
    monkeypatch.setattr(supervise, '_verify_completed_arm_identity', lambda *args: None)
    loads = []

    def load(d, u):
        loads.append((d.name, u))
        os.kill(os.getpid(), signal.SIGTERM)
        return {'actor': torch.tensor([0.])}

    monkeypatch.setattr(compare, '_load_model_state', load)
    rc = supervise.cli([
        '--out-dir', str(tmp_path), '--source-commit', 'a' * 40,
        '--resume', '--nonformal', '--synthetic', '--device', 'cpu',
        '--updates', '1', '--episodes', '1', '--cap', '2',
        '--validation', '1', '--hidden', '8', '--pin-cores', 'off',
    ])
    assert rc == 130
    status = json.loads((tmp_path / 'pair-status.json').read_text())
    assert status['status'] == 'stopped'
    assert status['phase'] == 'compare'
    assert len(loads) == 1
    assert not (tmp_path / 'comparison.json').exists()
