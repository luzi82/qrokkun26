import copy
import json
import signal
import sys
import pytest
from qrokkun_ai.v32 import run, supervise
from qrokkun_ai.v32.protocol import training_seeds, paired, FINAL_SEEDS, closure_valid
from qrokkun_ai.v31 import core


def args(tmp_path, *extra):
    return run.parser().parse_args(['train','--out-dir',str(tmp_path/'run'),'--source-commit','a'*40,*extra])

def fixture_closure():
    return ({'semantic':dict(schema='pv32_manifest_v1',nonformal=False,updates=1000)},
            dict(formal=True,completed_updates=1000,latest_validated_update=1000),dict(status='complete'))

def test_defaults(tmp_path):
    a=args(tmp_path)
    assert run.validate(a)==(10,list(range(0,1001,10)))
    assert (a.episodes,a.workers,a.cap,a.hidden,a.seed)==(20,10,7200,472,17)

def test_training_mapping():
    assert training_seeds(1)==list(range(100000,100020))
    assert training_seeds(1000)==list(range(119980,120000))

def test_original_rng_offsets():
    assert (core.P_OFFSET,core.S_TRAIN_OFFSET,core.S_VAL_OFFSET)==(26000000,25000000,24000000)

def test_checkpoint_names():
    assert run.ckpt_name(1000)=='checkpoint-001000.pt'
    assert run.eval_name(0)=='evaluation-000000.json'

def test_validation_checkpoint_alignment(tmp_path):
    a=args(tmp_path)
    assert run.validation_schedule(a)[1]==run.validate(a)[1]

def test_archive_requires_commit(tmp_path):
    a=args(tmp_path); a.source_commit='HEAD'
    with pytest.raises(ValueError,match='40-char'): run.validate(a)

def test_reject_u10_formal(tmp_path):
    with pytest.raises(ValueError,match='1000'): run.validate(args(tmp_path,'--updates','10'))

def test_reject_formal_cpu(tmp_path):
    with pytest.raises(ValueError,match='cuda'): run.validate(args(tmp_path,'--device','cpu'))

def test_locked_smoke(tmp_path):
    a=args(tmp_path,'--nonformal','--updates','1','--episodes','3','--workers','2','--validation','2','--device','cpu')
    assert run.validate(a)==(10,[0,1])
    assert not a.synthetic and a.cap==7200

def test_small_synthetic_fixture(tmp_path):
    a=args(tmp_path,'--nonformal','--synthetic','--hidden','8','--cap','2','--updates','1','--episodes','1','--validation','1','--workers','1','--device','cpu','--pin-cores','off')
    run.validate(a)

def test_complete_closure():
    assert closure_valid(*fixture_closure())

@pytest.mark.parametrize('change',['failed','incomplete','nonformal','stop'])
def test_final_blocked(change):
    m,c,s=fixture_closure()
    if change=='failed': s['status']='failed'
    if change=='incomplete': c['latest_validated_update']=990
    if change=='nonformal': m['semantic']['nonformal']=True
    if change=='stop': s['owner_stop']=True
    assert not closure_valid(m,c,s)

def test_pairing_ci():
    base=[dict(seed=s,frames=600) for s in FINAL_SEEDS]
    other=[dict(seed=s,frames=660) for s in FINAL_SEEDS]
    result=paired(other,base)
    assert result['mean_seconds_difference']==1 and result['ci95']==[1,1]
    assert result['bootstrap_samples']==10000

def test_pairing_rejects_duplicate():
    base=[dict(seed=s,frames=600) for s in FINAL_SEEDS]
    bad=copy.deepcopy(base); bad[-1]=bad[0]
    with pytest.raises(ValueError,match='64 unique'): paired(bad,base)

def test_single_stream_backend():
    b=core.ppo_backend('cuda')
    assert b['ppo_backend']=='cuda_hoisted_single_stream'
    assert b['host_dispatch_workers']==0 and not b['cuda_streams_enabled']

def test_duplicate_supervisor_does_not_clobber(tmp_path):
    a=args(tmp_path)
    launch=tmp_path/'run.launcher'; launch.mkdir(); (launch/'supervisor.json').write_text('owner')
    with pytest.raises(FileExistsError): supervise.execute(a)
    assert (launch/'supervisor.json').read_text()=='owner'

def test_failed_child_no_final(tmp_path):
    a=args(tmp_path)
    stages=[]
    def child(stage):
        stages.append(stage)
        return [sys.executable,'-c','raise SystemExit(7)']
    assert supervise.execute(a,child_argv=child)==1
    assert stages==['train']
    status=json.loads((tmp_path/'run.launcher/supervisor.json').read_text())
    assert status['child_exits'][0]['code']==7

def test_signal_reaps_child_no_final(tmp_path):
    a=args(tmp_path)
    stages=[]
    def child(stage):
        stages.append(stage)
        return [sys.executable,'-c','import os,signal,time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)']
    assert supervise.execute(a,child_argv=child)==1
    status=json.loads((tmp_path/'run.launcher/supervisor.json').read_text())
    assert stages==['train'] and status['owner_stop']==[signal.SIGTERM]
    assert status['child_exits'][0]['code']!=0

def test_archive_preflight_without_git(tmp_path,monkeypatch):
    from qrokkun_ai.v32.final import preflight
    from qrokkun_ai.v24.core import file_hash
    a=args(tmp_path)
    out=tmp_path/'run'; out.mkdir()
    m,c,s=fixture_closure()
    m['semantic'].update(source_commit=a.source_commit,source_tree_sha=core.source_digest())
    for k in ('updates','episodes','cap','seed','validation','hidden'):
        m['semantic'][k]=getattr(a,k)
    c['best_update']=0
    c['checkpoint_hashes']={}
    for u in (0,1000):
        path=out/run.ckpt_name(u); path.write_bytes(b'fixture checkpoint; loading belongs to final')
        c['checkpoint_hashes'][path.name]=file_hash(path)
    run.json_write(out/'manifest.json',m)
    c['manifest_sha']=file_hash(out/'manifest.json')
    run.json_write(out/'closed.json',c); run.json_write(out/'status.json',s)
    # No subprocess/Git operation participates in archive provenance checks.
    monkeypatch.setattr(supervise.subprocess,'Popen',lambda *a,**kw:pytest.fail('ambient subprocess'))
    assert preflight(a)==(m,c)


def test_stop_cleanup_deadline_retains_unresolved_child(tmp_path,monkeypatch):
    stages=[]; waits=[]; actions=[]
    class Child:
        pid=43210
        returncode=None
        def poll(self): return None
        def terminate(self): actions.append('TERM')
        def kill(self): actions.append('KILL')
        def wait(self,timeout=None):
            waits.append(timeout)
            if timeout is None:
                raise AssertionError('unbounded child wait')
            raise supervise.subprocess.TimeoutExpired('fake-child',timeout)
    monkeypatch.setattr(supervise.subprocess,'Popen',lambda *a,**kw:Child())
    def owner_stop(_):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    monkeypatch.setattr(supervise.time,'sleep',owner_stop)
    def child(stage):
        stages.append(stage)
        return ['fake-child']
    assert supervise.execute(args(tmp_path),child_argv=child)==1
    status=json.loads((tmp_path/'run.launcher/supervisor.json').read_text())
    assert waits and all(t is not None and 0 < t <= 10 for t in waits)
    assert actions==['TERM','KILL'] and waits==[10,5]
    assert status['status'] not in ('complete','nonformal_complete')
    assert status['owner_stop']==[signal.SIGTERM] and stages==['train']
    assert 'TimeoutExpired' in str(status['errors'])
    assert status['child_exits']==[dict(stage='train',pid=43210,code=None,cleanup_incomplete=True)]
