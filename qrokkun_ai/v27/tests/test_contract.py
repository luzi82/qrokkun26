import copy
import multiprocessing as mp
import numpy as np
import pytest
import torch
from qrokkun_ai.v27 import core
from qrokkun_ai.v21.agents.player_v21 import PlayerV21
from qrokkun_ai.v24.core import Spawner


def models():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    return PlayerV21(hidden=8), Spawner().requires_grad_(False).eval()


def test_serial_workers_and_baseline():
    from qrokkun_ai.v26.training import collect as baseline
    from qrokkun_ai.v21.train import train_player_v21 as v21
    p,s=models(); q=copy.deepcopy(p)
    a,stats=core.collect(p,s,'cpu',[100000,100001],3,True)
    b,_=core.collect(q,s,'cpu',[100000,100001],3,False)
    c=baseline(q,s,'cpu',[100000,100001],3,1,None,1)
    for x,y,z in zip(a,b,c):
        for key in vars(x):
            assert np.array_equal(getattr(x,key),getattr(y,key))
            assert np.array_equal(getattr(x,key),getattr(z,key))
    assert len({r['pid'] for r in stats})==min(2,core.cpu_budget()[0])
    assert sum(r['tasks'] for r in stats)==6
    assert not mp.active_children()
    ba=v21.rollouts_to_batch_v21(a,.9995,.999,torch.device('cpu'),reward_scale=.05)
    bb=v21.rollouts_to_batch_v21(b,.9995,.999,torch.device('cpu'),reward_scale=.05)
    for net,batch in [(p,ba),(q,bb)]:
        oa,oc=core.optimizers(net)
        core.train(net,oa,oc,batch,generator=torch.Generator().manual_seed(27000017))
    assert core.weight_diff(p.state_dict(),q.state_dict())==[]


def test_worker_cancel_cleanup():
    p,s=models()
    def stop(): raise InterruptedError('owner stop')
    with pytest.raises(InterruptedError): core.collect(p,s,'cpu',[1,2],2,True,check=stop)
    assert not mp.active_children()


def test_cpu_budget(): assert core.cpu_budget()[0]>=1

def test_source_digest(): assert len(core.source_digest())==64

def test_equal_bits(): assert core.weight_diff({'x':torch.tensor([1.])},{'x':torch.tensor([1.])})==[]

def test_signed_zero(): assert core.weight_diff({'x':torch.tensor([0.])},{'x':torch.tensor([-0.])})==['x']

def test_missing_weight(): assert core.weight_diff({}, {'x':torch.tensor(1)})==['x']

def test_fresh_adam():
    p,_=models(); a,b=core.optimizers(p); assert not a.state and not b.state


def test_formal_defaults():
    from qrokkun_ai.v27.run import parser,validate
    a=parser().parse_args(['--arm','Pv27c','--out-dir','unused','--source-commit','a'*40]); validate(a)
    assert a.updates==5


def test_reject_formal_reduction():
    from qrokkun_ai.v27.run import parser,validate
    a=parser().parse_args(['--arm','Pv27w','--out-dir','unused','--source-commit','a'*40,'--cap','1'])
    with pytest.raises(ValueError): validate(a)


def test_reject_source():
    from qrokkun_ai.v27.run import parser,validate
    a=parser().parse_args(['--arm','Pv27c','--out-dir','unused','--source-commit','x'])
    with pytest.raises(ValueError): validate(a)


def test_help():
    from qrokkun_ai.v27.run import parser
    with pytest.raises(SystemExit) as e: parser().parse_args(['--help'])
    assert e.value.code==0


def test_storage_bound(): assert 0<core.estimate(8,1)<1024**3

def test_nonfinite():
    with pytest.raises(FloatingPointError): core.finite({'x':torch.tensor(float('nan'))})


def test_synthetic_nonformal():
    from qrokkun_ai.v27.run import parser,validate
    a=parser().parse_args(['--arm','Pv27c','--out-dir','unused','--source-commit','a'*40,'--synthetic'])
    with pytest.raises(ValueError): validate(a)


def test_small_sequential_run(tmp_path):
    import json
    from qrokkun_ai.v27.supervise import main
    root=tmp_path/'pair'
    code=main(['--arm','Pv27c','--out-dir',str(root),'--source-commit','a'*40,'--nonformal','--synthetic','--device','cpu','--updates','1','--episodes','1','--cap','2','--validation','1','--hidden','8'])
    assert code==0
    report=json.loads((root/'comparison.json').read_text())
    assert report['per_update_named_differences']=={'0':[],'1':[]}
    assert (root/'Pv27w'/'checkpoint-00.pt').exists()


@pytest.fixture
def failed_collection(monkeypatch, tmp_path):
    from qrokkun_ai.v27 import run
    records={}; calls=[]
    monkeypatch.setattr(run,'json_write',lambda path,value: records.__setitem__(path.name,value))
    monkeypatch.setattr(torch,'save',lambda value,path: path.touch())
    def invoke(error, stop, cli=False):
        import os
        import signal
        records.clear(); calls.clear()
        def collect(*args,**kwargs):
            calls.append(True)
            if stop: os.kill(os.getpid(),signal.SIGTERM)
            raise error
        monkeypatch.setattr(core,'collect',collect)
        out=tmp_path/str(len(list(tmp_path.iterdir())))
        argv=['--arm','Pv27w','--out-dir',str(out),'--source-commit','a'*40,
              '--nonformal','--synthetic','--device','cpu','--updates','1',
              '--episodes','1','--cap','1','--validation','1','--hidden','8']
        try:
            return (run.cli if cli else run.main)(argv)
        finally:
            assert calls==[True]
            assert records['status.json']['completed_updates']==0
            assert set(p.name for p in out.iterdir())=={'checkpoint-00.pt'}
    return invoke,records


def test_owner_stop_precedes_worker_error(failed_collection):
    import signal
    invoke,records=failed_collection
    original=signal.getsignal(signal.SIGTERM)
    for error in (EOFError('worker EOF'),RuntimeError('worker failed')):
        with pytest.raises(InterruptedError,match='owner stop'):
            invoke(error,True)
        assert records['status.json']['status']=='stopped'
        assert invoke(error,True,cli=True)==130
        assert records['status.json']['status']=='stopped'
        assert signal.getsignal(signal.SIGTERM)==original


def test_worker_error_without_stop(failed_collection):
    invoke,records=failed_collection
    assert invoke(RuntimeError('worker failed'),False,cli=True)==3
    assert records['status.json']['status']=='failed'


def test_owner_stop_during_final_source_digest(monkeypatch, tmp_path):
    import json
    import os
    import signal
    from qrokkun_ai.v27 import run
    original_digest=core.source_digest
    calls=[]
    def digest():
        calls.append(True)
        if len(calls)==2: os.kill(os.getpid(),signal.SIGTERM)
        return original_digest()
    monkeypatch.setattr(core,'source_digest',digest)
    monkeypatch.setattr(torch,'save',lambda value,path: path.touch())
    out=tmp_path/'serial'
    code=run.cli(['--arm','Pv27c','--out-dir',str(out),'--source-commit','a'*40,
                  '--nonformal','--synthetic','--device','cpu','--updates','1',
                  '--episodes','1','--cap','2','--validation','1','--hidden','8'])
    assert len(calls)==2
    assert code==130
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='stopped'
    assert status['completed_updates']==1
