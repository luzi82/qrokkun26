"""Synthetic seeds/children only; never execute heldout evaluation."""
import json
import sys
import time
import pytest
from qrokkun_ai.v22 import run


def test_completed_tick_survives_interrupt(monkeypatch):
    class Env:
        def __init__(self): self.frame=0; self.dead=False
        def reset(self,seed): self.limit=seed
        def step(self,a):
            self.frame+=1
            self.dead=self.frame>=self.limit
    class Guard:
        ticks=0
        def check(self):
            self.ticks+=1
            if self.ticks==5: raise run.Stopped('synthetic signal')
    monkeypatch.setattr(run,'Gv20Env',Env)
    monkeypatch.setattr(run,'player_actions',lambda p,e,d:[0]*len(e))
    completed=[]
    with pytest.raises(run.Stopped):
        run.episodes(None,None,[1,99],100,'cpu',Guard(),'scripted',on_complete=completed.append)
    assert [r['seed'] for r in completed]==[1]


def test_supervisor_api():
    from qrokkun_ai.v22 import supervise
    assert callable(supervise.execute)


def supervisor_args(tmp_path,seconds=20):
    return run.parser().parse_args(['train','--out-dir',str(tmp_path/'run'),
        '--deadline-unix',str(time.time()+seconds),'--train-deadline-unix',str(time.time()+.2)])


def dummy(stage,out,behavior='normal'):
    code='import os,time,json,signal; from pathlib import Path; '
    if behavior=='fail': code+='raise SystemExit(3)'
    elif behavior=='missing': code+='pass'
    elif behavior in ('term','ignore'):
        code+='signal.signal(signal.SIGTERM, '+('signal.SIG_IGN' if behavior=='ignore' else 'lambda *a: exit(0)')+'); time.sleep(30)'
    elif stage=='train':
        code+=f"p=Path({str(out)!r}); p.mkdir(); (p/'best.pt').write_bytes(b'test'); import hashlib; (p/'training-closed.json').write_text(json.dumps(dict(best_sha=hashlib.sha256(b'test').hexdigest())))"
    else: code+=f"Path({str(out/'synthetic-final')!r}).touch()"
    return [sys.executable,'-c',code]


def assert_reaped(state,tmp_path):
    import os
    pid=json.loads((tmp_path/'run.launcher'/'supervisor.json').read_text()).get('child_pid')
    assert pid is None


@pytest.mark.usefixtures("synthetic_deadlines")
def test_supervisor_normal_idempotent(tmp_path):
    from qrokkun_ai.v22 import supervise
    a=supervisor_args(tmp_path)
    calls=[]
    def argv(stage):
        calls.append(stage); return dummy(stage,tmp_path/'run')
    state=supervise.execute(a,argv)
    assert state['stage']=='complete' and calls==['train','final']
    assert supervise.execute(a,lambda s:pytest.fail('duplicate'))==state
    assert_reaped(state,tmp_path)


@pytest.mark.parametrize('behavior',['fail','missing'])
@pytest.mark.usefixtures("synthetic_deadlines")
def test_supervisor_no_final(tmp_path,behavior):
    from qrokkun_ai.v22 import supervise
    a=supervisor_args(tmp_path); calls=[]
    def argv(stage):
        calls.append(stage); return dummy(stage,tmp_path/'run',behavior)
    if behavior=='missing':
        with pytest.raises(FileNotFoundError): supervise.execute(a,argv)
    else: assert supervise.execute(a,argv)['stage']=='failed'
    assert calls==['train']


@pytest.mark.parametrize('behavior',['term','ignore'])
@pytest.mark.usefixtures("synthetic_deadlines")
def test_supervisor_timeout_reaps(tmp_path,behavior):
    from qrokkun_ai.v22 import supervise
    import os
    a=supervisor_args(tmp_path,10.5)
    pids=[]
    original=supervise.subprocess.Popen
    def capture(*args,**kw):
        child=original(*args,**kw); pids.append(child.pid); return child
    from unittest.mock import patch
    with patch.object(supervise.subprocess,'Popen',capture):
        state=supervise.execute(a,lambda s:dummy(s,tmp_path/'run',behavior),poll=.02)
    assert state['stage']=='stopped'
    with pytest.raises(ProcessLookupError): os.kill(pids[0],0)
    with pytest.raises(ChildProcessError): os.waitpid(pids[0],os.WNOHANG)


@pytest.mark.usefixtures("synthetic_deadlines")
def test_primary_interrupt_metadata_synthetic(tmp_path,monkeypatch):
    from test_review import setup_run,records
    out,invoke=setup_run(tmp_path,monkeypatch,('--quick','--eval-seeds','2','--final-seeds','2'))
    monkeypatch.setattr(run,'episodes',records); invoke('train')
    calls=[]
    def synthetic(*args,**kwargs):
        calls.append(args[6]); rows,data=records(*args,**kwargs)
        if len(calls)==2:
            kwargs['on_complete'](dict(rows[0],synthetic_mock=True))
            raise run.Stopped('synthetic signal')
        for row in rows: kwargs['on_complete'](dict(row,synthetic_mock=True))
        return rows,data
    monkeypatch.setattr(run,'episodes',synthetic)
    invoke('final')
    assert (out/'final-scripted.json').exists()
    assert not (out/'final.json').exists()
    status=json.loads((out/'final-status.json').read_text())
    assert status['status']=='primary_incomplete' and status['mode']=='initial'
    meta=json.loads((out/'final-initial-completion.json').read_text())
    assert len(meta['completed_seeds'])==len(meta['unfinished_seeds'])==1
    rows=[json.loads(s) for s in (out/'final-initial-episodes.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['synthetic_mock']
    with pytest.raises(FileExistsError): invoke('final')


@pytest.mark.usefixtures("synthetic_deadlines")
def test_supervisor_signal_cleanup(tmp_path):
    import os
    import signal
    import subprocess
    code='''
import sys,time
from qrokkun_ai.v22 import run,supervise
# Test-only child bootstrap: caps only; all enforcement and clocks stay real.
run.HARD=time.time()+3600
run.TRAIN_HARD=run.HARD-2700
a=run.parser().parse_args(['train','--out-dir',sys.argv[1],'--deadline-unix',str(time.time()+60),'--train-deadline-unix',str(time.time()+20)])
supervise.execute(a,lambda stage:[sys.executable,'-c','import time; time.sleep(60)'])
'''
    parent=subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'run')])
    child_pid=None
    try:
        until=time.monotonic()+15
        while time.monotonic()<until:
            try:
                state=json.loads((tmp_path/'run.launcher'/'supervisor.json').read_text())
                child_pid=state['child_pid']
                if child_pid: break
            except (FileNotFoundError,json.JSONDecodeError): pass
            time.sleep(.05)
        assert child_pid
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=8)==0
        with pytest.raises(ProcessLookupError): os.kill(child_pid,0)
        assert json.loads((tmp_path/'run.launcher'/'supervisor.json').read_text())['stage']=='stopped'
    finally:
        if parent.poll() is None: parent.kill(); parent.wait()


@pytest.mark.parametrize('kind',['nan','inf','later','occupied'])
@pytest.mark.usefixtures("synthetic_deadlines")
def test_supervisor_fail_closed(tmp_path,kind):
    from qrokkun_ai.v22 import supervise
    a=supervisor_args(tmp_path)
    if kind=='occupied':
        p=tmp_path/'run'; p.mkdir(); (p/'keep').touch()
    else: a.deadline_unix={'nan':float('nan'),'inf':float('inf'),'later':run.HARD+1}[kind]
    with pytest.raises(ValueError, match='occupied output' if kind=='occupied' else 'finite deadlines|owner cutoff'):
        supervise.execute(a,lambda s:pytest.fail('must not start'))


@pytest.mark.parametrize('cause',['signal','hard'])
def test_real_guard_tick_persists_completed(tmp_path,monkeypatch,cause):
    import signal
    now=[run.HARD-60]
    monkeypatch.setattr(run.time,'time',lambda:now[0])
    guard=run.Guard(run.HARD)
    class Env:
        def __init__(self): self.frame=0; self.dead=False
        def reset(self,seed): self.limit=seed
        def step(self,a):
            self.frame+=1; self.dead=self.frame>=self.limit
            if self.limit==99 and self.frame==2:
                if cause=='signal': guard.signal(signal.SIGTERM)
                else: now[0]=run.HARD
    monkeypatch.setattr(run,'Gv20Env',Env)
    monkeypatch.setattr(run,'player_actions',lambda p,e,d:[0]*len(e))
    path=tmp_path/'synthetic.jsonl'
    def persist(row):
        import os
        with path.open('a') as f:
            f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
    with pytest.raises(run.Stopped,match='signal-15' if cause=='signal' else 'deadline'):
        run.episodes(None,None,[1,99],100,'cpu',guard,'scripted',on_complete=persist)
    assert [json.loads(s)['seed'] for s in path.read_text().splitlines()]==[1]
