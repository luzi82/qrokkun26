import json
import sys
import pytest
from qrokkun_ai.v23 import run

def test_direct_child_failure_and_exclusive(tmp_path):
    from qrokkun_ai.v23 import supervise as s
    a=run.parser().parse_args(['train','--out-dir',str(tmp_path/'run')])
    stages=[]
    def child(stage):
        stages.append(stage); return [sys.executable,'-c','raise SystemExit(7)']
    state=s.execute(a,child_argv=child,poll=.01)
    assert stages==['train'] and state['exit_code']==7 and state['stage']=='failed'
    with pytest.raises(ValueError): s.execute(a,child_argv=child,poll=.01)

def test_supervisor_signal_reaps_no_final(tmp_path):
    import os
    import signal
    import subprocess
    import time
    out=tmp_path/'signal-run'
    script="""
import sys
from qrokkun_ai.v23 import run,supervise
args=run.parser().parse_args(['train','--out-dir',sys.argv[1]])
state=supervise.execute(args,child_argv=lambda stage:[sys.executable,'-c','import time; time.sleep(60)'],poll=.01)
assert state['stage']=='stopped'
"""
    proc=subprocess.Popen([sys.executable,'-c',script,str(out)])
    statefile=out.with_name(out.name+'.launcher')/'supervisor.json'
    try:
        end=time.monotonic()+15
        while time.monotonic()<end:
            if statefile.exists():
                state=json.loads(statefile.read_text())
                if state.get('child_pid'): break
            time.sleep(.03)
        else: pytest.fail('child did not start')
        pid=state['child_pid']; proc.send_signal(signal.SIGTERM); assert proc.wait(timeout=15)==0
        state=json.loads(statefile.read_text()); assert state['stage']=='stopped' and state['child_pid'] is None
        with pytest.raises(ProcessLookupError): os.kill(pid,0)
        assert not (statefile.parent/'final.stdout').exists()
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()

@pytest.mark.parametrize('cancel', ['SIGTERM', 'SIGINT', 'deadline'])
def test_cancel_during_final_preflight_never_spawns(tmp_path, monkeypatch, cancel):
    import signal
    from qrokkun_ai.v23 import supervise as s
    a=run.parser().parse_args(['train','--out-dir',str(tmp_path/'run')])
    clock=[100.0]
    if cancel=='deadline': a.deadline_unix=101.0
    monkeypatch.setattr(s.time,'time',lambda: clock[0])
    calls=[]
    class Done:
        pid=12345
        returncode=0
        def poll(self): return 0
        def wait(self): return 0
    def popen(*args, **kwargs):
        calls.append(args)
        assert len(calls)==1, 'final Popen called after observed cancellation'
        out=tmp_path/'run'; out.mkdir()
        (out/'training-closed.json').write_text(json.dumps(dict(
            status='updates_complete',completed_updates=1000,best_sha='dummy')))
        return Done()
    def hashing(path):
        if cancel=='deadline': clock[0]=102.0
        else: signal.getsignal(getattr(signal,cancel))(getattr(signal,cancel),None)
        return 'dummy'
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    monkeypatch.setattr(run,'file_hash',hashing)
    state=s.execute(a,child_argv=lambda stage:['cpu-dummy'],poll=.01)
    assert len(calls)==1
    assert state['stage']=='stopped' and state['child_pid'] is None
    saved=json.loads((tmp_path/'run.launcher'/'supervisor.json').read_text())
    assert saved['stage']=='stopped'
    assert saved['reason']==('explicit deadline' if cancel=='deadline' else 'owner signal')
