import sys
from qrokkun_ai.v25 import run, supervise


def test_failed_child_reaped_and_no_final(tmp_path):
    a=run.parser().parse_args(['train','--out-dir',str(tmp_path/'run')])
    stages=[]
    def argv(stage):
        stages.append(stage)
        return [sys.executable,'-c','raise SystemExit(3)']
    result=supervise.execute(a,argv,poll=.01)
    assert stages==['train']
    assert result['stage']=='failed' and result['child_pid'] is None


def test_cancel_at_final_transition_never_spawns_final(tmp_path,monkeypatch):
    import signal,os
    a=run.parser().parse_args(['train','--out-dir',str(tmp_path/'run')])
    stages=[]
    def preflight(a):
        os.kill(os.getpid(),signal.SIGTERM)
    monkeypatch.setattr(run,'preflight_final',preflight)
    def argv(stage):
        stages.append(stage)
        return [sys.executable,'-c','pass']
    result=supervise.execute(a,argv,poll=.01)
    # argv construction is harmless; the cancellation recheck must precede Popen.
    assert result['stage']=='stopped' and result['child_pid'] is None
    assert not (tmp_path/'run.launcher'/'final.stdout').stat().st_size


def test_signal_reaps_owned_child(tmp_path):
    a=run.parser().parse_args(['train','--out-dir',str(tmp_path/'run')])
    result=supervise.execute(a,lambda stage:[sys.executable,'-c',
        'import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)'],poll=.01)
    assert result['stage']=='stopped' and result['child_pid'] is None
