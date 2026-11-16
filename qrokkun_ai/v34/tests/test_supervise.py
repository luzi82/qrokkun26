"""Small lifecycle regressions with in-memory children, never training."""
import json
import signal
import subprocess
from pathlib import Path
import pytest
from qrokkun_ai.v34 import run, final, fingerprint
from qrokkun_ai.v31 import core


def args(tmp_path, nonformal=False):
    return run.parser().parse_args(['train', '--out-dir', str(tmp_path/'run'),
        '--source-commit', 'a'*40, *(['--nonformal'] if nonformal else [])])


def test_public_preflight_actual_digest_and_restore(tmp_path, monkeypatch):
    a = args(tmp_path)
    legacy = core.source_digest
    digest = fingerprint.source_digest()
    s = {k:getattr(a,k) for k in ('source_commit','updates','episodes','cap','seed','validation','hidden')}
    s.update(schema='pv34_manifest_v1', nonformal=False, source_tree_sha=digest)
    c = dict(formal=True, completed_updates=1000, latest_validated_update=1000,
             best_update=0, manifest_sha='hash', saved_checkpoints=list(range(0,1001,10)),
             checkpoint_hashes={run.ckpt_name(u):'hash' for u in (0,1000)})
    docs = {'manifest.json':dict(semantic=s), 'closed.json':c, 'status.json':dict(status='complete')}
    for u in range(0,1001,10):
        docs[f'evaluation-{u:06d}.json'] = dict(update=u,total_frames=16,
            episodes=[dict(seed=v,frames=1) for v in range(600000,600016)])
    for label,u in [('best',0),('initial',0),('terminal',1000),('latest',1000)]:
        docs[label+'.json'] = dict(update=u,checkpoint=run.ckpt_name(u))
    read_text = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p:json.dumps(docs[p.name]) if p.name in docs else read_text(p))
    monkeypatch.setattr(final,'file_hash',lambda p:'hash')
    assert final.preflight(a)[1] == c
    assert core.source_digest is legacy
    s['source_tree_sha']='wrong'
    with pytest.raises(ValueError,match='source provenance mismatch'): final.preflight(a)
    assert core.source_digest is legacy


@pytest.mark.parametrize('mode',['formal','nonformal','failure','stop','kill-timeout','closure'])
def test_supervisor_wiring_and_stop(tmp_path, monkeypatch, mode):
    from qrokkun_ai.v34 import supervise
    a=args(tmp_path,mode=='nonformal')
    calls=[]
    def preflight(*a):
        calls.append('preflight')
        if mode=='closure': raise ValueError('closure rejected')
    monkeypatch.setattr(supervise,'preflight',preflight)
    class Child:
        pid=123
        returncode=None
        def __init__(self,argv,**kw):
            calls.append(argv)
            if mode in ('stop','kill-timeout'):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        def poll(self):
            return None if mode in ('stop','kill-timeout') else (7 if mode=='failure' else 0)
        def terminate(self): calls.append('term')
        def kill(self): calls.append('kill')
        def wait(self,timeout=None):
            calls.append(('wait',timeout))
            if timeout==10 or (timeout==5 and mode=='kill-timeout'):
                raise subprocess.TimeoutExpired('fake',timeout)
            self.returncode=7 if mode=='failure' else (-9 if mode=='stop' else 0)
            return self.returncode
    monkeypatch.setattr(supervise.subprocess,'Popen',Child)
    result=supervise.execute(a,poll=0)
    state=json.loads((tmp_path/'run.launcher/supervisor.json').read_text())
    launches=[x for x in calls if isinstance(x,list)]
    assert all(x[1:3]==['-m','qrokkun_ai.v34.run'] for x in launches)
    assert [x[3] for x in launches]==(['train','final'] if mode=='formal' else ['train'])
    assert ('preflight' in calls)==(mode in ('formal','closure'))
    assert result==(0 if mode in ('formal','nonformal') else 1)
    assert not (tmp_path/'run/final-claimed.json').exists()
    if mode in ('stop','kill-timeout'):
        assert state['owner_stop']==[signal.SIGTERM]
        assert 'TERM deadline' in str(state['errors'])
        assert 'term' in calls and 'kill' in calls
        assert ('wait',10) in calls and ('wait',5) in calls
        if mode=='kill-timeout':
            assert any("TimeoutExpired('fake', 5)" in e for e in state['errors'])
            assert state['child_exits'][0]['cleanup_incomplete']
    if mode=='failure': assert 'train child exit 7' in str(state['errors'])
