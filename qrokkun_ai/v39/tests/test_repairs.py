import hashlib
import tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_compact_archive_retains_old_dynamic_closure(tmp_path, monkeypatch):
    from qrokkun_ai.v37.fingerprint import source_files as old_files
    from qrokkun_ai.v39 import fingerprint, protocol
    root=fingerprint.ROOT
    old=old_files()
    assert len(old)==50
    assert set(old)<=set(fingerprint.source_files())
    for p in fingerprint.source_files():
        target=tmp_path/p.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    monkeypatch.setattr(protocol,'source_files',lambda:[tmp_path/p.relative_to(root) for p in fingerprint.source_files()])
    target=tmp_path/'source.tar'
    protocol.archive_source('a'*40,target)
    with tarfile.open(target) as archive:
        for name in ('__init__.py','run.py','supervise.py'):
            assert archive.getmember('qrokkun_ai/v38/'+name)
    # Execute the unchanged old enumerator against the compact, git-free tree.
    import runpy
    compact=runpy.run_path(str(tmp_path/'qrokkun_ai/v37/fingerprint.py'))
    assert len(compact['source_files']())==50
    assert compact['source_digest']()==protocol.OLD_DIGEST


def test_final_shutdown_preserves_training_stats(tmp_path, monkeypatch):
    from qrokkun_ai.v39.adapter import EpisodePool
    from qrokkun_ai.v37 import adapter as old
    monkeypatch.setattr(old.EpisodePool,'__init__',lambda self,*a,**kw:setattr(self,'root',str(kw['root'])))
    monkeypatch.setattr(EpisodePool,'close',lambda self:None)
    def pool(root, stats_path):
        p=EpisodePool(root=root,stats_path=stats_path)
        proc=SimpleNamespace(is_alive=lambda:True,pid=101)
        p.workers_count=1; p.worker_init_info=[{'pid':101}]
        conn=SimpleNamespace(send=lambda value:None)
        p.children=[(conn,proc)]
        p._recv_one=lambda *args:('stats',[{'root':str(root),'pid':101}])
        return p
    train=pool(tmp_path/'cache',tmp_path/'workers.json')
    train.shutdown_and_stats()
    before=(tmp_path/'workers.json').read_bytes()
    final=pool(tmp_path/'final-cache',tmp_path/'final-workers.json')
    final.shutdown_and_stats()
    assert (tmp_path/'workers.json').read_bytes()==before
    assert (tmp_path/'final-workers.json').is_file()
    assert (tmp_path/'final-workers.json').read_bytes()!=before


def test_verified_json_uses_pinned_bytes_and_stop(tmp_path):
    from qrokkun_ai.v39.protocol import read_pinned_json
    path=tmp_path/'final.json'; path.write_bytes(b'{"results": {}}')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    payload=read_pinned_json(tmp_path,'final.json',{'final.json':sha})
    path.write_bytes(b'{"results": {"changed": true}}')
    assert payload=={'results':{}}
    with pytest.raises(ValueError,match='bytes'):
        read_pinned_json(tmp_path,'final.json',{'final.json':sha})
    with pytest.raises(InterruptedError):
        read_pinned_json(tmp_path,'final.json',{'final.json':sha},lambda:True)


def test_shutdown_fast_exit_and_training_coverage(monkeypatch):
    from qrokkun_ai.v39 import adapter, final
    from qrokkun_ai.v37 import run
    written={}; closed=[]
    monkeypatch.setattr(run,'json_write',lambda path,value:written.update({str(path):value}))
    p=adapter.EpisodePool.__new__(adapter.EpisodePool)
    p.stats_path='workers.json'; p.workers_count=2
    p.worker_init_info=[{'pid':101},{'pid':102}]
    p.stop_check=lambda:False
    p.children=[]
    for pid in (101,102):
        state={'alive':True}
        proc=SimpleNamespace(pid=pid,is_alive=lambda state=state:state['alive'])
        conn=SimpleNamespace(send=lambda value,state=state:state.update(alive=False),pid=pid)
        p.children.append((conn,proc))
    p._recv_one=lambda conn,*args:('stats',[{'pid':conn.pid,'tasks':1,'cpu_s':.1,'frozen_S_verified':True,'cuda_initialized':False}])
    p.close=lambda:closed.append(True)
    stats=p.shutdown_and_stats()
    assert [r['pid'] for r in stats]==[101,102]
    assert closed==[True] and written['workers.json']==stats
    semantic=dict(schema='pv39_manifest_v1',nonformal=False,source_tree_sha='sha',checkpoint_boundaries=[],validation_boundaries=[],workers=2)
    closure=dict(formal=True,completed_updates=1000,latest_validated_update=1000,manifest_sha='sha',source_tree_sha='sha',workers_stats={'path':'workers.json','sha256':'sha'},worker_pids=[101,102])
    records={'manifest.json':{'semantic':semantic},'closed.json':closure,'status.json':dict(status='complete',errors=[],owner_stop=[]),'workers.json':stats[:1]}
    monkeypatch.setattr(final,'read',lambda out,name:records[name])
    monkeypatch.setattr(final,'file_hash',lambda path:'sha')
    monkeypatch.setattr(final,'source_digest',lambda:'sha')
    monkeypatch.setattr(final,'SCHEDULE',[])
    with pytest.raises(ValueError,match='worker.*coverage'): final.closure('.')
    records['workers.json']=[stats[0],stats[0]]
    with pytest.raises(ValueError,match='worker.*coverage'): final.closure('.')


def test_shutdown_forwards_owner_stop_and_keeps_failures(monkeypatch):
    from qrokkun_ai.v39 import adapter
    from qrokkun_ai.v37 import run
    owner=[]; closed=[]
    def stop():
        owner.append('owner STOP'); return True
    monkeypatch.setattr(adapter.old.EpisodePool,'__init__',lambda self,*a,**kw:None)
    monkeypatch.setattr(run,'json_write',lambda *args:None)
    p=adapter.EpisodePool(root='unused',stats_path='unused',stop_check=stop)
    p.children=[(SimpleNamespace(send=lambda value:None),SimpleNamespace(is_alive=lambda:True,pid=101))]
    def recv(conn,proc,check):
        if check(): raise InterruptedError('owner stop; worker failed')
        return 'stats',[{'pid':101}]
    p._recv_one=recv
    def close():
        closed.append(True); raise RuntimeError('cleanup failed')
    p.close=close
    with pytest.raises(BaseExceptionGroup) as caught: p.shutdown_and_stats()
    assert owner==['owner STOP'] and closed==[True]
    assert 'owner stop; worker failed' in repr(caught.value)
    assert 'cleanup failed' in repr(caught.value)
