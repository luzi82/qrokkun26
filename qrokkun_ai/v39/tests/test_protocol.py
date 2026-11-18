import time
import pytest
from qrokkun_ai.v39.protocol import require_coverage, expected_control_names, training_seeds, better, validate_record
from qrokkun_ai.v39.supervise import validate_deadline

def test_control_expected_schedule():
    names=expected_control_names()
    assert len([n for n in names if n.endswith('.pt')])==101
    require_coverage(dict.fromkeys(names))

def test_control_missing_expected_pin():
    pins=dict.fromkeys(expected_control_names()); pins.pop('checkpoint-000010.pt')
    with pytest.raises(ValueError,match='coverage'): require_coverage(pins)

def test_train_seed_window():
    assert training_seeds(1)==list(range(100000,100040))
    assert training_seeds(1000)[-1]==139999

def test_tie_keeps_earliest():
    assert not better(16,16) and better(17,16)

def test_cap_death_valid():
    validate_record(dict(frames=7200,cap_frames=7200,death=True,censored=False))

def test_short_survivor_invalid():
    with pytest.raises(ValueError): validate_record(dict(frames=7199,cap_frames=7200,death=False,censored=True))

def test_deadline_before_any_process(tmp_path):
    with pytest.raises(ValueError): validate_deadline(time.time()-1,tmp_path/'STOP')
    with pytest.raises(ValueError): validate_deadline(time.time()+57601,tmp_path/'STOP')

def test_stop_before_any_process(tmp_path):
    stop=tmp_path/'STOP'; stop.touch()
    with pytest.raises(InterruptedError): validate_deadline(time.time()+5000,stop)

def test_owned_group_leader_exit_descendant_killed(tmp_path):
    import sys
    from qrokkun_ai.v39.supervise import supervise_owned
    code="import os,signal,time; p=os.fork(); (signal.signal(signal.SIGTERM,signal.SIG_IGN),time.sleep(60)) if p==0 else time.sleep(.1)"
    result=supervise_owned([sys.executable,'-c',code],tmp_path/'launcher',time.time()+4000,tmp_path/'STOP',grace=.2)
    assert result['returncode']==0
    assert 9 in result['signals_attempted']
    assert not any(r['state']!='Z' for r in result['group_members'])

def test_git_free_archive_includes_content(tmp_path,monkeypatch):
    import tarfile
    from qrokkun_ai.v39 import protocol
    names=['qrokkun_ai/v25/reference.json','qrokkun_ai/v39/contract.json','qrokkun_ai/v39/world.py','qrokkun_env/gv39.py']
    paths=[]
    for i,name in enumerate(names):
        path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(str(i)); paths.append(path)
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    monkeypatch.setattr(protocol,'source_files',lambda:paths)
    target=tmp_path/'source.tar'; protocol.archive_source('a'*40,target)
    with tarfile.open(target) as archive:
        assert archive.pax_headers['qrokkun.source_commit']=='a'*40
        for p in paths: assert archive.extractfile(str(p.relative_to(tmp_path))).read()==p.read_bytes()

def test_frozen_control_is_consumed_without_resealing(tmp_path,monkeypatch):
    from qrokkun_ai.v39 import protocol
    pins=dict.fromkeys(expected_control_names(),'a'*64)
    monkeypatch.setattr(protocol,'prepare_control',lambda *args:dict(pins))
    target=tmp_path/'pins.json'; protocol.freeze_control(tmp_path,target)
    assert protocol.consume_control(tmp_path,target)==pins
    pins['checkpoint-000010.pt']='b'*64
    with pytest.raises(ValueError,match='changed'): protocol.consume_control(tmp_path,target)

def test_controller_forwards_cuda_smoke_and_deadline(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from qrokkun_ai.v39 import supervise
    out=tmp_path/'out'; launcher=tmp_path/'out.launcher'; launcher.mkdir()
    argv=[]
    class Child:
        pid=123; returncode=0
        def poll(self): return 0
    monkeypatch.setattr(supervise.subprocess,'Popen',lambda command,**kw:(argv.append(command) or Child()))
    a=SimpleNamespace(out_dir=str(out),control_dir='control',source_commit='a'*40,spawner_checkpoint='s',
        control_pins='pins',scratch=str(tmp_path),stop_path=str(tmp_path/'STOP'),device='cuda',smoke=True,
        deadline_epoch=time.time()+5000)
    assert supervise.stage_controller(a)==0
    assert len(argv)==1 and argv[0][argv[0].index('--device')+1]=='cuda'
    assert float(argv[0][argv[0].index('--deadline-epoch')+1])==a.deadline_epoch-3600
    assert '--control-pins' in argv[0] and '--scratch' in argv[0]

def test_paired_complete64_and_cap_death():
    from qrokkun_ai.v39.protocol import paired,FINAL_SEEDS
    rows=[dict(seed=seed,frames=7200,cap_frames=7200,death=True,censored=False) for seed in FINAL_SEEDS]
    for row in rows: validate_record(row)
    result=paired(rows,rows)
    assert result['mean_seconds_difference']==0 and result['ci95']==[0,0]
    assert result['bootstrap_seed']==17 and result['bootstrap_samples']==10000
    with pytest.raises(ValueError,match='64'): paired(rows[:-1],rows)

def test_old_source_closure_excludes_new_runtime(tmp_path):
    import hashlib,tarfile
    from qrokkun_ai.v37.fingerprint import source_files
    from qrokkun_ai.v39.protocol import ROOT,OLD_DIGEST
    paths=source_files()
    assert not any('/v39/' in str(p) or p.name=='gv39.py' for p in paths)
    target=tmp_path/'old.tar'
    with tarfile.open(target,'w') as archive:
        for p in paths: archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    digest=hashlib.sha256()
    with tarfile.open(target) as archive:
        for p in paths:
            name=str(p.relative_to(ROOT)); digest.update(name.encode()); digest.update(archive.extractfile(name).read())
    assert digest.hexdigest()==OLD_DIGEST
