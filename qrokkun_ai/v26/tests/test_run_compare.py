import json
import pytest
from qrokkun_ai.v26.run import parser, validate, estimate
from qrokkun_ai.v26.compare import difference


def test_formal_bounds_and_estimate():
    a = parser().parse_args(['--arm','Pv26c','--out-dir','unused','--trace-dir','trace','--source-commit','a'*40])
    validate(a)
    a.updates = 1
    with pytest.raises(ValueError): validate(a)
    a.nonformal = True
    validate(a)
    assert estimate(472,12,7200,20,16)['estimated_bytes'] > estimate(8,2,80,1,2)['estimated_bytes'] > 0


def test_difference_path():
    assert difference({'rng':{'counter':1}},{'rng':{'counter':2}})['path'] == '$.rng.counter'
    assert difference({'action':[0,1]},{'action':[0,2]})['path'] == '$.action[1]'
    assert difference({'weight':1},{'weight':1}) is None


def test_optin_real_input_metadata():
    import os
    from pathlib import Path
    import torch
    from qrokkun_ai.v25.core import P_PATH,S_PATH,S_SHA
    from qrokkun_ai.v24.core import PLAYER_SHA,file_hash
    if os.environ.get('PV25_TEST_REAL_INPUTS')!='1':pytest.skip('opt-in NAS metadata only')
    if not Path(P_PATH).exists() or not Path(S_PATH).exists():pytest.skip('NAS not mounted')
    assert file_hash(P_PATH)==PLAYER_SHA and file_hash(S_PATH)==S_SHA
    p=torch.load(P_PATH,map_location='cpu',weights_only=False)
    s=torch.load(S_PATH,map_location='cpu',weights_only=False)
    assert p['extra']['update']==400 and s['completed_updates']==980


def test_estimate_is_architecture_only(monkeypatch):
    import qrokkun_ai.v26.run as run
    def forbidden(*a,**kw):raise AssertionError('model/data access in estimate')
    monkeypatch.setattr(run,'PlayerV21',forbidden)
    monkeypatch.setattr(run,'inputs',forbidden)
    assert run.estimate()['estimated_bytes']==829905218304
    with pytest.raises(SystemExit):parser().parse_args(['--heldout'])


def test_incomplete_cli_status(tmp_path):
    from qrokkun_ai.v26.compare import main
    report=tmp_path/'report.json'
    assert main([str(tmp_path/'absent-c'),str(tmp_path/'absent-t'),'--out-report',str(report)])==3
    assert json.loads(report.read_text())['status']=='incomplete_or_invalid'
