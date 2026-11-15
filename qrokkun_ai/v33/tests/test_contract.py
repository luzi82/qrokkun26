import torch
import pytest
from qrokkun_ai.v33.initialization import random_player
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v33 import run
from qrokkun_ai.v33.protocol import schedule, better, training_seeds, paired, FINAL_SEEDS


def test_repeatability():
    a = random_player(8)
    PlayerV21(hidden=8)
    assert state_dict_sha256(a.state_dict()) == state_dict_sha256(random_player(8).state_dict())


def test_rng_isolation():
    before = torch.get_rng_state().clone()
    random_player(8)
    assert torch.equal(before, torch.get_rng_state())


def test_default_all_parameters():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        expected = PlayerV21(hidden=8)
    actual = random_player(8)
    assert all(torch.equal(v, actual.state_dict()[k]) for k,v in expected.state_dict().items())
    assert any(torch.count_nonzero(v) for k,v in actual.state_dict().items() if 'bias' in k)


def test_random_u0_gate_removed():
    assert 'historical_gate(records)' not in __import__('inspect').getsource(run.train_main)


def test_schedule():
    assert schedule() == list(range(0,1001,10))


def test_tie_earlier():
    assert not better(30,30) and better(31,30)


def test_scene_seeds():
    assert training_seeds(2) == list(range(100020,100040))


def args(*extra):
    return run.parser().parse_args(['train','--out-dir','/tmp/unused-pv33','--source-commit','a'*40,*extra])


def test_formal_config():
    assert run.validate(args())[1] == schedule()


def test_invalid_budget():
    with pytest.raises(ValueError): run.validate(args('--episodes','3'))


def test_invalid_source():
    a=args(); a.source_commit='HEAD'
    with pytest.raises(ValueError): run.validate(a)


def test_nonformal_config():
    assert run.validate(args('--nonformal','--synthetic','--updates','1','--episodes','3','--workers','2','--hidden','8','--device','cpu','--validation','2'))[1] == [0,1]


def test_nonformal_final_rejected():
    a=args('--nonformal'); a.command='final'
    with pytest.raises(ValueError): run.validate(a)


def test_paired():
    rows=[dict(seed=s,frames=120) for s in FINAL_SEEDS]
    x=paired(rows,[dict(seed=s,frames=60) for s in FINAL_SEEDS])
    assert x['mean_seconds_difference']==1 and x['ci95']==[1,1]


def test_missing_seed():
    with pytest.raises(ValueError): paired([],[])


def test_missing_reference(tmp_path):
    from qrokkun_ai.v33.reference import load_reference
    with pytest.raises((ValueError,FileNotFoundError)): load_reference(tmp_path)
