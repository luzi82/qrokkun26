"""Real historical safety checks plus behavioral calendar-isolation regressions."""
from types import SimpleNamespace
import time

import pytest
from conftest import deadline_caps
from qrokkun_ai.v22 import run, supervise


@pytest.mark.parametrize('now', [1794200000., 1794400000., 2200000000.])
def test_synthetic_caps_across_calendars(now, monkeypatch):
    original_parser=run.parser
    original_time=time.time
    with deadline_caps(now) as (hard, train):
        # A private module clock models different calendars without modifying the
        # shared time module used by other modules or subprocess timeout logic.
        monkeypatch.setattr(run, 'time', SimpleNamespace(time=lambda: now))
        a=run.parser().parse_args(['train', '--out-dir', 'synthetic'])
        assert (a.deadline_unix, a.train_deadline_unix)==(hard, train)
        assert now < train < hard
        run.Guard(train).check()
        g=run.Guard(hard+100)
        assert g.deadline==hard
        monkeypatch.setattr(run, 'time', SimpleNamespace(time=lambda: hard))
        with pytest.raises(run.Stopped, match='deadline'):
            g.check()
        assert time.time is original_time
    assert (run.HARD, run.TRAIN_HARD)==(1794322800, 1794320100)
    assert run.parser is original_parser
    a=run.parser().parse_args(['train', '--out-dir', 'synthetic'])
    assert (a.deadline_unix, a.train_deadline_unix)==(1794322800, 1794320100)


@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_historical_hard_boundary(offset, monkeypatch):
    assert run.HARD==1794322800
    assert run.TRAIN_HARD==1794320100
    monkeypatch.setattr(run, 'time', SimpleNamespace(time=lambda: 1794322800+offset))
    guard=run.Guard(run.HARD+100)
    assert guard.deadline==1794322800
    if offset < 0:
        guard.check()
    else:
        with pytest.raises(run.Stopped, match='deadline'):
            guard.check()


@pytest.mark.parametrize('kind', ['equal', 'reversed', 'train_cap', 'nan', 'inf', 'cutoff'])
def test_historical_supervisor_rejects_before_claim(kind, tmp_path, monkeypatch):
    assert (run.HARD, run.TRAIN_HARD)==(1794322800, 1794320100)
    now=1794310000
    monkeypatch.setattr(supervise, 'time', SimpleNamespace(time=lambda: now))
    a=run.parser().parse_args(['train', '--out-dir', str(tmp_path/'never')])
    if kind=='equal': a.train_deadline_unix=a.deadline_unix
    elif kind=='reversed': a.train_deadline_unix=a.deadline_unix+1
    elif kind=='train_cap': a.train_deadline_unix=run.TRAIN_HARD+1
    elif kind=='nan': a.train_deadline_unix=float('nan')
    elif kind=='inf': a.train_deadline_unix=float('inf')
    else: a.deadline_unix=now+10; a.train_deadline_unix=now
    with pytest.raises(ValueError):
        supervise.execute(a, lambda stage: pytest.fail('invalid launch'))
    assert not (tmp_path/'never').exists()
    assert not (tmp_path/'never.launcher').exists()
