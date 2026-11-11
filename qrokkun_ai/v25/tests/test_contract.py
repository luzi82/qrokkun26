import copy
import pytest
from qrokkun_ai.v25.core import historical_gate, REFERENCE, Selection


def test_historical_gate_checks_each_seed():
    rows = copy.deepcopy(REFERENCE['per_seed'])
    historical_gate(rows)
    rows[0]['frames'] += 1
    rows[1]['frames'] -= 1
    with pytest.raises(ValueError):
        historical_gate(rows)


def test_strict_crossing_and_best_ties():
    s = Selection()
    s.observe(0, 66.63020833333333)
    assert s.first_cross is None
    s.observe(500, 67)
    s.observe(1000, 67)
    assert (s.first_cross, s.first_cross_by500, s.best_update) == (500, 500, 500)


def test_crossing_after500_and_terminal_can_lose_to_u0():
    s=Selection()
    s.observe(0,66.63020833333333)
    s.observe(500,60)
    s.observe(510,67)
    s.observe(1000,61)
    assert s.first_cross==510 and s.first_cross_by500 is None and s.best_update==510
    t=Selection()
    t.observe(0,70); t.observe(1000,69)
    assert t.best_update==0
