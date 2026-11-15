from qrokkun_ai.v32.protocol import schedule, better

def test_formal_schedule():
    assert schedule() == list(range(0, 1001, 10))

def test_tie_keeps_u0():
    assert not better(100, 100)
