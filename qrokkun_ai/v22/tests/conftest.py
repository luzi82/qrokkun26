"""Opt-in caps for synthetic CPU tests; historical cutoff tests stay unpatched.

Only run's deadline constants and parser defaults are replaced, per test. The
parser has a literal TRAIN_HARD default, so changing constants alone is unsafe.
Supervise reads run's globals and resume.preflight imports them at call time.
No shared time module is patched: child timeouts retain real wall/monotonic time.
"""
import time
from contextlib import contextmanager

import pytest
from qrokkun_ai.v22 import run


@contextmanager
def deadline_caps(now=None):
    """Derive ordered caps one hour ahead, allowing old snapshot extensions.

    Explicit now is only for isolated calendar regression tests. Historical
    safety tests do not request this context/fixture and assert real constants.
    All production validation and enforcement remains active inside the context.
    """
    base=max(time.time() if now is None else now, 1794322800)
    hard, train=base+3600, base+900
    parser=run.parser
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, 'HARD', hard)
        patch.setattr(run, 'TRAIN_HARD', train)
        def synthetic_parser():
            p=parser()
            p.set_defaults(deadline_unix=hard, train_deadline_unix=train)
            return p
        patch.setattr(run, 'parser', synthetic_parser)
        yield hard, train


@pytest.fixture
def synthetic_deadlines():
    """Explicit opt-in; teardown restores caps/defaults without touching clocks."""
    with deadline_caps() as caps:
        yield caps
