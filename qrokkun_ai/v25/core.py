"""Locked identities, historical gate and validation selection."""
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict

THRESHOLD = 66.63020833333333
VALIDATION = list(range(600000, 600016))
HELDOUT = list(range(800000, 800064))
S_SHA = '554290b6c17a280dbfb8bb4076583a1230b278bf73a24a701f66fca76e5bfcc0'
P_PATH = '/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261109-1803-v21-exploratory/best_player_v21.pt'
S_PATH = '/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261111-1040-sv24-seed17-u1000/best.pt'
REFERENCE_SHA = 'af7638d6606cb4e9041f9746d73ec075c54d21ada0e8357f6ee6cb2fd083ae59'
_reference_bytes = Path(__file__).with_name('reference.json').read_bytes()
if hashlib.sha256(_reference_bytes).hexdigest() != REFERENCE_SHA:
    raise ValueError('historical reference hash mismatch')
REFERENCE = json.loads(_reference_bytes)


def historical_gate(records):
    keys = ('seed', 'frames', 'death', 'censored', 'cap_frames')
    if len(records) != 16 or any(tuple(a.get(k) for k in keys) != tuple(b[k] for k in keys)
                                 for a, b in zip(records, REFERENCE['per_seed'])):
        raise ValueError('u0 historical per-seed mismatch; no training permitted')


@dataclass
class Selection:
    best_mean: float = -1.0
    best_update: int = 0
    first_cross: int | None = None
    first_cross_by500: int | None = None

    def observe(self, update, mean):
        if mean > THRESHOLD and self.first_cross is None:
            self.first_cross = update
            if update <= 500:
                self.first_cross_by500 = update
        improved = mean > self.best_mean
        if improved:
            self.best_mean, self.best_update = mean, update
        return improved

    def report(self):
        return asdict(self)
