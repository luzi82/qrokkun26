"""Pure physical-array beam search. No environment, policy, RNG or future inputs."""
import math
import time
import numpy as np

ACTIONS = ('idle','n','ne','e','se','s','sw','w','nw')
DIRS = ((0,0),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1))
LOW = np.array([14,26], dtype=np.float32)
HIGH = np.array([306,226], dtype=np.float32)

def predict(bullets, horizon):
    """Rows x,y,vx,vy,radius,moved, in original environment order."""
    b = np.asarray(bullets, dtype=np.float64).reshape(-1,6).copy()
    live = np.ones(len(b), dtype=bool)
    result = []
    for _ in range(horizon):
        for row in b:
            if row[5]:
                row[:2] = [float(np.float32(row[k] + float(np.float32(row[k+2]/60)))) for k in (0,1)]
            row[5] = 1
        live &= (b[:,0]>=-24)&(b[:,0]<=344)&(b[:,1]>=-24)&(b[:,1]<=264)
        result.append(b[live,:5].copy())
    return result

def clearance(positions, bullets):
    if not len(bullets): return np.full(len(positions), np.inf)
    diff = positions.astype(np.float64)[:,None,:] - bullets[None,:,:2]
    return np.min(np.sqrt(np.sum(diff*diff, axis=2)) - (4+bullets[:,4]), axis=1)

def search(state, bullets, horizon=60, width=256, top=8):
    from qrokkun_env.gv36 import Gv36Env
    started = time.perf_counter()
    mover = Gv36Env(seed=0)
    beam = [(state, [], [[state.px, state.py]])]
    depth = expanded = 0
    for known in predict(bullets, horizon):
        children = []
        for rank, (parent, path, positions) in enumerate(beam):
            for action in range(9):
                mover.restore_player_state(parent)
                mover._integrate_player(*mover._parse_action(action))
                child = mover.get_player_state()
                pos = np.array([[child.px, child.py]], dtype=np.float32)
                clear = float(clearance(pos, known)[0])
                expanded += 1
                if clear <= 0: continue
                wall = float(np.min(np.minimum(pos-LOW, HIGH-pos)))
                children.append((-clear, -wall, rank, action, mover.position_key,
                                 child, path+[action], positions+[[child.px, child.py]]))
        children.sort(key=lambda x: x[:4])
        seen = set(); beam = []
        for item in children:
            if item[4] in seen: continue
            seen.add(item[4]); beam.append(item[5:])
            if len(beam) == width: break
        if not beam: break
        depth += 1
    candidates = [dict(actions=p, positions=xy) for _, p, xy in beam[:top]] if depth == horizon else []
    return dict(candidates=candidates, depth=depth, states_expanded=expanded,
                search_s=time.perf_counter()-started, modelcompletecaseN=int(depth == horizon),
                candidateN=len(candidates))
