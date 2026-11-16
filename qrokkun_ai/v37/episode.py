"""Minimal v31 episode copy: unchanged rollout with capture and sidecar hooks."""
from qrokkun_ai.v31.core import *
from qrokkun_ai.v31.core import _alloc_local_slot
from qrokkun_ai.v35.model import PlayerV35 as PlayerV21
from qrokkun_ai.v35.adapter import encode_obs as encode_obs_fast
from .teaching import CaptureEnv, Capture, finish
import json
from pathlib import Path
import uuid
SEARCH = False
ROOT = None

def save_sidecar(rows, ledger):
    path = Path(ROOT) / (uuid.uuid4().hex + '.npz')
    data = dict(ledger=np.array(json.dumps(ledger)))
    if rows:
        for key, values in zip(('player','bullets','pad','actions'), zip(*rows)):
            data[key] = np.stack(values)
    np.savez(path, **data)
    return path.name

def run_episode(
    p_cpu: PlayerV21,
    s_cpu: Spawner,
    seed: int,
    cap: int,
    *,
    validation: bool = False,
    stop_check: Callable[[], bool] = lambda: False,
    slot_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    episode_started = time.perf_counter()
    p_cpu.eval()
    s_cpu.eval()
    ep_seed = int(seed)
    p_gen = torch.Generator(device="cpu").manual_seed(ep_seed + P_OFFSET)
    s_gen = torch.Generator(device="cpu").manual_seed(ep_seed + (S_VAL_OFFSET if validation else S_TRAIN_OFFSET))

    env = CaptureEnv(lambda obs: sample_spawner_action(s_cpu, obs, s_gen, False), seed=ep_seed)
    env.reset(seed=ep_seed)

    buf = _alloc_local_slot(cap) if slot_arrays is None else slot_arrays
    pl_arr = buf["player"]
    bu_arr = buf["bullets"]
    pa_arr = buf["pad"]
    ac_arr = buf["actions"]
    lp_arr = buf["log_probs"]
    va_arr = buf["values"]
    rw_arr = buf["rewards"]
    dn_arr = buf["dones"]

    if SEARCH and not validation:
        env.capture = Capture(s_cpu, s_gen)
    obs = encode_obs_fast(env)
    n_frames = 0
    done = False
    elapsed = float(env.elapsed)

    for f in range(cap):
        if (f & 63) == 0 and stop_check():
            raise InterruptedError("owner stop")
        pl_arr[f] = obs[0]
        bu_arr[f] = obs[1]
        pa_arr[f] = obs[2]

        pt = torch.from_numpy(obs[0]).unsqueeze(0)
        bt = torch.from_numpy(obs[1]).unsqueeze(0)
        mt = torch.from_numpy(obs[2]).unsqueeze(0)

        if validation:
            dist = p_cpu.forward_policy(pt, bt, mt)
            act = int(dist.logits.argmax(-1).item())
            lp_val = 0.0
            v_val = 0.0
        else:
            dist, val = p_cpu(pt, bt, mt)
            act_t = torch.multinomial(dist.probs[0], 1, generator=p_gen)
            act = int(act_t.item())
            lp_val = float(dist.log_prob(act_t)[0].item())
            v_val = float(val[0].item())

        _, base, done_raw, info = env.step(act)
        done = bool(done_raw)
        elapsed = float(info.get("elapsed", env.elapsed))
        reward = float(v21.shaped_reward_v21(env, float(base), done))

        ac_arr[f] = act
        lp_arr[f] = lp_val
        va_arr[f] = v_val
        rw_arr[f] = reward
        dn_arr[f] = done
        n_frames = f + 1

        if done:
            break
        obs = encode_obs_fast(env)

    if done:
        terminated = True
        censored = False
        truncated = False
        bootstrap = 0.0
    else:
        terminated = False
        censored = True
        truncated = True
        if validation:
            bootstrap = 0.0
        else:
            pt = torch.from_numpy(obs[0]).unsqueeze(0)
            bt = torch.from_numpy(obs[1]).unsqueeze(0)
            mt = torch.from_numpy(obs[2]).unsqueeze(0)
            bootstrap = float(p_cpu.forward_value(pt, bt, mt)[0].item())

    meta = {
        "pure_rollout_s": time.perf_counter() - episode_started,
        "seed": ep_seed,
        "frames": n_frames,
        "terminated": terminated,
        "censored": censored,
        "truncated": truncated,
        "bootstrap_value": bootstrap,
        "elapsed": elapsed,
    }
    if not validation:
        rows, ledger = finish(env) if SEARCH else ([], dict(deathcount=int(done), censored=int(not done), searched=0, skippedearly=0, candidateN=0, modelcompletecaseN=0, validatedpathN=0, deathwithsuccessN=0, demoframes=0, search_s=0., replay_s=0., validate_s=0.))
        meta["teaching"] = save_sidecar(rows, ledger)
    return meta, buf
