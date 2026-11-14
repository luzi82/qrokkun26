"""Episode-parallel CPU rollout with SharedMemory worker pool and CUDA teacher/PPO."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import multiprocessing as mp
from multiprocessing.connection import wait as wait_conns
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qrokkun_env import constants as C
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v21.obs_v21 import (
    BULLET_FEAT_V21,
    MAX_BULLETS_V21,
    PLAYER_FEAT_V21,
    _player_feats_v21,
    compute_kinematic_expert_targets,
    order_bullets_v21,
)
from qrokkun_ai.v21.train import train_player_v21 as v21
from qrokkun_ai.v24.core import (
    LearnedEnv,
    Spawner,
    birth_angle,
    fire_mean,
    speed_sample,
    wrap,
    wrapped_sample,
)

P_OFFSET = 26_000_000
S_TRAIN_OFFSET = 25_000_000
S_VAL_OFFSET = 24_000_000
PERM_BASE = 27_000_000
FORMAL_SCHEDULES = {5: 1, 100: 10, 1000: 100}

_PAD_ROW = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)

TRAJ_FIELDS: tuple[tuple[str, tuple[int, ...], np.dtype], ...] = (
    ("player", (PLAYER_FEAT_V21,), np.dtype(np.float32)),
    ("bullets", (MAX_BULLETS_V21, BULLET_FEAT_V21), np.dtype(np.float32)),
    ("pad", (MAX_BULLETS_V21,), np.dtype(np.bool_)),
    ("actions", (), np.dtype(np.int64)),
    ("log_probs", (), np.dtype(np.float32)),
    ("values", (), np.dtype(np.float32)),
    ("rewards", (), np.dtype(np.float64)),
    ("dones", (), np.dtype(np.bool_)),
)


def configure_runtime(device: str) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "rollout": {"device": "cpu", "threads": 1, "batch": 1},
        "learner": {
            "device": device,
            "threads": int(torch.get_num_threads()),
            "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def cpu_budget() -> tuple[int, dict[str, int]]:
    limits: dict[str, int] = {"logical": os.cpu_count() or 1}
    if hasattr(os, "sched_getaffinity"):
        limits["affinity"] = max(1, len(os.sched_getaffinity(0)))
    groups = Path("/proc/self/cgroup").read_text().splitlines() if Path("/proc/self/cgroup").exists() else []
    for line in groups:
        _, controllers, relative = line.split(":", 2)
        base = Path("/sys/fs/cgroup")
        if controllers and "cpu" not in controllers.split(","):
            continue
        if controllers:
            base = base / "cpu"
        current = base / relative.lstrip("/")
        while current == base or base in current.parents:
            try:
                if (current / "cpu.max").exists():
                    quota, period = (current / "cpu.max").read_text().split()
                    if quota != "max":
                        limits[str(current / "cpu.max")] = max(1, math.floor(int(quota) / int(period)))
                elif (current / "cpu.cfs_quota_us").exists():
                    quota = int((current / "cpu.cfs_quota_us").read_text())
                    if quota > 0:
                        limits[str(current)] = max(1, quota // int((current / "cpu.cfs_period_us").read_text()))
            except (OSError, ValueError):
                pass
            if current == base:
                break
            current = current.parent
    return min(limits.values()), limits


def sorted_affinity_cpus() -> list[tuple[int, int]]:
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    ranked: list[tuple[int, int]] = []
    for cid in allowed:
        freq = 0
        freq_path = Path(f"/sys/devices/system/cpu/cpu{cid}/cpufreq/cpuinfo_max_freq")
        try:
            if freq_path.exists():
                freq = int(freq_path.read_text().strip())
        except (OSError, ValueError):
            freq = 0
        ranked.append((cid, freq))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def schedule_for(updates: int, nonformal: bool = False, checkpoint_every: int | None = None) -> tuple[int, list[int]]:
    if not nonformal:
        if updates not in FORMAL_SCHEDULES:
            raise ValueError(f"formal --updates must be one of {sorted(FORMAL_SCHEDULES)}, got {updates}")
        if checkpoint_every is not None and checkpoint_every != FORMAL_SCHEDULES[updates]:
            raise ValueError(f"formal --checkpoint-every cannot override cadence {FORMAL_SCHEDULES[updates]} for u={updates}")
        cadence = FORMAL_SCHEDULES[updates]
    else:
        if updates <= 0:
            raise ValueError(f"--updates must be positive, got {updates}")
        if checkpoint_every is not None:
            if checkpoint_every <= 0:
                raise ValueError(f"--checkpoint-every must be positive, got {checkpoint_every}")
            cadence = int(checkpoint_every)
        else:
            cadence = 1 if updates <= 10 else (10 if updates <= 100 else 100)
    boundaries = sorted({0, *range(cadence, updates + 1, cadence), int(updates)})
    return cadence, boundaries


def estimate_bytes(hidden: int, num_checkpoints: int) -> int:
    params = 2 * ((84 + 1) * hidden + (hidden + 1) * hidden) + (hidden + 1) * 10
    ckpt_bytes = params * 4 * 3 + 65536
    return ckpt_bytes * (num_checkpoints + 1) + 64 * 1024**2


@torch.no_grad()
def sample_spawner_action(
    spawner: Spawner,
    obs_np: np.ndarray,
    generator: torch.Generator,
    deterministic: bool = False,
) -> np.ndarray:
    x = torch.from_numpy(obs_np).unsqueeze(0)
    m, s = spawner.parameters_at(x)
    b = birth_angle(m[..., :2])
    b = wrap(b) if deterministic else wrapped_sample(b, s[..., 0], generator)
    speed = 115 + 85 * m[..., 2].tanh()
    if not deterministic:
        speed = speed_sample(speed, s[..., 1], generator)
    fire = fire_mean(x, m, b)
    fire = wrap(fire) if deterministic else wrapped_sample(fire, s[..., 2], generator)
    a = torch.stack([b, speed, fire], -1).detach()
    return a[0].numpy()


def encode_obs_fast(
    env: LearnedEnv, max_bullets: int = MAX_BULLETS_V21
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    player = np.asarray(_player_feats_v21(env), dtype=np.float32)
    max_b = min(max_bullets, MAX_BULLETS_V21)
    ordered = order_bullets_v21(env, max_b)
    bullets = np.empty((max_b, BULLET_FEAT_V21), dtype=np.float32)
    pad = np.ones((max_b,), dtype=np.bool_)
    n = len(ordered)
    if n:
        px, py = env.px, env.py
        fw, fh = C.FIELD_W, C.FIELD_H
        bullets[:n] = np.asarray(
            [((b.x - px) / fw, (b.y - py) / fh, b.vx / 200.0, b.vy / 200.0) for b in ordered],
            dtype=np.float32,
        )
        pad[:n] = False
    if n < max_b:
        bullets[n:] = _PAD_ROW
    return player, bullets, pad


@dataclass
class EpisodeTrajectory:
    seed: int
    frames: int
    terminated: bool
    censored: bool
    truncated: bool
    bootstrap_value: float
    elapsed: float
    player: np.ndarray
    bullets: np.ndarray
    pad: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "seed": int(self.seed),
            "frames": int(self.frames),
            "death": bool(self.terminated),
            "censored": bool(self.censored),
            "bootstrap": float(self.bootstrap_value) if self.truncated else 0.0,
            "elapsed": float(self.elapsed),
        }


def _alloc_local_slot(cap: int) -> dict[str, np.ndarray]:
    return {name: np.empty((cap, *shape), dtype=dtype) for name, shape, dtype in TRAJ_FIELDS}


@torch.no_grad()
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
    p_cpu.eval()
    s_cpu.eval()
    ep_seed = int(seed)
    p_gen = torch.Generator(device="cpu").manual_seed(ep_seed + P_OFFSET)
    s_gen = torch.Generator(device="cpu").manual_seed(ep_seed + (S_VAL_OFFSET if validation else S_TRAIN_OFFSET))

    env = LearnedEnv(lambda obs: sample_spawner_action(s_cpu, obs, s_gen, False), seed=ep_seed)
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
        "seed": ep_seed,
        "frames": n_frames,
        "terminated": terminated,
        "censored": censored,
        "truncated": truncated,
        "bootstrap_value": bootstrap,
        "elapsed": elapsed,
    }
    return meta, buf


def _slice_trajectory(meta: dict[str, Any], buf: dict[str, np.ndarray]) -> EpisodeTrajectory:
    n = int(meta["frames"])
    return EpisodeTrajectory(
        seed=int(meta["seed"]),
        frames=n,
        terminated=bool(meta["terminated"]),
        censored=bool(meta["censored"]),
        truncated=bool(meta["truncated"]),
        bootstrap_value=float(meta["bootstrap_value"]),
        elapsed=float(meta["elapsed"]),
        player=buf["player"][:n].copy(),
        bullets=buf["bullets"][:n].copy(),
        pad=buf["pad"][:n].copy(),
        actions=buf["actions"][:n].copy(),
        log_probs=buf["log_probs"][:n].copy(),
        values=buf["values"][:n].copy(),
        rewards=buf["rewards"][:n].copy(),
        dones=buf["dones"][:n].copy(),
    )


def collect_serial(
    p_cpu: PlayerV21,
    s_cpu: Spawner,
    seeds: list[int],
    cap: int,
    *,
    validation: bool = False,
    stop_check: Callable[[], bool] = lambda: False,
) -> tuple[list[EpisodeTrajectory], list[dict[str, object]]]:
    trajs: list[EpisodeTrajectory] = []
    slot = _alloc_local_slot(cap)
    for seed in seeds:
        if stop_check():
            raise InterruptedError("owner stop")
        meta, _ = run_episode(
            p_cpu, s_cpu, seed, cap, validation=validation, stop_check=stop_check, slot_arrays=slot
        )
        trajs.append(_slice_trajectory(meta, slot))
    return trajs, []


def _traj_layout(max_slots: int, cap: int) -> tuple[int, list[tuple[str, int, tuple[int, ...], np.dtype]]]:
    offset = 0
    specs: list[tuple[str, int, tuple[int, ...], np.dtype]] = []
    for name, elem_shape, dtype in TRAJ_FIELDS:
        full_shape = (max_slots, cap, *elem_shape)
        nbytes = int(np.prod(full_shape)) * dtype.itemsize
        rem = offset % 8
        if rem:
            offset += 8 - rem
        specs.append((name, offset, full_shape, dtype))
        offset += nbytes
    return max(offset, 64), specs


def _bind_traj_views(shm: SharedMemory, specs: list[tuple[str, int, tuple[int, ...], np.dtype]]) -> dict[str, np.ndarray]:
    return {
        name: np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=off)
        for name, off, shape, dtype in specs
    }


def _weight_layout(state_dict: dict[str, torch.Tensor]) -> tuple[int, list[tuple[str, int, int, tuple[int, ...], str]]]:
    offset = 0
    specs: list[tuple[str, int, int, tuple[int, ...], str]] = []
    for key in sorted(state_dict.keys()):
        t = state_dict[key].detach().cpu().contiguous()
        nbytes = t.numel() * t.element_size()
        rem = offset % 8
        if rem:
            offset += 8 - rem
        specs.append((key, offset, nbytes, tuple(t.shape), str(t.dtype).replace("torch.", "")))
        offset += nbytes
    return max(offset, 64), specs


def _pack_weights_to_shm(
    shm: SharedMemory,
    specs: list[tuple[str, int, int, tuple[int, ...], str]],
    state_dict: dict[str, torch.Tensor],
) -> None:
    assert shm.buf is not None
    mv = memoryview(shm.buf)
    for key, off, nbytes, _, _ in specs:
        raw = state_dict[key].detach().cpu().contiguous().numpy().tobytes()
        mv[off : off + nbytes] = raw


def _unpack_weights_from_shm(
    shm: SharedMemory,
    specs: list[tuple[str, int, int, tuple[int, ...], str]],
) -> dict[str, torch.Tensor]:
    assert shm.buf is not None
    out: dict[str, torch.Tensor] = {}
    for key, off, nbytes, shape, dtype_name in specs:
        np_dtype = getattr(np, dtype_name)
        arr = np.frombuffer(shm.buf, dtype=np_dtype, count=int(np.prod(shape)), offset=off).reshape(shape).copy()
        out[key] = torch.from_numpy(arr)
    return out


def _open_child_shm(name: str) -> SharedMemory:
    # Spawned children share the parent's resource_tracker fd; children only close()
    # and never unlink()/unregister(), leaving the single unregister to parent unlink().
    return SharedMemory(name=name, create=False)


def _pool_worker(
    conn,
    hidden: int,
    spawner_state: dict[str, torch.Tensor],
    weight_shm_name: str,
    weight_specs: list[tuple[str, int, int, tuple[int, ...], str]],
    traj_shm_name: str,
    traj_specs: list[tuple[str, int, tuple[int, ...], np.dtype]],
    stop_event,
    pinned_cpu: int | None,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if pinned_cpu is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {int(pinned_cpu)})
        except OSError:
            pinned_cpu = None

    w_shm = None
    t_shm = None
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    tasks = 0
    policy_version = -1

    try:
        w_shm = _open_child_shm(weight_shm_name)
        t_shm = _open_child_shm(traj_shm_name)
        views = _bind_traj_views(t_shm, traj_specs)

        p_cpu = PlayerV21(hidden=hidden).eval()
        p_cpu.requires_grad_(False)
        s_cpu = Spawner().eval()
        s_cpu.load_state_dict(spawner_state)
        s_cpu.requires_grad_(False)
        frozen_s_sha = state_dict_sha256(s_cpu.state_dict())

        conn.send(("ready", {"pid": os.getpid(), "pinned_cpu": pinned_cpu}))

        with torch.no_grad():
            while True:
                msg = conn.recv()
                if msg is None:
                    if state_dict_sha256(s_cpu.state_dict()) != frozen_s_sha:
                        raise RuntimeError("worker Spawner state mutated")
                    conn.send(
                        (
                            "stats",
                            {
                                "pid": os.getpid(),
                                "tasks": tasks,
                                "cpu_s": time.process_time() - started_cpu,
                                "wall_s": time.perf_counter() - started_wall,
                                "pinned_cpu": pinned_cpu,
                                "frozen_S_verified": True,
                                "cuda_initialized": bool(torch.cuda.is_initialized()),
                            },
                        )
                    )
                    break

                cmd = msg[0]
                if cmd == "sync":
                    _, ver, expected_sha = msg
                    sd = _unpack_weights_from_shm(w_shm, weight_specs)
                    p_cpu.load_state_dict(sd, strict=True)
                    p_cpu.eval()
                    actual_sha = state_dict_sha256(p_cpu.state_dict())
                    if actual_sha != expected_sha:
                        raise RuntimeError(f"worker weight sync hash mismatch: {actual_sha} != {expected_sha}")
                    policy_version = int(ver)
                    conn.send(("synced", policy_version))
                elif cmd == "run":
                    _, ver, slot_idx, seed, cap, validation = msg
                    if int(ver) != policy_version:
                        raise RuntimeError(f"worker stale policy_version {policy_version} != {ver}")
                    slot_buf = {k: arr[slot_idx, :cap] for k, arr in views.items()}
                    meta, _ = run_episode(
                        p_cpu,
                        s_cpu,
                        int(seed),
                        int(cap),
                        validation=bool(validation),
                        stop_check=stop_event.is_set,
                        slot_arrays=slot_buf,
                    )
                    tasks += 1
                    conn.send(("done", int(ver), int(slot_idx), meta))
                else:
                    raise RuntimeError(f"unknown worker command: {cmd!r}")
    except InterruptedError:
        try:
            conn.send(("stopped", "owner stop"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if w_shm is not None:
            try:
                w_shm.close()
            except Exception:
                pass
        if t_shm is not None:
            try:
                t_shm.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class EpisodePool:
    def __init__(
        self,
        p_template: PlayerV21,
        s_cpu: Spawner,
        max_slots: int,
        max_cap: int,
        workers: int,
        pin_cores: str = "auto",
    ) -> None:
        if workers <= 0:
            raise ValueError(f"EpisodePool requires workers >= 1, got {workers}")
        self.max_slots = int(max_slots)
        self.max_cap = int(max_cap)
        self.workers_count = int(workers)
        self.policy_version = -1
        self._closed = False

        ctx = mp.get_context("spawn")
        self.stop_event = ctx.Event()

        template_sd = {k: v.detach().cpu() for k, v in p_template.state_dict().items()}
        w_bytes, self.weight_specs = _weight_layout(template_sd)
        t_bytes, self.traj_specs = _traj_layout(self.max_slots, self.max_cap)

        self.w_shm: SharedMemory | None = None
        self.t_shm: SharedMemory | None = None
        self.children: list[tuple[Any, Any]] = []
        self.worker_init_info: list[dict[str, Any]] = []

        ranked_cpus = sorted_affinity_cpus()
        if pin_cores not in ("auto", "on", "off"):
            raise ValueError(f"invalid pin_cores={pin_cores!r}")
        do_pin = pin_cores == "on" or (pin_cores == "auto" and self.workers_count <= len(ranked_cpus))
        self.pinned_cpus: list[int | None] = [
            ranked_cpus[k % len(ranked_cpus)][0] if (do_pin and ranked_cpus) else None
            for k in range(self.workers_count)
        ]

        spawner_sd = {k: v.detach().cpu() for k, v in s_cpu.state_dict().items()}
        try:
            self.w_shm = SharedMemory(create=True, size=w_bytes)
            self.t_shm = SharedMemory(create=True, size=t_bytes)
            self.views = _bind_traj_views(self.t_shm, self.traj_specs)

            for k in range(self.workers_count):
                parent, child = ctx.Pipe()
                proc = ctx.Process(
                    target=_pool_worker,
                    args=(
                        child,
                        int(p_template.hidden),
                        spawner_sd,
                        self.w_shm.name,
                        self.weight_specs,
                        self.t_shm.name,
                        self.traj_specs,
                        self.stop_event,
                        self.pinned_cpus[k],
                    ),
                )
                try:
                    proc.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.children.append((parent, proc))

            for conn, proc in self.children:
                kind, payload = self._recv_one(conn, proc, lambda: False)
                if kind != "ready":
                    raise RuntimeError(f"unexpected worker handshake {kind}: {payload}")
                self.worker_init_info.append(payload[0])
        except BaseException:
            self.close()
            raise

    def _recv_one(self, conn, proc, stop_check: Callable[[], bool]):
        while not conn.poll(0.05):
            if stop_check():
                self.stop_event.set()
                raise InterruptedError("owner stop")
            if not proc.is_alive():
                raise RuntimeError(f"worker pid={proc.pid} exited with code {proc.exitcode}")
        try:
            msg = conn.recv()
        except EOFError as exc:
            if stop_check() or self.stop_event.is_set():
                raise InterruptedError("owner stop") from exc
            raise RuntimeError(f"worker pid={proc.pid} pipe closed unexpectedly (exitcode={proc.exitcode})") from exc
        kind = msg[0]
        if kind == "stopped":
            raise InterruptedError("owner stop")
        if kind == "error":
            if stop_check() or self.stop_event.is_set():
                raise InterruptedError("owner stop")
            raise RuntimeError(f"worker pid={proc.pid} failed:\n{msg[1]}")
        return msg[0], msg[1:]

    def sync_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        policy_version: int,
        stop_check: Callable[[], bool] = lambda: False,
    ) -> str:
        if stop_check():
            self.stop_event.set()
            raise InterruptedError("owner stop")
        assert self.w_shm is not None
        cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
        sha = state_dict_sha256(cpu_sd)
        _pack_weights_to_shm(self.w_shm, self.weight_specs, cpu_sd)
        for conn, _ in self.children:
            conn.send(("sync", int(policy_version), sha))
        for conn, proc in self.children:
            kind, rest = self._recv_one(conn, proc, stop_check)
            if kind != "synced" or int(rest[0]) != int(policy_version):
                raise RuntimeError(f"worker sync mismatch: {kind} {rest}")
        self.policy_version = int(policy_version)
        return sha

    def collect(
        self,
        seeds: list[int],
        cap: int,
        *,
        validation: bool = False,
        stop_check: Callable[[], bool] = lambda: False,
    ) -> list[EpisodeTrajectory]:
        n_eps = len(seeds)
        if n_eps > self.max_slots:
            raise ValueError(f"len(seeds)={n_eps} exceeds max_slots={self.max_slots}")
        if cap > self.max_cap:
            raise ValueError(f"cap={cap} exceeds max_cap={self.max_cap}")
        if self.policy_version < 0:
            raise RuntimeError("EpisodePool.sync_weights must be called before collect")

        ver = self.policy_version
        pending = list(enumerate(seeds))
        next_idx = 0
        inflight: dict[Any, tuple[Any, int]] = {}
        metas: dict[int, dict[str, Any]] = {}

        for conn, proc in self.children:
            if next_idx >= n_eps:
                break
            slot_idx, seed = pending[next_idx]
            next_idx += 1
            conn.send(("run", ver, slot_idx, int(seed), int(cap), bool(validation)))
            inflight[conn] = (proc, slot_idx)

        while inflight:
            if stop_check():
                self.stop_event.set()
                raise InterruptedError("owner stop")
            ready: list[Any] = list(wait_conns(list(inflight.keys()), timeout=0.05))
            if not ready:
                for conn, (proc, _) in inflight.items():
                    if not proc.is_alive():
                        raise RuntimeError(f"worker pid={proc.pid} exited with code {proc.exitcode}")
                continue
            for conn in ready:
                proc, expected_slot = inflight.pop(conn)
                kind, rest = self._recv_one(conn, proc, stop_check)
                if kind != "done":
                    raise RuntimeError(f"unexpected worker message during collect: {kind} {rest}")
                got_ver, got_slot, meta = rest
                if int(got_ver) != ver or int(got_slot) != expected_slot or int(meta["seed"]) != int(seeds[expected_slot]):
                    raise RuntimeError("worker result identity mismatch")
                if got_slot in metas:
                    raise RuntimeError(f"duplicate slot {got_slot}")
                metas[int(got_slot)] = meta
                if next_idx < n_eps:
                    slot_idx, seed = pending[next_idx]
                    next_idx += 1
                    conn.send(("run", ver, slot_idx, int(seed), int(cap), bool(validation)))
                    inflight[conn] = (proc, slot_idx)

        if len(metas) != n_eps:
            raise RuntimeError(f"incomplete episode collection: {len(metas)} != {n_eps}")

        trajs: list[EpisodeTrajectory] = []
        for slot_idx in range(n_eps):
            slot_buf = {k: arr[slot_idx, :cap] for k, arr in self.views.items()}
            trajs.append(_slice_trajectory(metas[slot_idx], slot_buf))
        return trajs

    def shutdown_and_stats(self) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        if self._closed:
            return stats
        try:
            for conn, proc in self.children:
                if proc.is_alive():
                    conn.send(None)
            for conn, proc in self.children:
                if proc.is_alive():
                    kind, rest = self._recv_one(conn, proc, lambda: False)
                    if kind == "stats":
                        stats.append(rest[0])
        finally:
            self.close()
        return stats

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        for conn, _ in self.children:
            try:
                conn.close()
            except Exception:
                pass
        for _, proc in self.children:
            try:
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2.0)
                proc.close()
            except Exception:
                pass
        for shm in (self.w_shm, self.t_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass


@torch.no_grad()
def assemble_batch(
    episodes: list[EpisodeTrajectory],
    p_dev: PlayerV21,
    device: str | torch.device,
    *,
    gamma: float = 0.9995,
    lam: float = 0.999,
    reward_scale: float = 0.05,
) -> dict[str, torch.Tensor]:
    dev = torch.device(device)
    non_empty = [ep for ep in episodes if ep.frames > 0]
    if not non_empty:
        raise ValueError("cannot assemble empty rollout batch")

    player_np = np.concatenate([ep.player for ep in non_empty], axis=0)
    bullets_np = np.concatenate([ep.bullets for ep in non_empty], axis=0)
    pad_np = np.concatenate([ep.pad for ep in non_empty], axis=0)
    actions_np = np.concatenate([ep.actions for ep in non_empty], axis=0)
    old_lp_np = np.concatenate([ep.log_probs for ep in non_empty], axis=0)
    values_np = np.concatenate([ep.values for ep in non_empty], axis=0)

    adv_parts: list[np.ndarray] = []
    ret_parts: list[np.ndarray] = []
    for ep in non_empty:
        n = ep.frames
        rews = [float(r) * reward_scale for r in ep.rewards]
        tail_val = float(ep.bootstrap_value) if ep.truncated else 0.0
        vals = [float(v) for v in ep.values] + [tail_val]
        dones = ep.dones
        running = 0.0
        adv_list = [0.0] * n
        for t in range(n - 1, -1, -1):
            mask = 0.0 if dones[t] else 1.0
            delta = rews[t] + gamma * vals[t + 1] * mask - vals[t]
            running = delta + gamma * lam * mask * running
            adv_list[t] = running
        adv_f32 = np.asarray(adv_list, dtype=np.float32)
        val_f32 = np.asarray(vals[:-1], dtype=np.float32)
        adv_parts.append(adv_f32)
        ret_parts.append(adv_f32 + val_f32)

    player_t = torch.from_numpy(player_np).to(device=dev, dtype=torch.float32)
    bullets_t = torch.from_numpy(bullets_np).to(device=dev, dtype=torch.float32)
    pad_t = torch.from_numpy(pad_np).to(device=dev, dtype=torch.bool)
    actions_t = torch.from_numpy(actions_np).to(device=dev, dtype=torch.int64)
    old_lp_t = torch.from_numpy(old_lp_np).to(device=dev, dtype=torch.float32)
    values_t = torch.from_numpy(values_np).to(device=dev, dtype=torch.float32)
    adv_t = torch.from_numpy(np.concatenate(adv_parts, axis=0)).to(device=dev, dtype=torch.float32)
    ret_t = torch.from_numpy(np.concatenate(ret_parts, axis=0)).to(device=dev, dtype=torch.float32)

    exp_acts_t, exp_logits_t = compute_kinematic_expert_targets(
        player_t,
        bullets_t,
        pad_t,
        horizon=p_dev.lookahead_horizon,
        top_k=p_dev.top_k,
    )

    return {
        "player": player_t,
        "bullets": bullets_t,
        "pad": pad_t,
        "actions": actions_t,
        "old_log_probs": old_lp_t,
        "values": values_t,
        "advantages": adv_t,
        "returns": ret_t,
        "expert_actions": exp_acts_t.to(dtype=torch.int64),
        "expert_logits": exp_logits_t.to(dtype=torch.float32),
    }


def optimizers(p: PlayerV21) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    a = torch.optim.Adam(p.actor_parameters(), lr=3e-4, eps=1e-8)
    c = torch.optim.Adam(p.critic_parameters(), lr=1e-3, eps=1e-8)
    assert not a.state and not c.state
    return a, c


def restore_optimizer(opt: torch.optim.Adam, saved_state: dict, device: str | torch.device) -> None:
    dev = torch.device(device)
    opt.load_state_dict(saved_state)
    capturable = bool(opt.defaults.get("capturable", False))
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                if k == "step" and not capturable:
                    st[k] = v.detach().cpu()
                else:
                    st[k] = v.detach().to(dev)


def finite(state: dict[str, torch.Tensor]) -> None:
    for k, v in state.items():
        if not torch.isfinite(v).all():
            raise FloatingPointError(k)


def weight_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    return [
        k
        for k in sorted(a.keys() | b.keys())
        if k not in a
        or k not in b
        or a[k].shape != b[k].shape
        or a[k].dtype != b[k].dtype
        or not torch.equal(
            a[k].detach().cpu().contiguous().view(torch.uint8),
            b[k].detach().cpu().contiguous().view(torch.uint8),
        )
    ]


def source_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for path in sorted([*root.glob("qrokkun_ai/**/*.py"), *root.glob("qrokkun_env/**/*.py")]):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def train(
    net: PlayerV21,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    generator: torch.Generator,
    stop_check: Callable[[], bool] = lambda: False,
    dagger_weight: float = 1.0,
    ppo_weight: float = 0.3,
    entropy_coef: float = 0.005,
    minibatch: int = v21.PPO_MINIBATCH,
) -> dict[str, float]:
    net.train()
    player, bullets, pad = batch["player"], batch["bullets"], batch["pad"]
    actions, old_lp = batch["actions"], batch["old_log_probs"]
    old_values, returns = batch["values"], batch["returns"]
    raw_adv = batch["advantages"]
    advantages = (raw_adv - raw_adv.mean()) / (raw_adv.std(unbiased=False) + 1e-8)
    exp_acts = batch["expert_actions"]
    exp_logits = batch["expert_logits"]

    actor_params = net.actor_parameters()
    critic_params = net.critic_parameters()

    n_samples = int(player.shape[0])
    pl_sum, vl_sum, dl_sum, kl_sum, agr_sum, seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    for _ in range(v21.PPO_EPOCHS):
        if stop_check():
            raise InterruptedError("owner stop")
        perm = torch.randperm(n_samples, device=player.device, generator=generator)
        for start in range(0, n_samples, minibatch):
            mb = perm[start : start + minibatch]
            dist, v_pred = net(player[mb], bullets[mb], pad[mb])
            log_prob = dist.log_prob(actions[mb])
            ratio = torch.exp(log_prob - old_lp[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1.0 - v21.PPO_CLIP, 1.0 + v21.PPO_CLIP) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            hard_ce = F.cross_entropy(dist.logits, exp_acts[mb])
            t_probs = F.softmax(exp_logits[mb], dim=-1)
            s_logp = F.log_softmax(dist.logits, dim=-1)
            soft_ce = -(t_probs * s_logp).sum(-1).mean()
            dagger_loss = 0.65 * hard_ce + 0.35 * soft_ce

            actor_loss = ppo_weight * policy_loss + dagger_weight * dagger_loss - entropy_coef * entropy
            opt_actor.zero_grad()
            actor_loss.backward()
            finite({n: v.grad for n, v in net.named_parameters() if v.grad is not None})
            nn.utils.clip_grad_norm_(actor_params, v21.PPO_MAX_GRAD_NORM)
            opt_actor.step()

            value_loss = F.mse_loss(v_pred, returns[mb])
            opt_critic.zero_grad()
            value_loss.backward()
            finite({n: v.grad for n, v in net.named_parameters() if v.grad is not None})
            nn.utils.clip_grad_norm_(critic_params, v21.PPO_MAX_GRAD_NORM)
            opt_critic.step()

            with torch.no_grad():
                approx_kl = float((old_lp[mb] - log_prob).mean().item())
                agr = float((dist.logits.argmax(dim=-1) == exp_acts[mb]).float().sum().item())
            bs = int(mb.shape[0])
            pl_sum += float(policy_loss.item()) * bs
            vl_sum += float(value_loss.item()) * bs
            dl_sum += float(dagger_loss.item()) * bs
            kl_sum += approx_kl * bs
            agr_sum += agr
            seen += bs

    with torch.no_grad():
        ret_var = torch.var(returns)
        ev = float(1.0 - torch.var(returns - old_values) / (ret_var + 1e-8))

    denom = max(seen, 1)
    return {
        "policy_loss": pl_sum / denom,
        "value_loss": vl_sum / denom,
        "dagger_loss": dl_sum / denom,
        "approx_kl": kl_sum / denom,
        "expert_agreement": agr_sum / denom,
        "explained_variance": ev,
        "n_samples": n_samples,
    }
