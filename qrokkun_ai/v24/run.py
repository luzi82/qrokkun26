"""Fixed-update Sv24 experiment. No resume and no default clock deadline."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import time
import subprocess
import re
import numpy as np
import torch
from .core import (
    Spawner,
    LearnedEnv,
    load_player,
    load_spawner,
    PLAYER_PATH,
    PLAYER_SHA,
    STATE_SHA,
    COMPARATOR_PATH,
    COMPARATOR_SHA,
    AIM_MAX_PX,
    STD_MIN,
    STD_MAX,
    STD_INIT,
    timed_returns,
    file_hash,
)
from qrokkun_env.gv20 import Gv20Env
from qrokkun_ai.v21.obs_v21 import encode_obs_v21


class Stopped(Exception):
    pass


class Guard:
    def __init__(self, deadline=None):
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError('finite optional deadline required')
        self.deadline = deadline
        self.reason = None

    def check(self):
        if self.reason:
            raise Stopped(self.reason)
        if self.deadline is not None and time.time() >= self.deadline:
            raise Stopped('deadline')

    def signal(self, s, *_):
        self.reason = f'signal-{s}'


def atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


@torch.no_grad()
def player_actions(player, envs, device):
    obs = [encode_obs_v21(e) for e in envs]
    p, b, m = [torch.as_tensor(np.stack(x), device=device) for x in zip(*obs)]
    return player.forward_policy(p, b, m).logits.argmax(-1).cpu().tolist()


def load_comparator(path):
    if file_hash(path) != COMPARATOR_SHA:
        raise ValueError('Sv22 comparator identity mismatch')
    from qrokkun_ai.v22.core import Spawner as OldSpawner

    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['update'] != 880 or ck['partial_update']:
        raise ValueError('Sv22 comparator update mismatch')
    net = OldSpawner()
    net.load_state_dict(ck['model'])
    return net.eval()


def episodes(net, player, seeds, cap, device, guard, mode='stochastic', train_rng=None, on_complete=None):
    traces = [[] for _ in seeds]
    envs = []
    hist = np.zeros(24, dtype=np.int64)
    count = 0
    sums = np.zeros(3)
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for i, seed in enumerate(seeds):
        guard.check()
        rng = (
            train_rng
            if train_rng is not None
            else torch.Generator().manual_seed(seed + (7000000 if mode == 'old' else 24000000))
        )

        def callback(obs, i=i, rng=rng):
            nonlocal count, sums, mins, maxs
            guard.check()
            with torch.no_grad():
                x = torch.from_numpy(obs).unsqueeze(0)
                if mode == 'old':
                    d = net.distribution(x)
                    raw = d.mean + d.stddev * torch.randn(d.mean.shape, generator=rng)
                    return raw[0].numpy()
                a, lp = net.sample(x, rng, deterministic=mode == 'deterministic')
                _, std = net.parameters_at(x)
                std = std[0].numpy()
                hist[min(23, int((a[0, 0].item() + math.pi) / (2 * math.pi) * 24))] += 1
                count += 1
                sums += std
                mins = np.minimum(mins, std)
                maxs = np.maximum(maxs, std)
            traces[i].append((envs[i].frame if i < len(envs) else 0, obs, a[0].numpy(), lp.item()))
            return a[0].numpy()

        if mode == 'scripted':
            e = Gv20Env()
        elif mode == 'old':
            from qrokkun_ai.v22.core import LearnedEnv as OldEnv

            e = OldEnv(callback)
        else:
            e = LearnedEnv(callback)
        envs.append(e)
        e.reset(seed)
    active = list(range(len(envs)))
    while active:
        guard.check()
        actions = player_actions(player, [envs[i] for i in active], device)
        for i, a in zip(active, actions):
            envs[i].step(a)
            e = envs[i]
            if (e.dead or e.frame >= cap) and on_complete:
                on_complete(
                    dict(
                        seed=seeds[i],
                        frames=e.frame,
                        seconds=e.frame / 60,
                        death=e.dead,
                        censored=not e.dead,
                        cap_frames=cap,
                    )
                )
        active = [i for i in active if not envs[i].dead and envs[i].frame < cap]
    records = [
        dict(seed=s, frames=e.frame, seconds=e.frame / 60, death=e.dead, censored=not e.dead, cap_frames=cap)
        for s, e in zip(seeds, envs)
    ]
    data = []
    for trace, e in zip(traces, envs):
        if trace:
            _, returns = timed_returns([t[0] for t in trace], e.frame)
            data.extend((t[1], t[2], t[3], g) for t, g in zip(trace, returns))
    stats = dict(
        decisions=count,
        birth_histogram=hist.tolist(),
        std_mean=(sums / count).tolist() if count else None,
        std_min=mins.tolist() if count else None,
        std_max=maxs.tolist() if count else None,
    )
    return records, data, stats


def ppo(net, opt, data, guard, epochs):
    x = torch.tensor(np.array([v[0] for v in data]), dtype=torch.float32)
    a = torch.tensor(np.array([v[1] for v in data]), dtype=torch.float64)
    old = torch.tensor([v[2] for v in data], dtype=torch.float64)
    ret = torch.tensor([v[3] for v in data], dtype=torch.float32)
    with torch.no_grad():
        adv = ret - net.value(x)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    before = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
    metrics = {}
    for _ in range(epochs):
        for ids in torch.randperm(len(x)).split(256):
            guard.check()
            ratio = (net.log_prob(x[ids], a[ids]) - old[ids]).exp()
            policy = -torch.minimum(ratio * adv[ids], ratio.clamp(0.8, 1.2) * adv[ids]).mean()
            value = (net.value(x[ids]) - ret[ids]).square().mean()
            loss = policy + 0.5 * value
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5, error_if_nonfinite=True)
            opt.step()
            metrics = dict(
                loss=loss.item(),
                policy_loss=policy.item(),
                value_loss=value.item(),
                grad_norm=grad.item(),
            )
    guard.check()
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    metrics['parameter_delta'] = (after - before).norm().item()
    return metrics


def summary(records):
    return dict(
        mean_seconds=float(np.mean([r['seconds'] for r in records])),
        deaths=sum(r['death'] for r in records),
        censors=sum(r['censored'] for r in records),
        per_seed=records,
    )


def compare(a, b, formal):
    assert [r['seed'] for r in a] == [r['seed'] for r in b]
    diff = np.array([x['seconds'] - y['seconds'] for x, y in zip(a, b)])
    rng = np.random.default_rng(17)
    means = diff[rng.integers(len(diff), size=(10000, len(diff)))].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return dict(
        mean_difference=float(diff.mean()),
        ci95=[float(lo), float(hi)],
        interpretation='evidence of learning' if formal and hi < 0 else 'inconclusive',
    )


def source_commit_literal(value):
    if not re.fullmatch(r'[0-9a-fA-F]{40}', value):
        raise argparse.ArgumentTypeError('--source-commit requires a literal 40-hex commit SHA')
    return value


def resolve_source_commit(explicit):
    if explicit is not None:
        return source_commit_literal(explicit)
    try:
        return source_commit_literal(
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.PIPE).strip()
        )
    except (OSError, subprocess.CalledProcessError, argparse.ArgumentTypeError) as e:
        raise ValueError(
            'Cannot discover source commit; supply --source-commit with the actual archive commit (40 hex)'
        ) from e


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'final'])
    p.add_argument('--player-checkpoint', default=PLAYER_PATH)
    p.add_argument('--comparator-checkpoint', default=COMPARATOR_PATH)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument(
        '--source-commit',
        type=source_commit_literal,
        help='archive source commit, literal 40 hex; otherwise discover local Git HEAD',
    )
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--episodes-per-update', type=int, default=16)
    p.add_argument('--eval-seeds', type=int, default=16)
    p.add_argument('--final-seeds', type=int, default=64)
    p.add_argument('--quick', action='store_true', help='NONFORMAL 120-frame smoke')
    p.add_argument('--nonformal', action='store_true')
    p.add_argument('--deadline-unix', type=float, default=None)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    return p


def validate_args(a):
    if min(a.updates, a.episodes_per_update, a.epochs) <= 0 or a.updates > 1000:
        raise ValueError('positive counts; updates <=1000')
    if not 1 <= a.eval_seeds <= 16 or not 1 <= a.final_seeds <= 64:
        raise ValueError('seed windows exceeded')
    if not (a.quick or a.nonformal) and (
        a.updates != 1000 or a.episodes_per_update != 16 or a.eval_seeds != 16 or a.final_seeds != 64
    ):
        raise ValueError('reduced counts require --nonformal or --quick')
    if not (a.quick or a.nonformal) and a.seed != 17:
        raise ValueError('formal seed must be 17; alternatives require --nonformal or --quick')
    if a.source_commit is not None:
        source_commit_literal(a.source_commit)
    if a.lr <= 0 or not math.isfinite(a.lr):
        raise ValueError('invalid learning rate')
    Guard(a.deadline_unix)


def main(a=None):
    a = a or parser().parse_args()
    validate_args(a)
    torch.set_num_threads(1)
    out = Path(a.out_dir)
    cap = 120 if a.quick else 7200
    formal = not (a.quick or a.nonformal)
    guard = Guard(a.deadline_unix)
    handlers = {}
    player, _ = load_player(a.player_checkpoint, device=a.device)
    oldnet = load_comparator(a.comparator_checkpoint)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    net = Spawner()
    initial = copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    rng = torch.Generator().manual_seed(a.seed)
    update = attempt = 0
    metrics = {}
    best = None
    last_error = None
    status = 'initialized'
    primary_complete = False
    if a.command == 'train':
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise ValueError('empty output required; resume unsupported')
        manifest = dict(
            config=vars(a),
            formal=formal,
            cap_frames=cap,
            player_sha=PLAYER_SHA,
            player_state_sha=STATE_SHA,
            comparator_sha=COMPARATOR_SHA,
            source_commit=resolve_source_commit(a.source_commit),
            parent_provenance='Sv24 global-scale physical policy fixing Sv23 state-dependent noise explosion and aim scale mismatch',
            protocol=dict(
                train_seed_start=2000000,
                validation_seeds=[600000, 600015],
                final_seeds=[700000, 700063],
                train_rng_seed=a.seed,
                sv24_eval_rng='episode seed +24000000',
                sv22_eval_rng='episode seed +7000000',
                field_rng='production episode:field',
                selection='minimum stochastic validation mean survival; ties keep earlier',
                gamma=1,
                gae_lambda=1,
                reward='negative elapsed frames /7200; MC to death or cap',
                entropy_coefficient=0,
            ),
            policy=dict(
                means=['birth_x', 'birth_y', 'speed_raw', 'aim_dx_raw', 'aim_dy_raw'],
                stds=['birth_rad', 'speed_px_s', 'fire_rad'],
                std_type='state-independent global parameters',
                std_min=list(STD_MIN),
                std_max=list(STD_MAX),
                std_init=list(STD_INIT),
                speed_mean='115+85*tanh(raw)',
                speed_distribution='physical Normal truncated [30,200]',
                birth='normalized vector; exact zero east; max-abs rescale; soft-damped derivative 1/hypot(r,0.25)',
                aim=f'pre-movement player + {AIM_MAX_PX}*tanh(raw) dx,dy pixels',
                fire='wrapped Normal about spawn-to-target bearing conditioned on sampled birth',
                angle_density='logsumexp images -5..5; omitted mass <4e-28 for sigma<=pi',
            ),
            architecture=net.architecture_metadata(),
            resume_supported=False,
            source_hashes={str(p): file_hash(p) for p in sorted(Path('qrokkun_ai/v24').rglob('*.py'))},
            production_hashes={
                p: file_hash(p)
                for p in [
                    'qrokkun_env/gv20.py',
                    'qrokkun_env/env.py',
                    'qrokkun_ai/v21/agents/player_v21.py',
                    'qrokkun_ai/v21/obs_v21.py',
                ]
            },
        )
        json_write(out / 'manifest.json', manifest)
    else:
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['config'] != {**vars(a), 'command': 'train'}:
            raise ValueError('final configuration mismatch')
        closed = json.loads((out / 'training-closed.json').read_text())
        if closed['completed_updates'] != 1000 or closed['status'] != 'updates_complete':
            raise ValueError('final requires 1000 completed updates and clean stop')
        if closed['best_sha'] != file_hash(out / 'best.pt') or closed['u1000_sha'] != file_hash(out / 'update-1000.pt'):
            raise ValueError('locked checkpoint mismatch')
        net, ck = load_spawner(out / 'best.pt')
        initial, _ = load_spawner(out / 'initial.pt')
        update = ck['completed_updates']
        with (out / 'final-claimed.json').open('x') as f:
            json.dump(closed, f)

    def report(state):
        json_write(
            out / ('status.json' if a.command == 'train' else 'final-status.json'),
            dict(
                status=state,
                completed_updates=update,
                attempted_updates=attempt,
                last_error=last_error,
                metrics=metrics,
                formal=formal,
                time=time.time(),
            ),
        )

    def save(name, partial=False):
        atomic(
            out / name,
            dict(
                **net.architecture_metadata(),
                model=net.state_dict(),
                optimizer=opt.state_dict(),
                completed_updates=update,
                attempted_updates=attempt,
                partial_update=partial,
                train_rng=rng.get_state(),
                torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),
                manifest=manifest,
                player_sha=PLAYER_SHA,
                player_state_sha=STATE_SHA,
                comparator_sha=COMPARATOR_SHA,
                metrics=metrics,
                best_validation_mean=best,
                resume_supported=False,
            ),
        )

    def evaluate(model, seeds, mode, label):
        done = []
        seeds = list(seeds)

        def progress(state, error=None):
            json_write(
                out / (label + '-completion.json'),
                dict(
                    status=state,
                    mode=mode,
                    completed_seeds=[r['seed'] for r in done],
                    unfinished_seeds=[s for s in seeds if s not in {r['seed'] for r in done}],
                    last_error=error,
                ),
            )

        def persist(record):
            append(out / (label + '-episodes.jsonl'), record)
            done.append(record)
            progress('running')

        progress('running')
        try:
            records, _, _ = episodes(model, player, seeds, cap, a.device, guard, mode, on_complete=persist)
            progress('complete')
            result = summary(records)
            json_write(out / (label + '.json'), result)
            return result
        except BaseException as e:
            progress('incomplete', str(e))
            raise

    try:
        for s in (signal.SIGTERM, signal.SIGINT):
            handlers[s] = signal.signal(s, guard.signal)
        if a.command == 'train':
            save('initial.pt')
            save('latest.pt')
            report('validation_controls')
            controls = {
                label: evaluate(model, range(600000, 600000 + a.eval_seeds), mode, 'validation-control-' + label)
                for label, model, mode in [
                    ('scripted', initial, 'scripted'),
                    ('initial', initial, 'stochastic'),
                    ('sv22', oldnet, 'old'),
                ]
            }
            json_write(out / 'validation-controls.json', controls)
            for attempt in range(1, a.updates + 1):
                report('rollout')
                guard.check()
                records, data, stats = episodes(
                    net,
                    player,
                    range(2000000 + (attempt - 1) * a.episodes_per_update, 2000000 + attempt * a.episodes_per_update),
                    cap,
                    a.device,
                    guard,
                    train_rng=rng,
                )
                report('ppo')
                metrics = ppo(net, opt, data, guard, a.epochs)
                update = attempt
                metrics.update(
                    rollout=summary(records),
                    distribution=stats,
                    frozen_player=dict(
                        file_sha=PLAYER_SHA,
                        state_sha=STATE_SHA,
                        update=400,
                        no_grad=all(p.grad is None and not p.requires_grad for p in player.parameters()),
                    ),
                )
                save('latest.pt')
                if update == 1000:
                    save('update-1000.pt')
                try:
                    if update == 1 or update % 10 == 0 or update == a.updates:
                        report('validation')
                        val = evaluate(
                            net, range(600000, 600000 + a.eval_seeds), 'stochastic', f'validation-{update}'
                        )
                        metrics['validation'] = val
                        if best is None or val['mean_seconds'] < best:
                            best = val['mean_seconds']
                            save('best.pt')
                        save('latest.pt')
                        if update != 1000:
                            save(f'update-{update}.pt')
                finally:
                    append(
                        out / 'progress.jsonl',
                        dict(completed_updates=update, attempted_updates=attempt, metrics=metrics, time=time.time()),
                    )
                report('trained')
                print(json.dumps(dict(completed_updates=update, **metrics)), flush=True)
            guard.check()
            status = 'updates_complete'
            save('latest.pt')
            json_write(
                out / 'training-closed.json',
                dict(
                    status=status,
                    completed_updates=update,
                    best_sha=file_hash(out / 'best.pt'),
                    u1000_sha=file_hash(out / 'update-1000.pt') if update == 1000 else None,
                ),
            )
            report(status)
        else:
            seeds = range(700000, 700000 + a.final_seeds)
            results = {}
            for label, model, mode in [
                ('learned', net, 'stochastic'),
                ('scripted', initial, 'scripted'),
                ('initial', initial, 'stochastic'),
                ('sv22', oldnet, 'old'),
            ]:
                report('final_' + label)
                results[label] = evaluate(model, seeds, mode, 'final-' + label)
            results['comparisons'] = {
                k: compare(results['learned']['per_seed'], results[k]['per_seed'], formal)
                for k in ('scripted', 'initial', 'sv22')
            }
            results['formal'] = formal
            results['conclusion'] = (
                'evidence of learning against all controls'
                if formal and all(c['ci95'][1] < 0 for c in results['comparisons'].values())
                else 'inconclusive'
            )
            json_write(out / 'final.json', results)
            primary_complete = True
            report('primary_complete_diagnostic_running')
            evaluate(net, seeds, 'deterministic', 'final-deterministic')
            report('primary_and_diagnostic_complete')
    except BaseException as e:
        last_error = f'{type(e).__name__}: {e}'
        if a.command == 'train':
            save('interrupted.pt', partial=attempt > update)
        report(
            'primary_complete_diagnostic_incomplete'
            if primary_complete
            else ('stopped' if isinstance(e, Stopped) else 'failed')
        )
        raise
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)


if __name__ == '__main__':
    main()
