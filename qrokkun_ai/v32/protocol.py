"""Pure experiment rules, separate from the unchanged numerical backend."""
import numpy as np

FINAL_SEEDS = list(range(800000, 800064))

def schedule(updates=1000):
    return sorted({0, *range(10, updates + 1, 10), updates})

def better(total, previous):
    return total > previous

def training_seeds(update, episodes=20):
    return list(range(100000 + (update-1)*episodes, 100000 + update*episodes))

def closure_valid(manifest, closed, status):
    s = manifest['semantic']
    return (s['schema'] == 'pv32_manifest_v1' and not s['nonformal'] and
            s['updates'] == 1000 and closed['formal'] and
            closed['completed_updates'] == closed['latest_validated_update'] == 1000 and
            status['status'] == 'complete' and not status.get('owner_stop') and
            not status.get('errors'))

def paired(candidate, original):
    for rows in (candidate, original):
        if [r['seed'] for r in rows] != FINAL_SEEDS:
            raise ValueError('exact 64 unique completed seeds required')
    d = np.array([a['frames']-b['frames'] for a,b in zip(candidate, original)], dtype=float)/60
    rng = np.random.default_rng(17)
    means = np.concatenate([d[rng.integers(64, size=(500,64))].mean(1) for _ in range(20)])
    return dict(mean_seconds_difference=float(d.mean()), ci95=np.quantile(means,[.025,.975]).tolist(),
                bootstrap_samples=10000, bootstrap_seed=17, descriptive_only=True)
