"""Consolidated validation, causal-priority, weight-verdict, and alignment contracts."""
from collections import Counter
import copy
import gzip
import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from qrokkun_ai.v21.agents.player_v21 import PlayerV21
from qrokkun_ai.v26.compare import _compare, cli as compare_cli, compare, difference
from qrokkun_ai.v26.run import main as run_main, parser, source_digest, validate
from qrokkun_ai.v26.trace import Reader, Writer, digest, json_write
from qrokkun_ai.v26.training import capture_forward
from qrokkun_ai.v26.validation import Scientific, schema, snapshot, tensor


@pytest.fixture(scope='module')
def tiny_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp('pv26-tiny-pair')
    for arm in ('Pv26c', 'Pv26t'):
        run_main([
            '--arm', arm,
            '--out-dir', str(root / arm),
            '--trace-dir', str(root / (arm + '-trace')),
            '--source-commit', 'a' * 40,
            '--nonformal',
            '--synthetic',
            '--device', 'cpu',
            '--updates', '1',
            '--episodes', '2',
            '--cap', '2',
            '--validation', '2',
            '--hidden', '8',
        ])
    return root


def reseal(root, dest, arm, mutation):
    out = dest / arm
    trace = dest / (arm + '-trace')
    shutil.copytree(root / arm, out)
    shutil.copytree(root / (arm + '-trace'), trace)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['trace_dir'] = str(trace)
    json_write(out / 'manifest.json', manifest)
    closure = json.loads((trace / 'closed.json').read_text())
    counts = Counter()
    for shard in closure['shards']:
        path = trace / shard['name']
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        mutation(rows)
        for row in rows:
            if row['kind'] == 'checkpoint':
                json_write(out / f"checkpoint-{row['key'][0]:02d}.json", row['data'])
        with gzip.open(path, 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
                counts[row['kind']] += 1
        shard['sha'] = digest(path.read_bytes())
    closure.update(events=sum(counts.values()), counts=dict(counts))
    json_write(trace / 'closed.json', closure)
    closed = json.loads((out / 'closed.json').read_text())
    closed.update(
        manifest_sha=digest((out / 'manifest.json').read_bytes()),
        trace_closure_sha=digest((trace / 'closed.json').read_bytes()),
        checkpoints={p.name: digest(p.read_bytes()) for p in out.glob('checkpoint-*.json')},
    )
    json_write(out / 'closed.json', closed)
    return out


def test_positive_tiny_pair(tiny_pair):
    report = compare(tiny_pair / 'Pv26c', tiny_pair / 'Pv26t')
    assert report['status'] == 'complete_equal'
    assert report['weights_bitwise_equal'] and report['full_trace_equal']


def test_schema_and_cuda_static_contracts(tiny_pair):
    args = parser().parse_args([
        '--arm', 'Pv26c',
        '--out-dir', 'unused',
        '--trace-dir', 'unused-trace',
        '--source-commit', 'a' * 40,
        '--device', 'cpu',
    ])
    with pytest.raises(ValueError, match='CUDA'):
        validate(args)
    for device in ('cuda', 'cuda:0'):
        args.device = device
        validate(args)
    args.device = 'cpu'
    args.nonformal = True
    validate(args)

    base = json.loads((tiny_pair / 'Pv26c' / 'manifest.json').read_text())
    schema(base['semantic'], base)

    def bad_schema(mutator, match=None):
        m = copy.deepcopy(base)
        mutator(m['semantic'], m)
        with pytest.raises((ValueError, KeyError), match=match):
            schema(m['semantic'], m)

    bad_schema(lambda s, m: s.__setitem__('source_tree_sha', '0' * 64), 'digest')
    bad_schema(lambda s, m: s.__setitem__('resume', True), 'resume')
    bad_schema(lambda s, m: m.__setitem__('schema', 1), 'schema')
    bad_schema(lambda s, m: s.__setitem__('nonformal', False), 'CUDA')
    bad_schema(lambda s, m: s.__setitem__('player_state_sha', 'bad'), 'input identity')
    bad_schema(lambda s, m: s.__setitem__('spawner_state_sha', 'bad'), 'input identity')
    bad_schema(lambda s, m: s['runtime'].pop('threads'))


def test_tensor_snapshot_and_scientific_in_memory(tiny_pair):
    manifest = json.loads((tiny_pair / 'Pv26c' / 'manifest.json').read_text())
    template = schema(manifest['semantic'], manifest)
    reader = Reader(tiny_pair / 'Pv26c-trace')
    try:
        rows = list(reader.events())
        p_row = next(r for r in rows if r['kind'] == 'P')
        valid_ref = p_row['data']['probs']
        for bad_ref in (None, {}, dict(valid_ref, shape=[999])):
            with pytest.raises(ValueError):
                tensor(reader, bad_ref, valid_ref['shape'])

        ck0 = next(r for r in rows if r['kind'] == 'checkpoint' and r['key'] == [0])
        preclip = next(r for r in rows if r['kind'] == 'actor_preclip')
        poststep = next(r for r in rows if r['kind'] == 'critic_poststep')

        d = copy.deepcopy(ck0['data'])
        d['gradients'].pop('policy_head.bias')
        with pytest.raises(ValueError, match='gradient'):
            snapshot(reader, d, template, 'checkpoint', [0])

        d = copy.deepcopy(ck0['data'])
        d['gradients']['policy_head.bias'] = preclip['data']['gradients']['policy_head.bias']
        with pytest.raises(ValueError, match='None'):
            snapshot(reader, d, template, 'checkpoint', [0])

        d = copy.deepcopy(ck0['data'])
        d['critic']['state'] = {'0': {}}
        with pytest.raises(ValueError, match='empty'):
            snapshot(reader, d, template, 'checkpoint', [0])

        d = copy.deepcopy(poststep['data'])
        d['critic']['state']['0'].pop('exp_avg')
        with pytest.raises(ValueError, match='Adam'):
            snapshot(reader, d, template, 'critic_poststep', poststep['key'])

        d = copy.deepcopy(preclip['data'])
        assert d['actor']['param_groups'][0]['maximize'] is False
        d['actor']['param_groups'][0]['maximize'] = 0
        with pytest.raises(ValueError, match='Adam'):
            snapshot(reader, d, template, 'actor_preclip', preclip['key'])

        init_data = next(r for r in rows if r['kind'] == 'global_init')['data']
        cuda_sem = copy.deepcopy(manifest['semantic'])
        cuda_sem['runtime']['device'] = 'cuda'
        with pytest.raises(ValueError, match='CUDA'):
            Scientific(reader, cuda_sem, template).check('global_init', [0], init_data)

        for field_mutation in (
            lambda r: r['data'].__setitem__('raw_logits', None),
            lambda r: r['data']['losses'].pop('entropy'),
        ):
            sci = Scientific(reader, manifest['semantic'], template)
            with pytest.raises((ValueError, KeyError)):
                for row in rows:
                    target = copy.deepcopy(row)
                    if target['kind'] == 'actor_preclip':
                        field_mutation(target)
                    sci.check(target['kind'], target['key'], target['data'])
    finally:
        reader.db.close()


def test_causal_difference_priority():
    a = dict(draws=[dict(pre=0, output=0, post=0)], player=0, features=0, raw_logits=0, probs=0, actions=0)
    b = dict(draws=[dict(pre=0, output=1, post=1)], player=0, features=0, raw_logits=1, probs=1, actions=1)
    assert difference(a, b)['path'] == '$.raw_logits'
    b['draws'][0]['pre'] = 1
    assert difference(a, b)['path'] == '$.draws[0].pre'
    b = dict(a, draws=[dict(pre=0, output=1, post=1)])
    assert difference(a, b)['path'] == '$.draws[0].output'

    sa = dict(before=0, events=[dict(type='S', observation=0, means=0, std=0, action=0, logp=0)], action=0, after=0)
    sb = dict(before=0, events=[dict(type='S', observation=0, means=1, std=1, action=1, logp=1)], action=1, after=1)
    assert difference(sa, sb)['path'] == '$.events[0].means'


def test_forward_hook_cleanup_on_exception():
    net = PlayerV21(hidden=8)
    before = (len(net.policy_body._forward_pre_hooks), len(net.policy_head._forward_hooks))
    with pytest.raises(RuntimeError, match='forward failed'):
        with capture_forward(net):
            raise RuntimeError('forward failed')
    assert before == (len(net.policy_body._forward_pre_hooks), len(net.policy_head._forward_hooks))


def test_per_update_weights_and_diagnostic_independence(tiny_pair, tmp_path):
    bundles = []
    for label, workers in [('Pv26c', 1), ('Pv26t', 2)]:
        out = tmp_path / label
        out.mkdir()
        trace = tmp_path / (label + '-trace')
        w = Writer(trace, reserve=0)
        for u in range(21):
            weight = torch.tensor([1.0 if label == 'Pv26t' and u == 3 else 0.0])
            json_write(
                out / f'checkpoint-{u:02d}.json',
                dict(model={'policy_head.bias': w.encode(weight), 'value_head.bias': w.encode(weight)}),
            )
        w.close()
        rows = [dict(kind='checkpoint', key=[0], data=dict(rng=workers))]
        bundles.append(SimpleNamespace(
            path=out,
            m=dict(arm=label, workers=workers),
            s=dict(updates=20, nonformal=True, source_tree_sha=source_digest()),
            reader=Reader(trace),
            events=lambda rows=rows: iter(rows),
        ))
    try:
        report = _compare(*bundles)
        assert report['status'] == 'complete_equal' and report['weights_bitwise_equal']
        assert not report['full_trace_equal'] and not report['all_updates_weights_bitwise_equal']
        assert report['earliest_weight_difference']['update'] == 3
        assert len(report['per_update_weights']) == 21
        assert not report['per_update_weights']['3']['weights_bitwise_equal']
        assert report['per_update_weights']['20']['weights_bitwise_equal']
    finally:
        for b in bundles:
            b.reader.db.close()

    def mutate_metric(rows):
        row = next(r for r in rows if r['kind'] == 'update_end')
        row['data']['metrics']['approx_kl'] = row['data']['metrics']['policy_loss']

    right = reseal(tiny_pair, tmp_path / 'diag', 'Pv26t', mutate_metric)
    out_report = tmp_path / 'diag-report.json'
    assert compare_cli([str(tiny_pair / 'Pv26c'), str(right), '--out-report', str(out_report)]) == 0
    saved = json.loads(out_report.read_text())
    assert saved['weights_bitwise_equal'] and not saved['full_trace_equal']


def test_reseal_unilateral_stage_swap_rejected(tiny_pair, tmp_path):
    def swap_stage(rows):
        i = next(i for i, r in enumerate(rows) if r['kind'] == 'actor_prestep')
        rows[i], rows[i + 1] = rows[i + 1], rows[i]

    right = reseal(tiny_pair, tmp_path, 'Pv26t', swap_stage)
    with pytest.raises(ValueError):
        compare(tiny_pair / 'Pv26c', right)


def test_reseal_joint_omission_rejected(tiny_pair, tmp_path):
    def strip_weights(rows):
        for row in rows:
            if 'model' in row['data']:
                for ref in row['data']['model'].values():
                    ref.pop('blob', None)
                    ref.pop('bytes', None)

    left = reseal(tiny_pair, tmp_path, 'Pv26c', strip_weights)
    right = reseal(tiny_pair, tmp_path, 'Pv26t', strip_weights)
    with pytest.raises(ValueError):
        compare(left, right)


def test_reseal_false_termination_rejected(tiny_pair, tmp_path):
    def flip_end(rows):
        for row in rows:
            if row['kind'] == 'episode_end':
                assert row['data']['censored'] and not row['data']['terminated']
                row['data'].update(terminated=True, truncated=False, censored=False)

    left = reseal(tiny_pair, tmp_path, 'Pv26c', flip_end)
    right = reseal(tiny_pair, tmp_path, 'Pv26t', flip_end)
    with pytest.raises(ValueError, match='termination'):
        compare(left, right)


def test_real_death_frame_alignment(tmp_path, monkeypatch):
    import qrokkun_ai.v26.run as run_mod

    original = run_mod.collect

    def perturbed(p, s, device, seeds, cap, workers, writer, update, validation=False, check=lambda: None):
        if workers > 1 and update == 1 and not validation:
            with torch.no_grad():
                p.policy_head.bias[1] += 30
        return original(p, s, device, seeds, cap, workers, writer, update, validation, check)

    monkeypatch.setattr(run_mod, 'collect', perturbed)
    for label in ('Pv26c', 'Pv26t'):
        run_mod.main([
            '--arm', label,
            '--out-dir', str(tmp_path / label),
            '--trace-dir', str(tmp_path / (label + '-trace')),
            '--source-commit', 'a' * 40,
            '--nonformal',
            '--synthetic',
            '--device', 'cpu',
            '--updates', '2',
            '--episodes', '1',
            '--cap', '90',
            '--validation', '1',
            '--hidden', '8',
        ])
    ends = []
    for label in ('Pv26c', 'Pv26t'):
        reader = Reader(tmp_path / (label + '-trace'))
        try:
            ends.append({
                tuple(r['key']): r['data']['frames']
                for r in reader.events()
                if r['kind'] == 'episode_end'
            })
        finally:
            reader.db.close()
    assert ends[0] != ends[1], 'fixture must exercise actual differing death frames'
    result = compare(tmp_path / 'Pv26c', tmp_path / 'Pv26t')
    assert result['status'] == 'complete_different'
    assert result['events'][0] != result['events'][1]
    assert result['structural_first_difference_per_update']
    assert sum(result['missing_event_counts']) > 0
    assert result['first_difference_per_update']['2']['event'][1][0] == 2
    assert result['first_difference_per_update']['2']['difference']['path'] != '$event'
