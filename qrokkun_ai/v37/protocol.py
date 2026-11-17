"""Trial pairing and explicit source archive; no formal/final seed protocol."""
import json
import re
from pathlib import Path
import subprocess
import tarfile

def schedule(updates):
    return sorted({0, *range(10,updates+1,10), updates})

def better(total, previous): return total > previous

def archive_source(commit, destination):
    """Archive checkout bytes verified against commit, or deployed closure bytes.

    Without root .git, commit is explicit caller-supplied provenance, not a
    locally verified Git identity. The existing source_tree_sha identifies the
    executed production bytes independently of that provenance.
    """
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', commit):
        raise ValueError('--source-commit must be a 40-char hex SHA')
    root = Path(__file__).resolve().parents[2]
    from .fingerprint import source_files
    paths = source_files()
    if not (root / '.git').exists():
        with tarfile.open(destination, 'w', format=tarfile.PAX_FORMAT,
                          pax_headers={'qrokkun.source_commit': commit,
                                       'qrokkun.source_identity': 'explicit-provenance; current-production-closure; git-unverified'}) as archive:
            for path in paths:
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        return
    # Explicit git-dir prevents discovery of an unrelated ancestor checkout.
    subprocess.run(['git', '--git-dir=' + str(root / '.git'),
                    '--work-tree=' + str(root), 'archive', '--format=tar',
                    '-o', str(Path(destination).resolve()), commit], cwd=root, check=True)
    with tarfile.open(destination) as archive:
        for path in paths:
            name=str(path.relative_to(root))
            try:
                archived = archive.extractfile(name)
            except KeyError as exc:
                raise ValueError('source commit lacks production dependency: '+name) from exc
            if archived is None or archived.read() != path.read_bytes():
                raise ValueError('source commit does not contain current implementation: '+name)

def verify_control(directory, search, stop=lambda: False):
    manifest=json.loads((directory/'manifest.json').read_text())
    control=manifest['semantic']
    status=json.loads((directory/'status.json').read_text())
    closed=json.loads((directory/'closed.json').read_text())
    if (manifest['arm'] not in ('Pv37-control','Pv37m') or status['status'] != 'complete'
        or status.get('errors') or status.get('owner_stop') or closed['completed_updates'] != (1000 if search.get('nonformal') is False else 2)):
        raise ValueError('completed Gv36 Pv37 control trial required')
    keys=('environment','initial_player_state_sha','spawner_state_sha','inputs','source_commit',
          'source_tree_sha','updates','episodes','cap','seed','validation','validation_seeds',
          'hidden','hyperparameters','runtime','initialization','architecture','fresh_adam',
          'player_mode_training','player_mode_evaluation','spawner_mode','validation_boundaries')
    for key in keys:
        # The manifest has crossed JSON serialization; initializer mappings use tuples.
        expected = json.loads(json.dumps(search[key]))
        if control[key] != expected: raise ValueError('paired control mismatch: '+key)

    if search.get('nonformal') is False:
        for key in ('checkpoint_boundaries','checkpoint_cadence','validation_cadence','initialization_seed','training_seed','P_offset','S_train_offset','S_validation_offset','permutation_seed','formal_contract','nonformal','formal_authorized'):
            if control[key] != search[key]:
                raise ValueError('paired control mismatch: '+key)
        if manifest['workers'] != 10 or manifest['pin_cores'] != 'on':
            raise ValueError('formal control worker configuration mismatch')
        from qrokkun_ai.v38.protocol import control_pin
        return control_pin(directory, stop)
    return None
