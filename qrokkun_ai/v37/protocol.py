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

def verify_control(directory, search):
    manifest=json.loads((directory/'manifest.json').read_text())
    control=manifest['semantic']
    status=json.loads((directory/'status.json').read_text())
    closed=json.loads((directory/'closed.json').read_text())
    if (manifest['arm'] != 'Pv37-control' or status['status'] != 'complete'
        or status.get('errors') or status.get('owner_stop') or closed['completed_updates'] != 2):
        raise ValueError('completed Gv36 Pv37 control trial required')
    keys=('environment','initial_player_state_sha','spawner_state_sha','inputs','source_commit',
          'source_tree_sha','updates','episodes','cap','seed','validation','validation_seeds',
          'hidden','hyperparameters','runtime','initialization','architecture','fresh_adam',
          'player_mode_training','player_mode_evaluation','spawner_mode','validation_boundaries')
    for key in keys:
        # The manifest has crossed JSON serialization; initializer mappings use tuples.
        expected = json.loads(json.dumps(search[key])) if key == 'initialization' else search[key]
        if control[key] != expected: raise ValueError('paired control mismatch: '+key)
