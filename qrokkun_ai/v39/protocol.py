"""Version-local identity and frozen original Pv38 control coverage."""
import json
import hashlib
import re
import tarfile
import subprocess
from pathlib import Path
from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v35.protocol import paired, FINAL_SEEDS, better
from qrokkun_ai.v38.protocol import validate_record
from .fingerprint import ROOT, source_files

SCHEDULE = list(range(0,1001,10))
OLD_COMMIT = 'e032de5ec7091937db7007f4b881705b0ce17350'
OLD_DIGEST = '0978f56b8902fb93a665cd5be5164c7bb3ed19d178efbc16613eb12a151c8fb3'

def training_seeds(update):
    return list(range(100000+(update-1)*40,100000+update*40))

def read(root, name): return json.loads((Path(root)/name).read_text())

def checked_hash(path, stop=lambda:False):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if stop(): raise InterruptedError('owner stop')
            block=stream.read(1024*1024)
            if not block: break
            digest.update(block)
    if stop(): raise InterruptedError('owner stop')
    return digest.hexdigest()

def read_pinned_json(root, name, pins, stop=lambda:False):
    """Parse precisely the bytes checked against the already frozen identity."""
    if stop(): raise InterruptedError('owner stop')
    data=(Path(root)/name).read_bytes()
    if stop(): raise InterruptedError('owner stop')
    if hashlib.sha256(data).hexdigest()!=pins[name]: raise ValueError('control JSON bytes changed: '+name)
    result=json.loads(data)
    if stop(): raise InterruptedError('owner stop')
    return result

def expected_control_names():
    return {'manifest.json','closed.json','status.json','initial.json','best.json','latest.json','terminal.json',
        'updates.jsonl','final.json','final-status.json','final-best.json','final-terminal.json','source.tar'} | {
        f'{prefix}-{u:06d}.{suffix}' for u in SCHEDULE for prefix,suffix in [('checkpoint','pt'),('evaluation','json')]}

def require_coverage(pins):
    if set(pins) != expected_control_names(): raise ValueError('control pin coverage must be generated from 0,10..1000')

def prepare_control(root, stop=lambda:False):
    """Read/hash only: never load historical checkpoint tensors or write control."""
    root=Path(root)
    def file_hash(path): return checked_hash(path,stop)
    def read(root,name):
        if stop(): raise InterruptedError('owner stop')
        data=(Path(root)/name).read_bytes()
        if stop(): raise InterruptedError('owner stop')
        return json.loads(data)
    m,c,t=[read(root,n) for n in ('manifest.json','closed.json','status.json')]
    s=m['semantic']
    if (m['arm']!='Pv38m' or s['source_commit']!=OLD_COMMIT or s['source_tree_sha']!=OLD_DIGEST
        or s['episodes']!=20 or s['updates']!=1000 or s['cap']!=7200 or s['hidden']!=472
        or s['seed']!=17 or s['nonformal'] or not c['formal'] or t['status']!='complete'
        or t.get('errors') or t.get('owner_stop') or c['completed_updates']!=1000
        or c['source_tree_sha']!=OLD_DIGEST or c['manifest_sha']!=file_hash(root/'manifest.json')
        or c['saved_checkpoints']!=SCHEDULE or s['validation_boundaries']!=SCHEDULE):
        raise ValueError('original full Pv38 closure required')
    totals=[]
    for u in SCHEDULE:
        if stop(): raise InterruptedError('owner stop')
        e=read(root,f'evaluation-{u:06d}.json')
        if [r['seed'] for r in e['episodes']]!=list(range(600000,600016)): raise ValueError('validation coverage')
        for row in e['episodes']: validate_record(row)
        total=sum(r['frames'] for r in e['episodes'])
        if total!=e['total_frames']: raise ValueError('validation total')
        totals.append((u,total))
    if max(totals,key=lambda x:x[1])[0]!=c['best_update']: raise ValueError('earliest best')
    pins={}
    for name in sorted(expected_control_names()):
        if stop(): raise InterruptedError('owner stop')
        pins[name]=file_hash(root/name)
    require_coverage(pins)
    if pins['source.tar']!=m['source_archive_sha']: raise ValueError('old archive identity')
    if set(c['checkpoint_hashes'])!={f'checkpoint-{u:06d}.pt' for u in SCHEDULE}: raise ValueError('old checkpoint hash coverage')
    for name,sha in c['checkpoint_hashes'].items():
        if pins[name]!=sha: raise ValueError('old checkpoint bytes')
    # Validate the saved source against its own old closure, never the Pv39 digest.
    from qrokkun_ai.v37.fingerprint import source_files as old_source_files
    old_hash=hashlib.sha256()
    with tarfile.open(root/'source.tar') as archive:
        for path in old_source_files():
            if stop(): raise InterruptedError('owner stop')
            name=str(path.relative_to(ROOT))
            old_hash.update(name.encode()); old_hash.update(archive.extractfile(name).read())
    if old_hash.hexdigest()!=OLD_DIGEST: raise ValueError('old source archive closure')
    for name,u in [('initial',0),('best',c['best_update']),('latest',1000),('terminal',1000)]:
        if read(root,name+'.json')!={'update':u,'checkpoint':f'checkpoint-{u:06d}.pt'}: raise ValueError('old pointer')
    f=read(root,'final.json'); fs=read(root,'final-status.json')
    if f['status']!='complete' or fs['status']!='complete' or fs['errors'] or fs['owner_stop'] or f['spawner_offset']!=24000000:
        raise ValueError('old final closure')
    for label,u in [('best',c['best_update']),('terminal',1000)]:
        r=f['results'][label]
        if read(root,'final-'+label+'.json')!=r or r['update']!=u or r['file_sha']!=pins[f'checkpoint-{u:06d}.pt']:
            raise ValueError('old final checkpoint identity')
        if r['player_mode']!='argmax' or r['spawner_mode']!='stochastic' or [x['seed'] for x in r['episodes']]!=FINAL_SEEDS:
            raise ValueError('old final modes/seeds')
        for row in r['episodes']:
            validate_record(row)
            if row['state_sha']!=r['state_sha']: raise ValueError('old final state identity')
    return pins

def archive_source(commit, destination):
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('explicit source commit required')
    paths=source_files()
    if (ROOT/'.git').exists():
        subprocess.run(['git','--git-dir='+str(ROOT/'.git'),'archive','--format=tar','-o',str(destination),commit],check=True)
        with tarfile.open(destination) as archive:
            for p in paths:
                if archive.extractfile(str(p.relative_to(ROOT))).read()!=p.read_bytes():
                    raise ValueError('immutable committed source differs: '+str(p))
    else:
        with tarfile.open(destination,'w',pax_headers={'qrokkun.source_commit':commit}) as archive:
            for p in paths: archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)


def freeze_control(root, destination, stop=lambda:False):
    """Predeclare the complete control identity before scientific execution."""
    pins=prepare_control(root, stop)
    payload=dict(schema='pv39_control_pins_v1',control_dir=str(Path(root).resolve()),pins=pins)
    with Path(destination).open('x') as stream:
        json.dump(payload,stream,sort_keys=True,indent=2)
    return payload

def consume_control(root, pin_path, stop=lambda:False):
    if stop(): raise InterruptedError('owner stop')
    payload=json.loads(Path(pin_path).read_text())
    if stop(): raise InterruptedError('owner stop')
    if payload['schema']!='pv39_control_pins_v1' or payload['control_dir']!=str(Path(root).resolve()):
        raise ValueError('frozen control location/schema')
    require_coverage(payload['pins'])
    if prepare_control(root,stop)!=payload['pins']: raise ValueError('frozen control changed')
    if stop(): raise InterruptedError('owner stop')
    return payload['pins']

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description='Freeze validated original Pv38 hashes; no tensor loads')
    p.add_argument('--control-dir',required=True)
    p.add_argument('--control-pins',required=True)
    a=p.parse_args(argv)
    freeze_control(a.control_dir,a.control_pins)
    return 0

if __name__=='__main__': raise SystemExit(main())
