"""Sequential c -> w -> offline resolved raw weights comparison."""
import json
from pathlib import Path
import torch
from .run import parser,validate,main as run,json_write
from .core import weight_diff
from qrokkun_ai.v26.cli import boundary


def compare(root):
    manifests=[json.loads((root/arm/'manifest.json').read_text()) for arm in ('Pv27c','Pv27w')]
    if manifests[0]['semantic']!=manifests[1]['semantic']: raise ValueError('source/input/runtime mismatch')
    updates=manifests[0]['semantic']['updates']; differences={}
    for arm in ('Pv27c','Pv27w'):
        status=json.loads((root/arm/'status.json').read_text())
        if status['status']!='complete' or status['completed_updates']!=updates: raise RuntimeError('incomplete arm')
    for u in range(updates+1):
        states=[torch.load(root/arm/f'checkpoint-{u:02d}.pt',map_location='cpu',weights_only=False)['model'] for arm in ('Pv27c','Pv27w')]
        differences[str(u)]=weight_diff(*states)
    code=2 if differences[str(updates)] else 0
    timings={arm:json.loads((root/arm/'closed.json').read_text())['timings'] for arm in ('Pv27c','Pv27w')}
    speed={k:timings['Pv27c'][k]/timings['Pv27w'][k] for k in ('collector_s','ppo_s','total_s')}
    json_write(root/'comparison.json',dict(result='WEIGHTPASS' if code==0 else 'DIFFERENT',exitcode=code,per_update_named_differences=differences,timings=timings,serial_over_workers_wall_ratio=speed,speed_is_separate=True))
    return code


def main(argv=None):
    a=parser().parse_args(argv); validate(a); root=Path(a.out_dir); root.mkdir(parents=True,exist_ok=False)
    args=list(argv) if argv is not None else __import__('sys').argv[1:]
    expected=None
    for arm in ('Pv27c','Pv27w'):
        call=args.copy(); call[call.index('--arm')+1]=arm; call[call.index('--out-dir')+1]=str(root/arm)
        result=run(call,expected=expected)
        expected=json.loads((root/arm/'manifest.json').read_text())['semantic']
        if result: return result
    return compare(root)


def cli(argv=None): return boundary(main,argv)
if __name__=='__main__': raise SystemExit(cli())
