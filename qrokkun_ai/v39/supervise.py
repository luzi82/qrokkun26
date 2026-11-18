"""Single group-owning absolute deadline supervisor; no GPU work at import."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent

def group_members(pgid):
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            data = (entry/'stat').read_text()
            fields = data[data.rfind(')')+2:].split()
            if int(fields[2]) == pgid:
                rows.append(dict(pid=int(entry.name), state=fields[0]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows

def supervise_owned(argv, launcher, deadline, stop_path, grace=5.):
    validate_deadline(deadline, stop_path)
    launcher = Path(launcher)
    launcher.mkdir(parents=True, exist_ok=False)
    record = dict(argv=argv, deadline_epoch=deadline, errors=[], stop=[], signals_attempted=[], pid=None)
    child = None
    handlers = {}
    def persist():
        temp = launcher/'status.tmp'
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(launcher/'status.json')
    def received(sig, frame): record['stop'].append('signal '+str(sig))
    def members():
        try: return group_members(child.pid)
        except BaseException as exc:
            record['errors'].append('group membership unknown: '+repr(exc))
            raise
    def send(sig):
        record['signals_attempted'].append(int(sig))
        try: os.killpg(child.pid, sig)
        except ProcessLookupError: pass
        except BaseException as exc: record['errors'].append('killpg: '+repr(exc))
    record['started_epoch'] = time.time()
    monotonic_start = time.monotonic()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM): handlers[sig] = signal.signal(sig, received)
        with (launcher/'stdout').open('xb') as stdout, (launcher/'stderr').open('xb') as stderr:
            validate_deadline(deadline, stop_path)
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     start_new_session=True)
            record['pid'] = child.pid
            # PID/argv are the first persisted identity, before /proc registration.
            persist()
            record['initial_members'] = members(); persist()
            while child.poll() is None:
                if stop_path and Path(stop_path).exists():
                    if 'owner STOP' not in record['stop']: record['stop'].append('owner STOP')
                if time.time() >= deadline-120:
                    if 'deadline' not in record['stop']: record['stop'].append('deadline')
                if record['stop']: break
                time.sleep(.1)
    except BaseException as exc:
        record['errors'].append(repr(exc))
    finally:
        if child is not None:
            # Also clean descendants when the leader has already exited.
            send(signal.SIGTERM)
            end = time.monotonic()+grace
            try:
                while time.monotonic()<end:
                    child.poll()
                    if not any(r['state']!='Z' for r in members()): break
                    time.sleep(.05)
                remaining = members()
                if any(r['state']!='Z' for r in remaining): send(signal.SIGKILL)
                child.wait(timeout=5)
                end = time.monotonic()+5
                while time.monotonic()<end:
                    remaining = members()
                    if not any(r['state']!='Z' for r in remaining): break
                    time.sleep(.05)
                record['group_members'] = remaining
                record['zombies'] = [r for r in remaining if r['state']=='Z']
                if any(r['state']!='Z' for r in remaining): record['errors'].append('live group remains')
            except BaseException as exc:
                record['errors'].append('cleanup unknown: '+repr(exc))
                send(signal.SIGKILL)
                try: child.wait(timeout=5)
                except BaseException as exc: record['errors'].append('reap: '+repr(exc))
            record['returncode'] = child.returncode
        for sig, old in handlers.items(): signal.signal(sig, old)
        record['success'] = child is not None and child.returncode==0 and not record['errors'] and not record['stop']
        record['finished_epoch'] = time.time()
        record['elapsed_seconds'] = time.monotonic()-monotonic_start
        persist()
    return record

def validate_deadline(deadline, stop_path):
    remaining=deadline-time.time()
    if not 3720 < remaining <= 57600: raise ValueError('future absolute deadline requires final reserve3600+cleanup120, maximum16h')
    if stop_path and Path(stop_path).exists(): raise InterruptedError('owner STOP')

def stage_controller(a):
    """One controller, no restart; all child stages remain in the outer group."""
    record=dict(errors=[],owner_stop=[],stages=[])
    out=Path(a.out_dir)
    def persist():
        target=out.with_name(out.name+'.launcher')/'controller.json'
        target.write_text(json.dumps(record,indent=2)+'\n')
    common=['--out-dir',str(out),'--control-dir',a.control_dir,'--source-commit',a.source_commit,
        '--spawner-checkpoint',a.spawner_checkpoint,'--control-pins',a.control_pins,
        '--scratch',a.scratch,'--stop-path',a.stop_path,'--device',a.device]
    if a.smoke: common+=['--smoke']
    try:
        for stage,module,deadline in [('train','qrokkun_ai.v39.run',a.deadline_epoch-3600),('final','qrokkun_ai.v39.final',a.deadline_epoch-120)]:
            if stage=='final' and a.smoke: break
            if Path(a.stop_path).exists(): record['owner_stop'].append('owner STOP'); persist(); return 1
            if time.time()>=deadline: record['errors'].append(stage+' deadline'); persist(); return 1
            if stage=='final':
                from .final import closure
                closure(out,lambda:Path(a.stop_path).exists())
            if Path(a.stop_path).exists(): record['owner_stop'].append('owner STOP'); persist(); return 1
            child=subprocess.Popen([sys.executable,'-m',module,*common,'--deadline-epoch',str(deadline)],stdin=subprocess.DEVNULL)
            row=dict(stage=stage,pid=child.pid,returncode=None); record['stages'].append(row); persist()
            while child.poll() is None:
                if Path(a.stop_path).exists() or time.time()>=deadline:
                    if Path(a.stop_path).exists(): record['owner_stop'].append('owner STOP')
                    if time.time()>=deadline: record['errors'].append(stage+' deadline')
                    # Outer owner also cleans descendants after this direct child is reaped.
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait(timeout=5)
                    row['returncode']=child.returncode
                    persist(); return 1
                time.sleep(.25)
            row['returncode']=child.returncode
            if child.returncode: record['errors'].append(stage+' exit '+str(child.returncode))
            persist()
            if record['errors'] or record['owner_stop']: return 1
        return 0
    except BaseException as exc:
        record['errors'].append(repr(exc))
        if Path(a.stop_path).exists() and 'owner STOP' not in record['owner_stop']:
            record['owner_stop'].append('owner STOP')
        persist()
        return 1

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('control-dir','out-dir','scratch','source-commit','stop-path','spawner-checkpoint','control-pins'): p.add_argument('--'+key,required=True)
    p.add_argument('--deadline-epoch',required=True,type=float)
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--controller',action='store_true',help=argparse.SUPPRESS)
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    if a.controller: return stage_controller(a)
    validate_deadline(a.deadline_epoch,a.stop_path)
    from .run import validate_inputs
    validate_inputs(a)
    if Path(a.out_dir).exists(): raise FileExistsError(a.out_dir)
    command=[sys.executable,'-m','qrokkun_ai.v39.supervise',*(sys.argv[1:] if argv is None else argv),'--controller']
    result=supervise_owned(command,Path(a.out_dir).with_name(Path(a.out_dir).name+'.launcher'),a.deadline_epoch,a.stop_path)
    return 0 if result['success'] else 1

if __name__=='__main__': raise SystemExit(main())
