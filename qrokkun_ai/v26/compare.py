"""Offline complete-bundle comparator. Exit 0 final weights equal, 2 final weights different, 3 incomplete/invalid."""
import argparse
import tempfile
from contextlib import closing
import json
import math
from pathlib import Path
import struct
import sqlite3
import zlib
import numpy as np
from .trace import Reader, digest, json_write


def difference(a,b,path='$',readers=None):
    if isinstance(a,dict) and isinstance(b,dict):
        if 'blob' in a and 'blob' in b:
            if readers:
                ra,rb=readers[0].raw(a),readers[1].raw(b)
                if a['dtype']==b['dtype'] and a['shape']==b['shape'] and ra==rb:return None
            elif a==b:return None
            result=dict(path=path,left=a,right=b)
            if readers:
                ra,rb=readers[0].raw(a),readers[1].raw(b)
                offset=next((i for i,(x,y) in enumerate(zip(ra,rb)) if x!=y),min(len(ra),len(rb)))
                result.update(byte_index=offset,left_raw_hex=ra[max(0,offset-8):offset+9].hex(),right_raw_hex=rb[max(0,offset-8):offset+9].hex())
                try:
                    d=a['dtype']
                    dtype=np.dtype('>f8' if d=='python.float64.be' else d.replace('torch.','').replace('numpy:',''))
                    if dtype.kind in 'fiu' and a['dtype']==b['dtype']:
                        i=offset//dtype.itemsize
                        av=np.frombuffer(ra,dtype=dtype)[i]; bv=np.frombuffer(rb,dtype=dtype)[i]
                        delta=float(bv)-float(av)
                        result.update(tensor_flat_index=i,left_value=repr(av),right_value=repr(bv),numerical_difference=repr(delta))
                except (ValueError,TypeError,IndexError): pass
            return result
        if set(a)!=set(b): return dict(path=path,missing_left=sorted(set(b)-set(a)),missing_right=sorted(set(a)-set(b)))
        # Sampler pre-state precedes model inputs; draw results follow outputs.
        # Nested S records are emitted after their draw, so project their upstream
        # fields first while retaining list positions and source event chronology.
        for name in ('global_pre','before','pre'):
            if name in a:
                d=difference(a[name],b[name],path+'.'+name,readers)
                if d:return d
        for field in ('draws','events'):
            if isinstance(a.get(field),list) and isinstance(b.get(field),list):
                for i,(x,y) in enumerate(zip(a[field],b[field])):
                    if isinstance(x,dict) and isinstance(y,dict):
                        for name in ('pre',):
                            if name in x and name in y:
                                d=difference(x[name],y[name],f'{path}.{field}[{i}].{name}',readers)
                                if d:return d
                if field=='events':
                    for i,(x,y) in enumerate(zip(a[field],b[field])):
                        if x.get('type')==y.get('type')=='S':
                            for name in ('observation','means','std'):
                                if name in x and name in y:
                                    d=difference(x[name],y[name],f'{path}.{field}[{i}].{name}',readers)
                                    if d:return d
        priority = ['global_pre','before','pre','rng','player','bullets','pad','features','raw_logits','logits','probs','values','observation','means','std','draws','events','output','post','actions','action','after','global_post']
        for k in sorted(a,key=lambda x:(priority.index(x) if x in priority else len(priority),x)):
            d=difference(a[k],b[k],path+'.'+k,readers)
            if d: return d
        return None
    if isinstance(a,list) and isinstance(b,list):
        for i,(x,y) in enumerate(zip(a,b)):
            d=difference(x,y,f'{path}[{i}]',readers)
            if d:return d
        if len(a)!=len(b):return dict(path=path,left_length=len(a),right_length=len(b))
        return None
    if type(a)!=type(b) or a!=b:return dict(path=path,left=a,right=b)
    return None


REQUIRED={
 'global_init':{'python','numpy_legacy','numpy_generators','torch_cpu','torch_cuda'},
 'global_final':{'python','numpy_legacy','numpy_generators','torch_cpu','torch_cuda'},
 'opponent':{'model','frozen','device'},
 'reset':{'seed','state','events','initial_generators'},
 'P':{'active','player','bullets','pad','features','raw_logits','logits','probs','values','teacher_actions','teacher_logits','teacher_probs','actions','logp','draws','global_pre','global_post'},
 'G':{'before','after','events','action','reward','observation','base_reward','done','info'},
 'episode_end':{'seed','frames','terminated','truncated','censored','bootstrap','state','rng_draw_counts','S_draw_count','P_draw_count','unused'},
 'bootstrap':{'active','player','bullets','pad','features','values'},
 'rollouts':{'flattened_order','rollouts'},'batch':{'batch','global_rng','training_rng'},
 'permutation':{'source','counter','pre','post','output','global_rng'},
 'checkpoint':{'model','actor','critic','gradients','global_rng','training_rng'},
 'update_end':{'metrics','frames'}}
for kind in ('minibatch','actor_preclip','actor_postclip','actor_prestep','actor_poststep','critic_preclip','critic_postclip','critic_prestep','critic_poststep'):
    REQUIRED[kind]={'model','actor','critic','gradients','global_rng','training_rng'} | ({'indices','normalized_advantages'} if kind=='minibatch' else {'loss'})


for kind in ('actor_preclip','actor_postclip','actor_prestep','actor_poststep'):
    REQUIRED[kind] |= {'losses','teacher_probs','logits','values','features','raw_logits','probs'}


class Bundle:
    def __init__(self,path,expected_commit=None):
        self.path=Path(path)
        self.m=json.loads((self.path/'manifest.json').read_text())
        self.c=json.loads((self.path/'closed.json').read_text())
        self.s=self.m['semantic']
        from .validation import schema
        try:self.template=schema(self.s,self.m,expected_commit)
        except (KeyError,TypeError) as e:raise ValueError('missing/invalid semantic schema '+str(e)) from e
        if self.c['status']!='complete' or self.c['completed_updates']!=self.s['updates']:
            raise ValueError('incomplete run')
        if digest((self.path/'manifest.json').read_bytes())!=self.c['manifest_sha']:
            raise ValueError('manifest hash mismatch')
        self.reader=Reader(self.m['trace_dir'])
        if digest((Path(self.m['trace_dir'])/'closed.json').read_bytes())!=self.c['trace_closure_sha']:
            raise ValueError('trace closure mismatch')
        expected={f'checkpoint-{u:02d}.json' for u in range(self.s['updates']+1)}
        if set(self.c['checkpoints'])!=expected:raise ValueError('checkpoint coverage')
        for n,h in self.c['checkpoints'].items():
            if digest((self.path/n).read_bytes())!=h:raise ValueError('checkpoint hash mismatch')
            self.reader.validate(json.loads((self.path/n).read_text()))
        if self.s['heldout'] or self.s['validation_seeds']!=list(range(600000,600000+self.s['validation'])):
            raise ValueError('heldout/validation contract')
        from .run import HYPER, PLAYER_SHA, STATE_SHA, S_SHA, S_STATE
        if self.s['hyperparameters']!=HYPER:
            raise ValueError('hyperparameter identity')
        if not self.s['synthetic'] and any(self.s[k]!=v for k,v in dict(player_sha=PLAYER_SHA,player_state_sha=STATE_SHA,spawner_sha=S_SHA,spawner_state_sha=S_STATE).items()):
            raise ValueError('locked input identity')
        if not self.s['nonformal']:
            for k,v in dict(updates=20,episodes=12,cap=7200,seed=17,validation=16,hidden=472).items():
                if self.s[k]!=v:raise ValueError('formal bounds '+k)
            if self.s['synthetic'] or self.s['heldout'] or self.s['validation_seeds']!=list(range(600000,600016)):
                raise ValueError('formal identity/seeds')

    def events(self):
        from .validation import Episodes, Scientific, model
        from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
        episodes=Episodes(self.reader,self.s['runtime']['device'])
        scientific=Scientific(self.reader,self.s,self.template)
        seen=set(); resets={}; counts={}; ends={}; steps={}; permutations={}; checkpoints=set(); updates=set()
        total=0; pframes={}; gframes={}; stagekeys={}; update_kinds={}; permkeys={}
        for row in self.reader.events():
            k,key,data=row['kind'],row['key'],row['data']
            if total and key[0]!=last_update:
                if pframes!=gframes:raise ValueError('missing P/G frame barrier coverage')
                if any(keys!=stagekeys.get('minibatch') for keys in stagekeys.values()):raise ValueError('optimizer substep key coverage')
                pframes.clear();gframes.clear();stagekeys.clear()
            if k not in REQUIRED or REQUIRED[k]!=set(data):raise ValueError('missing required fields '+k)
            if 'model' in REQUIRED[k] and k!='opponent':
                from .validation import snapshot
                snapshot(self.reader,data,self.template,k,key)
            state_fields={'player' ,'elapsed','frame','dt','spawn_acc','dead','bullets','next_event','event_clock','s_history','field_history','python_rng','P','S','birth'}
            state_values=[]
            if k in ('reset','episode_end'):state_values=[data['state']]
            if k=='G':state_values=[data['before'],data['after']]
            for state in state_values:
                if not state_fields <= set(state):raise ValueError('missing environment state field')
                if set(state['python_rng'])!={'rng','s_rng','field_rng'}:raise ValueError('missing state RNG')
            if k=='P':
                fkey=(key[0],key[1],key[2]+1)
                pframes[fkey]=data['active']
                for draw in data['draws']:
                    if not {'source','counter','pre','post','output'}<=set(draw):raise ValueError('missing sampler field')
            if k=='G':
                fkey=(key[0],key[1],key[3]);gframes.setdefault(fkey,[]).append(key[2])
                for event in data['events']:
                    if event['type']=='rng' and not {'source','counter','pre','post','output'}<=set(event):raise ValueError('missing S/env sampler field')
                    if event['type']=='S' and not {'observation','means','std','action','logp'}<=set(event):raise ValueError('missing S field')
            if k=='batch' and not {'player','bullets','pad','actions','old_log_probs','values','advantages','returns','expert_actions','expert_logits'}<=set(data['batch']):
                raise ValueError('missing batch field')
            if k in ('minibatch','actor_preclip','actor_postclip','actor_prestep','actor_poststep','critic_preclip','critic_postclip','critic_prestep','critic_poststep'):
                stagekeys.setdefault(k,set()).add(tuple(key))
            opponent_hash=episodes.check(k,key,data)
            scientific.check(k,key,data)
            if k=='opponent' and opponent_hash!=self.s['spawner_state_sha']:raise ValueError('opponent actual state hash mismatch')
            if k=='checkpoint' and key==[0]:
                if state_dict_sha256(model(self.reader,data['model'],self.template))!=self.s['player_state_sha']:raise ValueError('initial P actual state hash mismatch')
            identity=(k,tuple(key))
            if identity in seen:raise ValueError('duplicate event '+repr(identity))
            # Keep only one update's keys in RAM (bounded by cap/episodes, not u20 trace).
            if total and key[0]!=last_update:
                seen.clear()
            seen.add(identity); last_update=key[0]; total+=1
            if k=='reset':
                ep=tuple(key[:3]); resets[ep]=data['seed'];counts[ep]=0
            if k=='G':
                ep=tuple(key[:3])
                if ep not in counts or key[3]!=counts[ep]+1:raise ValueError('missing/out-of-order G frame')
                counts[ep]+=1
            if k=='episode_end':
                ep=tuple(key[:3]); n=data['frames']
                if ep not in counts or counts[ep]!=n or not 0<n<=self.s['cap'] or data['seed']!=resets[ep]:
                    raise ValueError('episode frame/seed coverage')
                if not data['terminated'] and (not data['censored'] or n!=self.s['cap']):raise ValueError('invalid censor')
                ends[ep]=n
            if k in ('batch','rollouts'):update_kinds.setdefault(k,set()).add(key[0])
            if k=='permutation':
                permutations[key[0]]=permutations.get(key[0],0)+1
                permkeys.setdefault(key[0],set()).add(key[1])
            if k=='critic_poststep':steps[key[0]]=steps.get(key[0],0)+1
            if k=='checkpoint':
                u=key[0];checkpoints.add(u)
                if data!=json.loads((self.path/f'checkpoint-{u:02d}.json').read_text()):raise ValueError('checkpoint trace mismatch')
            if k=='update_end':updates.add(key[0])
            yield row
        scientific.finish()
        if pframes!=gframes:raise ValueError('missing P/G frame barrier coverage')
        if any(keys!=stagekeys.get('minibatch') for keys in stagekeys.values()):raise ValueError('optimizer substep key coverage')
        wanted=set()
        for u in range(1,self.s['updates']+1):
            for i in range(self.s['episodes']):
                ep=(u,'train',i);wanted.add(ep)
                if resets.get(ep)!=100000+(u-1)*self.s['episodes']+i:raise ValueError('train seed coverage')
            n=sum(ends.get((u,'train',i),0) for i in range(self.s['episodes']))
            if permutations.get(u)!=4 or permkeys.get(u)!=set(range(4)) or steps.get(u)!=4*math.ceil(n/512):raise ValueError('optimizer coverage')
        vals={0,10,20}&set(range(self.s['updates']+1))
        if self.s['nonformal']: vals.add(self.s['updates'])
        for u in vals:
            for i in range(self.s['validation']):
                ep=(u,'validation',i);wanted.add(ep)
                if resets.get(ep)!=600000+i:raise ValueError('validation seed coverage')
        if set(resets)!=wanted or set(ends)!=wanted:raise ValueError('incomplete episode coverage')
        if checkpoints!=set(range(self.s['updates']+1)) or updates!=set(range(1,self.s['updates']+1)):
            raise ValueError('update/checkpoint coverage')
        if any(update_kinds.get(k)!=set(range(1,self.s['updates']+1)) for k in ('batch','rollouts')):raise ValueError('missing rollout/batch coverage')
        for kind in ('global_init','global_final','opponent'):
            if self.reader.closure['counts'].get(kind)!=1:raise ValueError('missing lifecycle event '+kind)
        # Every minibatch has exactly all eight optimizer substep snapshots.
        n=sum(steps.values())
        for kind in ('minibatch','actor_preclip','actor_postclip','actor_prestep','actor_poststep','critic_preclip','critic_postclip','critic_prestep','critic_poststep'):
            if self.reader.closure['counts'].get(kind)!=n:raise ValueError('missing optimizer substep '+kind)


def _compare(a,b):
    if a.m['arm']!='Pv26c' or b.m['arm']!='Pv26t' or a.m['workers']!=1 or b.m['workers'] not in (2,4):
        raise ValueError('invalid arm mapping')
    if not a.s['nonformal'] and b.m['workers']!=4:raise ValueError('formal thread workers')
    d=difference(a.s,b.s)
    if d:raise ValueError('semantic identity mismatch '+repr(d))
    first=None; per_update={}; structural={}; missing=[0,0]; count=[0,0]
    # Scratch index is outside the immutable evidence set, under approved TMPDIR.
    # Validate both full streams before judging equality; joint omissions fail too.
    with tempfile.TemporaryDirectory(prefix='pv26-align-') as scratch:
        with closing(sqlite3.connect(str(Path(scratch)/'alignment.sqlite'))) as index:
            index.execute('CREATE TABLE events (side INTEGER, identity TEXT, pos INTEGER, row TEXT, PRIMARY KEY(side, identity))')
            index.execute('CREATE INDEX identity_index ON events(identity)')
            for side,bundle in enumerate((a,b)):
                for pos,row in enumerate(bundle.events()):
                    identity=json.dumps([row['kind'],row['key']],separators=(',',':'))
                    index.execute('INSERT INTO events VALUES (?,?,?,?)',(side,identity,pos,json.dumps(row)))
                    count[side]+=1
                index.commit()
            query = """SELECT l.row,r.row FROM events l LEFT JOIN events r
                ON r.side=1 AND r.identity=l.identity WHERE l.side=0
                UNION ALL SELECT NULL,r.row FROM events r WHERE r.side=1
                AND NOT EXISTS (SELECT 1 FROM events l WHERE l.side=0 AND l.identity=r.identity)"""
            # Order by observable source position, never event-kind lexical order.
            index.execute('CREATE TABLE aligned (position INTEGER, leftrow TEXT, rightrow TEXT)')
            for lx,ry in index.execute(query):
                row=json.loads(lx or ry)
                identity=json.dumps([row['kind'],row['key']],separators=(',',':'))
                position=index.execute('SELECT MIN(pos) FROM events WHERE identity=?',(identity,)).fetchone()[0]
                index.execute('INSERT INTO aligned VALUES (?,?,?)',(position,lx,ry))
            for lx,ry in index.execute('SELECT leftrow,rightrow FROM aligned ORDER BY position,rowid'):
                x=json.loads(lx) if lx else None; y=json.loads(ry) if ry else None
                if x is None or y is None:
                    d=dict(path='$event',left=None if x is None else [x['kind'],x['key']],right=None if y is None else [y['kind'],y['key']],structural=True)
                else:d=difference(x['data'],y['data'],readers=(a.reader,b.reader))
                if d:
                    row=x or y
                    result=dict(event=[row['kind'],row['key']],difference=d,cause='observed difference; ultimate cause not proven')
                    if d.get('structural'):
                        missing[0]+=x is None;missing[1]+=y is None
                        structural.setdefault(str(row['key'][0]),result)
                    if first is None:first=result
                    per_update.setdefault(str(row['key'][0]),result)
    per_update_weights={}; earliest_weight_difference=None
    # Independent of trace traversal: resolve every named tensor at every update.
    for u in range(a.s['updates']+1):
        ca=json.loads((a.path/f'checkpoint-{u:02d}.json').read_text())['model']
        cb=json.loads((b.path/f'checkpoint-{u:02d}.json').read_text())['model']
        if set(ca)!=set(cb):raise ValueError('named model coverage mismatch')
        named={name:difference(ca[name],cb[name],readers=(a.reader,b.reader)) for name in ca}
        equal=all(d is None for d in named.values())
        per_update_weights[str(u)]=dict(weights_bitwise_equal=equal,named_actor_critic=named)
        if not equal and earliest_weight_difference is None:
            earliest_weight_difference=dict(update=u,named_actor_critic={n:d for n,d in named.items() if d is not None})
    final_equal=per_update_weights[str(a.s['updates'])]['weights_bitwise_equal']
    from .run import source_digest
    if source_digest()!=a.s['source_tree_sha']:raise ValueError('execution source changed during comparison')
    return dict(status='complete_equal' if final_equal else 'complete_different',formal=not a.s['nonformal'],
                weights_bitwise_equal=final_equal,full_trace_equal=first is None,
                final_update=a.s["updates"],per_update_weights=per_update_weights,
                earliest_weight_difference=earliest_weight_difference,
                all_updates_weights_bitwise_equal=all(v["weights_bitwise_equal"] for v in per_update_weights.values()),
                trace_status="equal" if first is None else "different",
                first_difference=first,first_difference_per_update=per_update,final_named_actor_critic=named,
                structural_first_difference_per_update=structural,missing_event_counts=missing,
                events=count,causal_stage_chain=['RNG','P input/output','actions','S/G','batch/GAE','gradients','Adam'],
                interpretation='Execution diagnostic only; no strength, promotion, or uninstrumented speedup conclusion.')


def compare(control,threaded,expected_commit=None):
    bundles=[]
    try:
        for path in (control,threaded):bundles.append(Bundle(path,expected_commit))
        return _compare(*bundles)
    finally:
        for bundle in bundles:bundle.reader.db.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit');p.add_argument('control');p.add_argument('threaded');p.add_argument('--out-report',required=True)
    args=p.parse_args(argv)
    try:
        report=compare(args.control,args.threaded,args.expected_commit)
        code=0 if report['weights_bitwise_equal'] else 2
    except InterruptedError:
        raise
    except (OSError,ValueError,KeyError,TypeError,EOFError,sqlite3.DatabaseError,zlib.error) as e:
        report=dict(status='incomplete_or_invalid',error=repr(e));code=3
    json_write(args.out_report,report)
    print(json.dumps(report,indent=2))
    return code


def cli(argv=None):
    from .cli import boundary
    return boundary(main,argv)


if __name__=='__main__':raise SystemExit(cli())
