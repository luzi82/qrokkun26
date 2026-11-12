"""Typed bundle contracts and exact CPU environment evidence replay.

Replay uses the traced frozen S weights and recorded P actions, never checkpoint
files or training. Only one update's live episodes are retained.
"""
import re
import json
import struct
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v24.core import Spawner
from .trace import Writer, digest


class Encoder(Writer):
    def __init__(self): pass
    def blob(self, raw, dtype, shape):
        return dict(blob=digest(raw),dtype=dtype,shape=list(shape),bytes=len(raw))


encode=Encoder().encode


def tensor(reader, ref, shape=None, dtype='torch.float32'):
    if not isinstance(ref,dict) or set(ref)!={'blob','shape','dtype','bytes'}:
        raise ValueError('required typed raw tensor reference')
    if ref['dtype']!=dtype or (shape is not None and ref['shape']!=list(shape)):
        raise ValueError('tensor shape/dtype mismatch')
    raw=reader.raw(ref)
    return torch.frombuffer(bytearray(raw),dtype=getattr(torch,dtype[6:])).reshape(ref['shape']).clone()


def model(reader, refs, template):
    expected=template.state_dict()
    if set(refs)!=set(expected):raise ValueError('missing named model field')
    return {n:tensor(reader,refs[n],v.shape,str(v.dtype)) for n,v in expected.items()}


def schema(s,m,expected_commit=None):
    from .run import source_digest, HYPER
    ints={'updates':20,'episodes':12,'cap':7200,'seed':17,'validation':16,'hidden':472}
    for k,maximum in ints.items():
        if type(s[k]) is not int or not 0<s[k]<=maximum:raise ValueError('semantic limits '+k)
    for k in ('nonformal','synthetic','deterministic','resume'):
        if type(s[k]) is not bool:raise ValueError('semantic flag '+k)
    if s['resume'] or s['heldout']!=[] or type(s['heldout']) is not list:raise ValueError('resume/heldout contract')
    if type(s['validation_seeds']) is not list or any(type(v) is not int for v in s['validation_seeds']):raise ValueError('validation seed schema')
    for k,n in [('source_commit',40),('source_tree_sha',64)]:
        if not isinstance(s[k],str) or not re.fullmatch('[0-9a-fA-F]{'+str(n)+'}',s[k]):raise ValueError('source identity '+k)
    if s['source_tree_sha']!=source_digest():raise ValueError('execution source digest mismatch; use comparator from same frozen archive')
    if expected_commit is not None and s['source_commit']!=expected_commit:raise ValueError('expected parent commit mismatch')
    for k,v in dict(P_seed_offset=26000000,S_train_offset=25000000,S_val_offset=24000000,permutation_seed=27000000+s['seed']).items():
        if type(s[k]) is not int or s[k]!=v:raise ValueError('RNG mapping '+k)
    if s['hyperparameters']!=HYPER or any(type(s['hyperparameters'][k])!=type(v) for k,v in HYPER.items()):raise ValueError('hyperparameter schema')
    with torch.random.fork_rng(devices=[]):p=PlayerV21(hidden=s['hidden'])
    if json.dumps(s['architecture'],sort_keys=True)!=json.dumps(p.architecture_metadata(),sort_keys=True):raise ValueError('architecture schema')
    r=s['runtime']
    types=dict(torch=str,numpy=str,threads=int,device=str,deterministic=bool,cudnn_deterministic=bool,cudnn_benchmark=bool,matmul_precision=str,cuda_matmul_allow_tf32=bool,cudnn_allow_tf32=bool)
    for k,t in types.items():
        if type(r[k]) is not t:raise ValueError('runtime schema '+k)
    for k in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if r[k] is not None and type(r[k]) is not str:raise ValueError('runtime environment '+k)
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?',r['device']) or r['matmul_precision'] not in ('highest','high','medium'):raise ValueError('runtime device/precision')
    if not s['nonformal'] and torch.device(r['device']).type!='cuda':raise ValueError('formal execution requires CUDA')
    if type(m['workers']) is not int or type(m['arm']) is not str or type(m['trace_dir']) is not str:raise ValueError('manifest execution schema')
    if r['threads']!=1 or r['deterministic']!=s['deterministic']:raise ValueError('runtime flags')
    for k in ('player_sha','player_state_sha','spawner_sha','spawner_state_sha'):
        if not isinstance(s[k],str) or not (re.fullmatch('[0-9a-f]{64}',s[k]) or (s['synthetic'] and s[k]=='synthetic')):raise ValueError('input identity '+k)
    for k,v in dict(schema=2,trace_instrumented=True,uninstrumented_speedup_claim=False,trace_durable_bundle=False).items():
        if type(m[k])!=type(v) or m[k]!=v:raise ValueError('manifest flag '+k)
    return p


def snapshot(reader,data,p,kind,key):
    model(reader,data['model'],p)
    shapes={n:list(v.shape) for n,v in p.named_parameters()}
    if set(data['gradients'])!=set(shapes):raise ValueError('missing named gradient field')
    fresh=key[0]==0
    # Actor gradients exist after its first backward; critic gradients only after
    # its first backward. Subsequent minibatches retain previous gradients.
    first=key[0]==1 and len(key)>1 and key[1]==0
    actor_ready=not fresh and not (first and kind=='minibatch')
    critic_ready=not fresh and not (first and kind in ('minibatch','actor_preclip','actor_postclip','actor_prestep','actor_poststep'))
    for n,shape in shapes.items():
        ready=actor_ready if n.startswith('policy') else critic_ready
        ref=data['gradients'][n]
        if ready:tensor(reader,ref,shape)
        elif ref is not None:raise ValueError('gradient None stage legality')
    for arm,prefix in [('actor','policy'),('critic','value')]:
        opt=data[arm]; names=[n for n in shapes if n.startswith(prefix)]
        if set(opt)!={'state','param_groups'} or len(opt['param_groups'])!=1:raise ValueError('optimizer schema')
        group=opt['param_groups'][0]
        expected=adam_config(arm,len(names))
        from .compare import difference
        if difference(group,expected):raise ValueError('incomplete/typed Adam param_groups configuration')
        params=group['params']
        if params!=list(range(len(names))):raise ValueError('optimizer named parameter coverage')
        stepped=not fresh and not (first and (kind in ('minibatch','actor_preclip','actor_postclip','actor_prestep') or (arm=='critic' and kind in ('actor_poststep','critic_preclip','critic_postclip','critic_prestep'))))
        if not stepped:
            if opt['state']:raise ValueError('fresh Adam state must be empty')
        else:
            if set(opt['state'])!={str(i) for i in params}:raise ValueError('missing Adam parameter state')
            for i,n in enumerate(names):
                state=opt['state'][str(i)]
                if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('missing Adam field')
                tensor(reader,state['step'],[])
                for k in ('exp_avg','exp_avg_sq'):tensor(reader,state[k],shapes[n])


class Episodes:
    def __init__(self,reader,device):self.reader=reader;self.device=device;self.live={};self.s=None
    def check(self,k,key,d):
        from .kernel import Env
        from qrokkun_ai.v21.obs_v21 import encode_obs_v21
        import numpy as np
        if k=='opponent':
            with torch.random.fork_rng(devices=[]):self.s=Spawner().requires_grad_(False).eval()
            sd=model(self.reader,d['model'],self.s);self.s.load_state_dict(sd)
            if d['frozen'] is not True or d['device']!='cpu':raise ValueError('opponent contract')
            return state_dict_sha256(sd)
        if k=='reset':
            if self.s is None:raise ValueError('missing opponent')
            ep=tuple(key[:3]); e=Env(self.s,d['seed'],key[1]=='validation')
            expected=dict(seed=d['seed'],state=e.state(),initial_generators=e.initial_generators,events=e.drain())
            # CUDA P generators have a different raw layout. Their initial state
            # is checked by continuity; S and Python replay always run on CPU.
            expected=encode(expected)
            initial=d['initial_generators']['P']
            tensor(self.reader,initial,dtype='torch.uint8')
            if self.device.startswith('cuda'):
                if initial['shape']!=[16] or self.reader.raw(initial)!=struct.pack('<QQ',d['seed']+26000000,0):raise ValueError('CUDA P reset seed/offset state')
            elif initial!=expected['initial_generators']['P']:raise ValueError('P reset seed state')
            expected['initial_generators']['P']=d['initial_generators']['P'];expected['state']['P']=d['initial_generators']['P']
            if expected!=d:raise ValueError('reset state/events evidence mismatch')
            self.live[ep]=dict(env=e,P=d['state']['P'],count=0,action=None)
        if k=='P':
            epbase=tuple(key[:2]); active=[ep[2] for ep,v in self.live.items() if ep[:2]==epbase and not v['env'].dead and v['count']==key[2]]
            if any(type(i) is not int for i in d['active']) or d['active']!=active:raise ValueError('P active order/coverage')
            draws=d['draws']
            if len(draws)!=(len(active) if key[1]=='train' else 0):raise ValueError('P draw coverage')
            actions=tensor(self.reader,d['actions'],[len(active)],'torch.int64')
            for j,i in enumerate(active):
                v=self.live[epbase+(i,)];v['action']=int(actions[j])
                if draws:
                    draw=draws[j]
                    if set(draw)!={'source','counter','pre','post','output'} or draw['source']!='P.multinomial' or (type(draw['counter']) is not int or draw['counter']!=v['count']) or draw['pre']!=v['P']:raise ValueError('P draw chain/counter')
                    tensor(self.reader,draw['pre'],v['P']['shape'],dtype='torch.uint8');tensor(self.reader,draw['post'],v['P']['shape'],dtype='torch.uint8')
                    output=tensor(self.reader,draw['output'],[1],'torch.int64')
                    if int(output[0])!=v['action']:raise ValueError('P output/action')
                    if draw['pre']['shape']==list(torch.Generator().get_state().shape):
                        generator=torch.Generator();generator.set_state(tensor(self.reader,draw['pre'],dtype='torch.uint8'))
                        probs=tensor(self.reader,d['probs'])
                        sampled=torch.multinomial(probs[j],1,generator=generator)
                        if encode(sampled)!=draw['output'] or encode(generator.get_state())!=draw['post']:raise ValueError('P actual draw output/state')
                    v['P']=draw['post']
        if k=='G':
            v=self.live[tuple(key[:3])];e=v['env']
            if d['action']!=v['action']:raise ValueError('G P action mismatch')
            _,expected,_=e.traced_step(d['action']); expected=encode(expected)
            expected['before']['P']=expected['after']['P']=v['P']
            for name in expected:
                if expected[name]!=d[name]:raise ValueError('G state/events evidence mismatch '+name)
            v['count']+=1
        if k=='bootstrap':
            active=[ep[2] for ep,v in self.live.items() if ep[:2]==tuple(key) and not v['env'].dead]
            if not active or d['active']!=active or any(type(i) is not int for i in d['active']):raise ValueError('bootstrap active order/coverage')
            obs=[encode_obs_v21(self.live[tuple(key)+(i,)]['env']) for i in active]
            for j,name in enumerate(('player','bullets','pad')):
                if d[name]!=encode(torch.tensor(np.stack([o[j] for o in obs]))):raise ValueError('bootstrap final observation mismatch')
            values=tensor(self.reader,d['values'],[len(active)])
            for i,v in zip(active,values.tolist()):self.live[tuple(key)+(i,)]['bootstrap']=encode(float(v))
        if k=='episode_end':
            v=self.live.pop(tuple(key[:3]));e=v['env'];state=encode(e.state());state['P']=v['P']
            if d['bootstrap']!=v.get('bootstrap',encode(0.0)):raise ValueError('episode bootstrap float bits mismatch')
            if not e.dead and 'bootstrap' not in v:raise ValueError('missing bootstrap evidence')
            counts={n:getattr(e,n).counter for n in ('rng','s_rng','field_rng')}
            if any(type(d[n]) is not bool for n in ('terminated','truncated','censored')) or d['terminated']!=e.dead or d['truncated']!= (not e.dead) or d['censored']!= (not e.dead) or d['frames']!=v['count']:raise ValueError('episode end termination/frame state')
            if any(type(c) is not int for c in d['rng_draw_counts'].values()) or any(type(d[n]) is not int for n in ('S_draw_count','P_draw_count')):raise ValueError('episode end counter types')
            if d['state']!=state or d['S_draw_count']!=e.counter[0] or d['P_draw_count']!=(v['count'] if key[1]=='train' else 0) or d['rng_draw_counts']!=counts or d['unused']!=[n for n,c in counts.items() if c==0]:raise ValueError('episode end state/counters/unused')


def fields(value, names):
    if not isinstance(value,dict) or set(value)!=set(names):
        raise ValueError('incomplete/unknown scientific fields '+str(names))


def scalar(reader, ref):
    if not isinstance(ref,dict) or set(ref)!={'blob','shape','dtype','bytes'} or ref['shape']!=[] or ref['dtype']!='python.float64.be':
        raise ValueError('required raw Python scalar bits')
    return struct.unpack('>d',reader.raw(ref))[0]


def rng(reader, data):
    fields(data,('python','numpy_legacy','numpy_generators','torch_cpu','torch_cuda'))
    state=data['python']
    if not isinstance(state,list) or len(state)!=3 or type(state[0]) is not int or state[0]!=3 or not isinstance(state[1],list) or len(state[1])!=625 or any(type(x) is not int for x in state[1]):raise ValueError('Python RNG state')
    import random
    random.Random().setstate((3,tuple(state[1]),None if state[2] is None else scalar(reader,state[2])))
    state=data['numpy_legacy']
    if not isinstance(state,list) or len(state)!=5 or state[0]!='MT19937' or type(state[2]) is not int or not 0<=state[2]<=624 or type(state[3]) is not int or state[3] not in (0,1):raise ValueError('NumPy RNG state')
    array(reader,state[1],[624],'<u4');scalar(reader,state[4])
    # The producer creates no NumPy Generators. An empty registry is normative.
    if data['numpy_generators']!={}:raise ValueError('unexpected NumPy Generator registry')
    tensor(reader,data['torch_cpu'],torch.get_rng_state().shape,'torch.uint8')
    if not isinstance(data['torch_cuda'],list):raise ValueError('CUDA RNG state list')
    for state in data['torch_cuda']:tensor(reader,state,[16],'torch.uint8')


def array(reader,ref,shape,dtype):
    import numpy as np
    if not isinstance(ref,dict) or set(ref)!={'blob','shape','dtype','bytes'} or ref['shape']!=list(shape) or ref['dtype']!='numpy:'+np.dtype(dtype).str:raise ValueError('required typed NumPy array')
    return np.frombuffer(reader.raw(ref),dtype=dtype).reshape(shape).copy()


def execution(s):
    """Producer-order grammar; G order is canonical episode index, not walltime."""
    def collection(u,phase):
        n=s['episodes'] if phase=='train' else s['validation']
        for i in range(n):yield 'reset',[u,phase,i,0]
        active=list(range(n));frames=0
        for f in range(s['cap']):
            if not active:break
            yield 'P',[u,phase,f]
            remaining=[]
            for i in active:
                row=yield 'G',[u,phase,i,f+1]
                frames+=1
                if not row['data']['done']:remaining.append(i)
            active=remaining
        if active:yield 'bootstrap',[u,phase]
        for i in range(n):yield 'episode_end',[u,phase,i]
        return frames
    yield 'global_init',[0]
    yield 'opponent',[0]
    yield 'checkpoint',[0]
    yield from collection(0,'validation')
    stages=('minibatch','actor_preclip','actor_postclip','actor_prestep','actor_poststep','critic_preclip','critic_postclip','critic_prestep','critic_poststep')
    for u in range(1,s['updates']+1):
        n=yield from collection(u,'train')
        yield 'rollouts',[u]
        yield 'batch',[u]
        step=0
        for epoch in range(4):
            yield 'permutation',[u,epoch]
            for start in range(0,n,512):
                for stage in stages:yield stage,[u,step]
                step+=1
        yield 'checkpoint',[u]
        if u in (10,20) or (s['nonformal'] and u==s['updates']):yield from collection(u,'validation')
        yield 'update_end',[u]
    yield 'global_final',[s['updates']]


class Scientific:
    """Resolve every scientific position, retaining at most one update's rollouts."""
    def __init__(self,reader,s,p):
        self.reader=reader;self.s=s;self.p=p;self.ros=[];self.pending={};self.n=0;self.mb=0
        self.order=execution(s);self.expected=next(self.order);self.training=None

    def check(self,k,key,d):
        import numpy as np
        from qrokkun_ai.v21.train import train_player_v21 as v21
        r=self.reader
        if self.expected!=(k,key):raise ValueError('scientific execution order: expected '+repr(self.expected)+' got '+repr((k,key)))
        try:self.expected=self.order.send(dict(data=d))
        except StopIteration:self.expected=None
        if k in ('global_init','global_final'):
            rng(r,d)
            if self.s['runtime']['device'].startswith('cuda') and not d['torch_cuda']:raise ValueError('missing CUDA RNG states')
        for name in ('global_rng','global_pre','global_post'):
            if name in d:
                rng(r,d[name])
                if self.s['runtime']['device'].startswith('cuda') and not d[name]['torch_cuda']:raise ValueError('missing CUDA RNG states')
        if 'training_rng' in d:
            tensor(r,d['training_rng'],[16] if self.s['runtime']['device'].startswith('cuda') else torch.get_rng_state().shape,'torch.uint8')
            if k=='batch':self.training=d['training_rng']
            elif self.training is not None and d['training_rng']!=self.training:raise ValueError('training RNG continuity')
        if k=='reset' and key[1]=='train':
            if key[2]==0:self.ros=[]
            self.ros.append(v21.RolloutV21(seed=d['seed']))
        if k in ('P','bootstrap'):
            n=len(d['active'])
            if not n:raise ValueError('empty active batch')
            for name,tail,dt in [('player',[self.p.player_feat],'torch.float32'),('bullets',[self.p.max_bullets,self.p.bullet_feat],'torch.float32'),('pad',[self.p.max_bullets],'torch.bool'),('values',[],'torch.float32')]:tensor(r,d[name],[n]+tail,dt)
        if k=='bootstrap':tensor(r,d['features'],[len(d['active']),self.p.in_dim])
        if k=='P':
            n=len(d['active'])
            for name,tail,dt in [('features',[self.p.in_dim],'torch.float32'),('raw_logits',[9],'torch.float32'),('logits',[9],'torch.float32'),('probs',[9],'torch.float32'),('teacher_logits',[9],'torch.float32'),('teacher_probs',[9],'torch.float32'),('teacher_actions',[],'torch.int64'),('actions',[],'torch.int64'),('logp',[],'torch.float32')]:tensor(r,d[name],[n]+tail,dt)
            if key[1]=='train':
                values={name:tensor(r,d[name]).numpy() for name in ('player','bullets','values','logp','teacher_logits')}
                values['pad']=tensor(r,d['pad'],dtype='torch.bool').numpy()
                for name in ('actions','teacher_actions'):values[name]=tensor(r,d[name],dtype='torch.int64').numpy()
                for j,i in enumerate(d['active']):self.pending[i]={name:v[j].copy() for name,v in values.items()}
        if k=='G' and key[1]=='train':
            ro=self.ros[key[2]];v=self.pending.pop(key[2])
            for name in ('player','bullets','pad'):getattr(ro,name).append(v[name])
            for name,source,convert in [('actions','actions',int),('log_probs','logp',float),('values','values',float),('expert_actions','teacher_actions',int)]:getattr(ro,name).append(convert(v[source]))
            ro.expert_logits.append(v['teacher_logits']);ro.rewards.append(scalar(r,d['reward']));ro.dones.append(d['done'])
        if k=='episode_end':
            scalar(r,d['bootstrap'])
            if key[1]=='train':
                ro=self.ros[key[2]]
                for name in ('terminated','truncated','censored'):setattr(ro,name,d[name])
                ro.bootstrap_value=scalar(r,d['bootstrap']);ro.elapsed=scalar(r,d['state']['elapsed'])
        if k=='rollouts':
            expected=dict(flattened_order=[(ro.seed,len(ro.actions)) for ro in self.ros],rollouts=[{name:np.asarray(v) if isinstance(v,list) else v for name,v in vars(ro).items()} for ro in self.ros])
            if d!=encode(expected):raise ValueError('rollout contents/length/order mismatch')
            self.n=sum(len(ro.actions) for ro in self.ros)
        if k=='batch':
            expected=v21.rollouts_to_batch_v21(self.ros,.9995,.999,torch.device('cpu'),reward_scale=.05)
            if d['batch']!=encode(expected):raise ValueError('batch typed contents/GAE/order mismatch')
            self.ros=[]
        if k=='permutation':
            if d['source']!='PPO.randperm' or type(d['counter']) is not int or d['counter']!=key[1] or d['pre']!=self.training:raise ValueError('permutation source/counter/RNG')
            shape=[16] if self.s['runtime']['device'].startswith('cuda') else list(torch.get_rng_state().shape)
            for name in ('pre','post'):tensor(r,d[name],shape,'torch.uint8')
            self.perm=tensor(r,d['output'],[self.n],'torch.int64')
            if not torch.equal(self.perm.sort().values,torch.arange(self.n)):raise ValueError('permutation indices coverage')
            self.offset=0;self.training=d['post']
        if k=='minibatch':
            expected=self.perm[self.offset:self.offset+512];self.mb=len(expected);self.offset+=self.mb
            if not torch.equal(tensor(r,d['indices'],[self.mb],'torch.int64'),expected):raise ValueError('minibatch/permutation indices order')
            tensor(r,d['normalized_advantages'],[self.mb])
        if k.startswith(('actor_','critic_')):
            tensor(r,d['loss'],[])
            if k.startswith('actor_'):
                fields(d['losses'],('policy','entropy','hard_ce','soft_ce','dagger'))
                for ref in d['losses'].values():tensor(r,ref,[])
                for name,shape in [('features',[self.mb,self.p.in_dim]),('raw_logits',[self.mb,9]),('probs',[self.mb,9]),('teacher_probs',[self.mb,9]),('logits',[self.mb,9]),('values',[self.mb])]:tensor(r,d[name],shape)
        if k=='update_end':
            fields(d['metrics'],('policy_loss','value_loss','dagger_loss','approx_kl','expert_agreement','explained_variance','n_samples'))
            if type(d['frames']) is not int or d['frames']!=self.n or type(d['metrics']['n_samples']) is not int or d['metrics']['n_samples']!=self.n:raise ValueError('update sample count')
            for name,ref in d['metrics'].items():
                if name!='n_samples':scalar(r,ref)  # one-sample variance NaN is legitimate

    def finish(self):
        if self.expected is not None:raise ValueError('incomplete scientific execution order '+repr(self.expected))


from functools import lru_cache


@lru_cache(None)
def adam_config(arm,n):
    # Derive defaults from the producer's actual installed Adam implementation.
    opt=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))],lr=3e-4 if arm=='actor' else 1e-3,eps=1e-8)
    group=opt.state_dict()['param_groups'][0]
    group['params']=list(range(n))
    return encode(group)
