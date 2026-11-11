"""CPU-only, diagnostic Sv24 versus frozen P renderer."""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from .core import LearnedEnv, load_player, file_hash, PLAYER_SHA, STATE_SHA
from .core import load_spawner as production_load_spawner
from .run import player_actions
from qrokkun_env.env import ACTIONS
from qrokkun_ai.v21.render_player_v21_demo import render_frame, load_bullet_sprites

def load_spawner(path, *, expected_update=100, expected_spawner_sha256):
    if file_hash(path) != expected_spawner_sha256:
        raise ValueError('Sv checkpoint identity mismatch')
    net, ck = production_load_spawner(path)
    if ck.get('completed_updates') != expected_update or ck.get('partial_update') is not False:
        raise ValueError('Sv completed update/partial identity mismatch')
    if ck.get('player_sha') != PLAYER_SHA or ck.get('player_state_sha') != STATE_SHA:
        raise ValueError('Sv frozen player identity mismatch')
    return net, ck

@torch.no_grad()
def rollout(net,player,seed,*,cap=7200,on_frame=None):
    rng=torch.Generator().manual_seed(seed+24000000)
    def callback(obs):
        action, _ = net.sample(torch.from_numpy(obs).unsqueeze(0), rng, deterministic=False)
        return action[0].numpy()
    env=LearnedEnv(callback); env.reset(seed)
    while not env.dead and env.frame<cap:
        action=player_actions(player,[env],'cpu')[0]
        env.step(action)
        if on_frame: on_frame(env,ACTIONS[action])
    return dict(seed=seed,frames=env.frame,seconds=env.frame/60,death=env.dead,
                censored=not env.dead,cap_frames=cap)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spawner-checkpoint',required=True,type=Path)
    p.add_argument('--player-checkpoint',required=True,type=Path)
    p.add_argument('--assets',required=True,type=Path)
    p.add_argument('--output','--out',dest='output',required=True,type=Path)
    p.add_argument('--scale',type=int,choices=[3],default=3)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--expected-update',type=int,default=100)
    p.add_argument('--expected-spawner-sha256',required=True)
    args=p.parse_args(); torch.set_num_threads(1)
    if args.output.exists() or args.output.with_suffix(args.output.suffix+'.json').exists(): p.error('output already exists')
    if not args.assets.is_absolute(): p.error('--assets must be absolute')
    bullets=load_bullet_sprites(args.assets)
    sprites={}
    for action in ACTIONS:
        path=args.assets/('player.png' if action=='idle' else f'player_{action}.png')
        with Image.open(path) as im: sprites[action]=im.convert('RGBA')
    net,ck=load_spawner(args.spawner_checkpoint,expected_update=args.expected_update,
                     expected_spawner_sha256=args.expected_spawner_sha256)
    actual_sha=file_hash(args.spawner_checkpoint)
    actual_update=ck['completed_updates']
    player,meta=load_player(args.player_checkpoint,device='cpu')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='sv24-render-'))
    count=0; last=None
    def draw(env,action,hit_step=None):
        nonlocal count,last
        last=(env,action)
        im=render_frame(env,action,args.assets,3,bullet_sprites=bullets,
                        player_sprites=sprites,seed=args.seed,hit_step=hit_step)
        ImageDraw.Draw(im).text((12,28),f'Sv24 u{actual_update} vs Pv21 u{meta["extra"]["update"]} | diagnostic only',fill='white')
        im.convert('RGB').save(work/f'{count:06d}.png',compress_level=1)
        count+=1
    try:
        result=rollout(net,player,args.seed,on_frame=draw)
        if result['death']:
            for step in range(30): draw(*last,hit_step=step)
        subprocess.run(['ffmpeg','-n','-threads','2','-framerate','60','-i',str(work/'%06d.png'),
            '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movie_timescale','15360',str(args.output)],check=True)
        sidecar=dict(result=result,spawner_checkpoint=str(args.spawner_checkpoint.resolve()),
            player_checkpoint=str(args.player_checkpoint.resolve()),mode='stochastic',
            player_mode='deterministic argmax',fixed_cap_frames=7200,device='cpu',
            python=__import__('sys').executable,spawner_sha256=actual_sha,spawner_update=actual_update,
            player_sha256=file_hash(args.player_checkpoint),player_state_sha256=meta['state_dict_sha256'],
            player_update=meta['extra']['update'],video_frames=count,hit_tail_frames=count-result['frames'],
            diagnostic_only=True,promotion_eligible=False,visualization_only=True,
            visualization='Python diagnostic visualization (not Godot)',train_rng=None,spawner_partial_update=False,
            rng={'mode':'stochastic','generator':'torch CPU','seed':args.seed+24000000,
                 'derivation':'seed + 24000000 (run.episodes evaluation; train_rng=None)',
                 'sampling':'Sv24 net.sample(x, rng, deterministic=False): wrapped birth/fire and bounded speed'},**result)
        args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(sidecar,indent=2)+'\n')
    finally: shutil.rmtree(work)

if __name__=='__main__': main()
