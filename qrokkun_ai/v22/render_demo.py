"""CPU-only, diagnostic Sv22 versus frozen P renderer."""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from .core import Spawner, LearnedEnv, load_player, file_hash, PLAYER_SHA
from .run import player_actions
from qrokkun_env.env import ACTIONS
from qrokkun_ai.v21.render_player_v21_demo import render_frame, load_bullet_sprites

SV_SHA='f9dd57396ac2ccf5ba5ea2c71c9272d710a6a98fe0e0b181c1bc346a24cd5325'

def load_spawner(path):
    if file_hash(path)!=SV_SHA: raise ValueError('Sv checkpoint identity mismatch')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['update']!=320 or ck['partial_update'] is not False:
        raise ValueError('Sv update identity mismatch')
    net=Spawner(); net.load_state_dict(ck['model'],strict=True)
    return net.eval()

@torch.no_grad()
def rollout(net,player,seed,*,cap=7200,on_frame=None):
    rng=torch.Generator().manual_seed(seed+7000000)
    def callback(obs):
        d=net.distribution(torch.from_numpy(obs).unsqueeze(0))
        return (d.mean+d.stddev*torch.randn(d.mean.shape,generator=rng))[0].numpy()
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
    args=p.parse_args(); torch.set_num_threads(1)
    if args.output.exists(): p.error('output already exists')
    if not args.assets.is_absolute(): p.error('--assets must be absolute')
    bullets=load_bullet_sprites(args.assets)
    sprites={}
    for action in ACTIONS:
        path=args.assets/('player.png' if action=='idle' else f'player_{action}.png')
        with Image.open(path) as im: sprites[action]=im.convert('RGBA')
    net=load_spawner(args.spawner_checkpoint)
    player,meta=load_player(args.player_checkpoint,device='cpu')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='sv22-render-'))
    count=0; last=None
    def draw(env,action,hit_step=None):
        nonlocal count,last
        last=(env,action)
        im=render_frame(env,action,args.assets,3,bullet_sprites=bullets,
                        player_sprites=sprites,seed=args.seed,hit_step=hit_step)
        ImageDraw.Draw(im).text((12,28),'Sv22 u320 vs Pv21 u400 | diagnostic only',fill='white')
        im.convert('RGB').save(work/f'{count:06d}.png',compress_level=1)
        count+=1
    try:
        result=rollout(net,player,args.seed,on_frame=draw)
        if result['death']:
            for step in range(30): draw(*last,hit_step=step)
        subprocess.run(['ffmpeg','-n','-threads','2','-framerate','60','-i',str(work/'%06d.png'),
            '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p',str(args.output)],check=True)
        sidecar=dict(result=result,spawner_checkpoint=str(args.spawner_checkpoint.resolve()),
            player_checkpoint=str(args.player_checkpoint.resolve()),mode='stochastic',
            player_mode='deterministic argmax',fixed_cap_frames=7200,device='cpu',
            python=__import__('sys').executable,spawner_sha256=SV_SHA,spawner_update=320,
            player_sha256=PLAYER_SHA,player_update=400,video_frames=count,hit_tail_frames=count-result['frames'],
            diagnostic_only=True,promotion_eligible=False,visualization_only=True,
            visualization='Python diagnostic visualization',spawner_partial_update=False,
            rng={'mode':'stochastic','generator':'torch CPU','seed':args.seed+7000000,
                 'derivation':'seed + 7000000 (run.episodes evaluation; train_rng=None)',
                 'sampling':'mean + stddev * torch.randn(mean.shape, generator=rng)'},**result)
        args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(sidecar,indent=2)+'\n')
    finally: shutil.rmtree(work)

if __name__=='__main__': main()
