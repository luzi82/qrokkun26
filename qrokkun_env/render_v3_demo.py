#!/usr/bin/env python3
"""Render both_v3 player vs spawner demo mp4."""
from __future__ import annotations
import argparse, math, shutil, subprocess
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR, Qrokkun26Env, _move_toward, _spawn_interval
from qrokkun_env.godot_rng import f32
from qrokkun_env.obs_rich import OBS_DIM_RICH, vectorize_rich
from qrokkun_env.render_player_demo import load_sprite, paste_centered, BULLET_COLORS
from qrokkun_env.agents.player_v3 import PlayerV3 as PlayerAC
from qrokkun_env.agents.spawner_v3 import SpawnerV3 as SpawnerAC, SPAWNER_ACTIONS, spawn_from_action

@torch.no_grad()
def p_act(net, env, device):
    x = torch.tensor(vectorize_rich(env), dtype=torch.float32, device=device)
    return int(net(x)[0].probs.argmax().item())

@torch.no_grad()
def s_act(net, env, device):
    x = torch.tensor(vectorize_rich(env), dtype=torch.float32, device=device)
    return int(net(x)[0].probs.argmax().item())

def render(env, scale, ps, bs, last, title):
    W, H = int(C.VIEW_W)*scale, int(C.VIEW_H)*scale
    img = Image.new('RGBA', (W,H), (18,18,28,255)); d = ImageDraw.Draw(img)
    fx0,fy0 = int(C.FIELD_X*scale), int(C.FIELD_Y*scale)
    fx1,fy1 = int((C.FIELD_X+C.FIELD_W)*scale), int((C.FIELD_Y+C.FIELD_H)*scale)
    d.rectangle((fx0,fy0,fx1,fy1), fill=(28,32,48,255), outline=(90,100,140,255))
    for b in env.bullets:
        paste_centered(img, bs.get(b.kind, bs[0]), b.x, b.y, scale)
    paste_centered(img, ps.get(last, ps['idle']), env.px, env.py, scale)
    d.rectangle((0,0,W,14), fill=(10,10,16,230))
    d.text((6,2), f'{title}  t={env.elapsed:5.2f}s  bullets={len(env.bullets):3d}', fill=(220,230,255,255))
    return img.convert('RGB')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--player', type=Path, default=Path('runs/both_v3_player.pt'))
    ap.add_argument('--spawner', type=Path, default=Path('runs/both_v3_spawner.pt'))
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--scale', type=int, default=3)
    ap.add_argument('--max-seconds', type=float, default=45.0)
    ap.add_argument('--out', type=Path, default=Path('dist/both_v3_demo.mp4'))
    ap.add_argument('--assets', type=Path, default=Path('assets'))
    args = ap.parse_args()
    device = torch.device('cpu')
    pck = torch.load(args.player, map_location=device, weights_only=False)
    sck = torch.load(args.spawner, map_location=device, weights_only=False)
    hidden = int(pck.get('hidden', 512))
    player = PlayerAC(hidden=hidden).to(device)
    player.load_state_dict(pck['state_dict']); player.eval()
    spawner = SpawnerAC(hidden=int(sck.get('hidden', hidden)), n_actions=int(sck.get('n_actions', SPAWNER_ACTIONS))).to(device)
    spawner.load_state_dict(sck['state_dict']); spawner.eval()

    player_map = {"idle":"player.png","n":"player_n.png","ne":"player_ne.png","e":"player_e.png","se":"player_se.png","s":"player_s.png","sw":"player_sw.png","w":"player_w.png","nw":"player_nw.png"}
    ps = {k: load_sprite(args.assets/fn, int(C.PLAYER_RADIUS), (240,240,250)) for k,fn in player_map.items()}
    bf = {0:"bullet_small.png",1:"bullet_med.png",2:"bullet_big.png",3:"bullet_lime.png"}
    bs = {k: load_sprite(args.assets/fn, int(C.BULLET_RADIUS[k]), BULLET_COLORS[k]) for k,fn in bf.items()}

    env = Qrokkun26Env(seed=args.seed); env.reset(seed=args.seed)
    frames = args.out.with_suffix('').parent / '_v3_demo_frames'
    if frames.exists(): shutil.rmtree(frames)
    frames.mkdir(parents=True)
    last='idle'; title='v3 player vs v3 spawner'
    max_steps = int(args.max_seconds / C.DT)
    n=0
    for i in range(max_steps):
        render(env, args.scale, ps, bs, last, title).save(frames/f'f{i:06d}.png')
        env.elapsed += env.dt; env.spawn_acc += env.dt
        interval = _spawn_interval(env.elapsed); spawns=0
        while env.spawn_acc >= interval:
            env.spawn_acc -= interval
            spawn_from_action(env, s_act(spawner, env, device), rng_jitter=False)
            spawns += 1
            thr=8.0
            p_double = 0.20 if env.elapsed>18.0 else (0.10 if env.elapsed>thr else 0.0)
            if spawns==1 and p_double>0 and env.rng.randf()<p_double:
                env.spawn_acc += interval
            interval = _spawn_interval(env.elapsed)
        a = p_act(player, env, device); last = ACTIONS[a]
        dx,dy = ACTION_TO_DIR[last]
        if dx or dy:
            nn=math.hypot(dx,dy); dx,dy=dx/nn,dy/nn
            env.pvx,env.pvy=_move_toward(env.pvx,env.pvy,dx*C.PLAYER_MAX_SPEED,dy*C.PLAYER_MAX_SPEED,C.PLAYER_ACCEL*env.dt)
        else:
            env.pvx=env.pvy=0.0
        env.px=f32(env.px+f32(env.pvx*env.dt)); env.py=f32(env.py+f32(env.pvy*env.dt))
        env.px=f32(min(max(env.px,C.FIELD_X+C.PLAYER_MARGIN),C.FIELD_X+C.FIELD_W-C.PLAYER_MARGIN))
        env.py=f32(min(max(env.py,C.FIELD_Y+C.PLAYER_MARGIN),C.FIELD_Y+C.FIELD_H-C.PLAYER_MARGIN))
        env._integrate_bullets()
        n=i+1
        if env._check_hit():
            env.dead=True
            for j in range(30):
                render(env, args.scale, ps, bs, last, title+'  HIT').save(frames/f'f{i+1+j:06d}.png')
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(['ffmpeg','-y','-framerate','60','-i',str(frames/'f%06d.png'),
        '-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(args.out)])
    shutil.rmtree(frames, ignore_errors=True)
    print(f'wrote {args.out} frames={n} elapsed={env.elapsed:.2f}s dead={env.dead}', flush=True)

if __name__ == '__main__':
    main()
