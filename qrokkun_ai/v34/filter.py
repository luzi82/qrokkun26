"""Conservative O(1) sufficient impossibility predicate; see README proof."""
import math
import struct
from qrokkun_env import constants as C


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def harmless(env, b):
    x, y, vx, vy, radius = map(float, (b.x,b.y,b.vx,b.vy,b.radius))
    px, py, dt = float(env.px),float(env.py),float(env.dt)
    if not all(map(math.isfinite,(x,y,vx,vy,radius,px,py,dt))) or radius < 0 or dt != C.DT:
        return False
    x0,x1,y0,y1 = C.BULLET_CULL
    if not (x0 <= x <= x1 and y0 <= y <= y1 and 14 <= px <= 306 and 26 <= py <= 226):
        return False
    if max(abs(vx*dt),abs(vy*dt)) > 1e6:
        return False
    qx,qy = f32(vx*dt),f32(vy*dt)
    # Every pre-step coordinate is inside cull. Include final removed step.
    ex = 2**-24*(344+abs(qx)) + 2**-149
    ey = 2**-24*(264+abs(qy)) + 2**-149
    bounds=[]
    for z,q,e,lo,hi in ((x,qx,ex,x0,x1),(y,qy,ey,y0,y1)):
        if abs(q)>e:
            ratio=((hi-z) if q>0 else (z-lo))/(abs(q)-e)
            # Avoid ill-conditioned enormous lifetime arithmetic. This retains
            # uncertainty; it never declares safety from a fixed horizon.
            if ratio <= 2**32:
                bounds.append(math.ceil(ratio)+2)
    if not bounds:
        return False
    n=min(bounds)
    delay=int(not b.moved)
    # Norm bound for any action, including diagonal rounding and position addition.
    speed = C.PLAYER_MAX_SPEED*dt*(1+2**-24)**2 + math.sqrt(2)*2**-24*512
    drift=math.hypot(ex,ey)
    r=C.PLAYER_RADIUS+radius
    dx,dy=x-px,y-py
    distance=math.hypot(dx,dy)
    if distance == 0:
        return False
    # Fixed separating hyperplane: lower bound affine in moving step m.
    slope=(dx*qx+dy*qy)/distance-speed-drift
    margin=distance-r-speed*delay+min(0.,n*slope)
    # Binary64 evaluation guard, derived generous bound on operation count.
    guard=128*2**-52*(1+distance+r+n*(abs(qx)+abs(qy)+speed+drift))
    if margin > guard:
        return True
    # Legal rectangle can prove safety even when radial reach disc is loose.
    for z,q,e,lo,hi in ((x,qx,ex,14,306),(y,qy,ey,26,226)):
        lower=min(z,z+n*q)-n*e
        upper=max(z,z+n*q)+n*e
        if lower-hi-r > guard or lo-upper-r > guard:
            return True
    return False
