import numpy as np, math, sys
sys.path.insert(0, 'visualization')
from toro_footstep_walk import DEFAULT_FOOTSTEPS, align_footsteps_to_stone1, load_run_json

xy_scale = 1.0
json_path = 'visualization/run_012.json'

fp = align_footsteps_to_stone1(DEFAULT_FOOTSTEPS, json_path, xy_scale=xy_scale)
stones = load_run_json(json_path)
for s in stones:
    s['cx'] *= xy_scale
    s['cy'] *= xy_scale

def point_in_stone(px, py, s):
    yaw = math.radians(s['yaw_deg'] + 90.0)
    dx, dy = px - s['cx'], py - s['cy']
    lx =  dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    inside = abs(lx) <= s['height']/2 and abs(ly) <= s['width']/2
    return inside, s['height']/2 - abs(lx), s['width']/2 - abs(ly)

print("  #  side   foot_x   foot_y     stone   in?  margin_h  margin_w")
for i, entry in enumerate(fp):
    side, px, py = entry[0], entry[1], entry[2]
    best = min(stones, key=lambda s: (px-s['cx'])**2+(py-s['cy'])**2)
    inside, mh, mw = point_in_stone(px, py, best)
    ok = "OK " if inside else "OUT"
    print("%3d  %4s  %7.3f  %7.3f  %8s  %3s  %+7.1fcm  %+7.1fcm" % (
        i, side, px, py, best['name'], ok, mh*100, mw*100))
