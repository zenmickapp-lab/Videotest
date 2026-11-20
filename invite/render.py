import math, random, sys, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

D = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
DUR = 36.0
N = int(DUR * FPS)

def L(p): return Image.open(os.path.join(D, p)).convert('RGBA')
EM = {k: L(f'em/{k}.png') for k in ['lion','elephant','giraffe','zebra','monkey','monkey_face','tiger','hippopotamus','rhinoceros','birthday_cake','butterfly','balloon','hibiscus','palm_tree','herb','sunflower','sparkles','glowing_star','party_popper','wrapped_gift','paw_prints','leaf_fluttering_in_wind','crown']}
MIR = [L(f'cut/m{i}.png') for i in range(4)]

def font(sz, w='ExtraBold', f='Baloo2.ttf'):
    ft = ImageFont.truetype(os.path.join(D, 'fonts', f), sz)
    try: ft.set_variation_by_name(w)
    except Exception: pass
    return ft

# ---------- helpers ----------
def ease(t): t = max(0, min(1, t)); return t*t*(3-2*t)
def back(t):
    t = max(0, min(1, t)); c = 1.70158
    return 1 + (c+1)*(t-1)**3 + c*(t-1)**2
def seg(t, a, b): return max(0, min(1, (t-a)/(b-a)))

_cache = {}
def scaled(img, h, key=None, flip=False):
    h = max(2, int(h))
    k = (id(img), h, flip)
    if k in _cache: return _cache[k]
    w = max(2, int(img.width*h/img.height))
    r = img.resize((w, h), Image.LANCZOS)
    if flip: r = r.transpose(Image.FLIP_LEFT_RIGHT)
    if len(_cache) > 400: _cache.clear()
    _cache[k] = r
    return r

def paste(c, img, cx, by, h, alpha=1.0, rot=0, flip=False, shadow=True, anchor='bottom'):
    if h < 3 or alpha <= 0.01: return
    im = scaled(img, h, flip=flip)
    if rot: im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1:
        a = im.getchannel('A').point(lambda v: int(v*alpha)); im = im.copy(); im.putalpha(a)
    x = int(cx - im.width/2)
    y = int(by - im.height) if anchor == 'bottom' else int(by - im.height/2)
    if shadow and anchor == 'bottom':
        sw = int(im.width*0.8); sh = max(4, int(im.width*0.12))
        s = Image.new('RGBA', (sw, sh), (0,0,0,0))
        ImageDraw.Draw(s).ellipse([0,0,sw-1,sh-1], fill=(40,20,0,int(90*alpha)))
        s = s.filter(ImageFilter.GaussianBlur(sh/4))
        c.alpha_composite(s, (int(cx-sw/2), int(by-sh/2)), ) if 0 <= int(cx-sw/2) < W and 0<=int(by-sh/2)<H-sh else None
    c.alpha_composite(im, (0,0), (max(0,-x), max(0,-y))) if False else _comp(c, im, x, y)

def _comp(c, im, x, y):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x+im.width), min(H, y+im.height)
    if x1 <= x0 or y1 <= y0: return
    c.alpha_composite(im.crop((x0-x, y0-y, x1-x, y1-y)), (x0, y0))

def hat(color1, color2, size):
    s = size
    im = Image.new('RGBA', (s, int(s*1.4)), (0,0,0,0)); d = ImageDraw.Draw(im)
    top = (s//2, int(s*0.25)); bl = (int(s*0.12), int(s*1.35)); br = (int(s*0.88), int(s*1.35))
    d.polygon([top, bl, br], fill=color1)
    for i in range(1, 4):
        f = i/4.0
        y = int(top[1] + (bl[1]-top[1])*f); half = int((br[0]-bl[0])/2*f)
        d.line([(s//2-half, y), (s//2+half, y)], fill=color2, width=max(3, s//12))
    r = int(s*0.14); d.ellipse([s//2-r, top[1]-r, s//2+r, top[1]+r], fill=(255,255,255))
    return im
HATS = [hat((240,70,120),(255,220,80),120), hat((70,150,255),(255,255,255),120),
        hat((255,190,40),(240,70,120),120), hat((120,210,90),(255,240,120),120), hat((170,90,240),(255,200,60),120)]

# ---------- backgrounds ----------
def make_bg(w, h, seed=1, horizon=0.55):
    rnd = random.Random(seed)
    y = np.linspace(0, 1, h)[:, None]
    top = np.array([255, 140, 70]); mid = np.array([255, 200, 110]); hz = np.array([255, 236, 170])
    t = np.clip(y/horizon, 0, 1)
    sky = np.where(t < 0.6, top + (mid-top)*(t/0.6), mid + (hz-mid)*((t-0.6)/0.4))
    arr = np.repeat(sky, w, axis=1).reshape(h, w, 3) if sky.ndim == 2 else None
    arr = np.broadcast_to(sky[:, None, :] if sky.ndim == 2 else sky, (h, w, 3)).astype(np.float32).copy()
    # sun glow
    yy, xx = np.mgrid[0:h, 0:w]
    sx, sy = w*0.5, h*horizon*0.86
    dist = np.sqrt((xx-sx)**2 + (yy-sy)**2)
    glow = np.exp(-(dist/(w*0.35))**2)[..., None]
    arr = arr + (np.array([255, 250, 220]) - arr)*glow*0.85
    sun = (dist < w*0.11)[..., None]
    arr = np.where(sun, np.array([255, 252, 235]), arr)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert('RGBA')
    d = ImageDraw.Draw(img)
    hy = int(h*horizon)
    # distant hills
    for layer, col in [(0, (214, 140, 90)), (1, (190, 120, 70))]:
        pts = [(0, h)]
        for x in range(0, w+40, 40):
            pts.append((x, hy - 40 + layer*30 - 50*math.sin(x/(w*0.18)+layer*2+seed) - 25*math.sin(x/97.0+seed)))
        pts.append((w, h)); d.polygon(pts, fill=col)
    # acacia silhouettes
    for i in range(5):
        ax = rnd.uniform(0.05, 0.95)*w; ab = hy + rnd.uniform(-10, 40); sc = rnd.uniform(0.6, 1.2)
        col = (110, 60, 35)
        d.line([(ax, ab), (ax+10*sc, ab-150*sc)], fill=col, width=int(10*sc))
        d.line([(ax+6*sc, ab-90*sc), (ax-45*sc, ab-150*sc)], fill=col, width=int(7*sc))
        d.ellipse([ax-140*sc, ab-200*sc, ax+150*sc, ab-140*sc], fill=col)
        d.ellipse([ax-90*sc, ab-225*sc, ax+100*sc, ab-170*sc], fill=col)
    # savanna ground
    g = np.linspace(0, 1, h-hy)[:, None]
    gc = np.array([214, 170, 70]) + (np.array([120, 140, 40]) - np.array([214, 170, 70]))*g
    ground = np.broadcast_to(gc[:, None, :], (h-hy, w, 3)).astype(np.uint8)
    gi = Image.fromarray(np.ascontiguousarray(ground)).convert('RGBA')
    mask = Image.new('L', (w, h-hy), 0); md = ImageDraw.Draw(mask)
    pts = [(0, h-hy)] + [(x, 30 + 18*math.sin(x/150.0+seed)) for x in range(0, w+30, 30)] + [(w, h-hy)]
    md.polygon(pts, fill=255)
    img.paste(gi, (0, hy), mask)
    d = ImageDraw.Draw(img)
    # grass tufts
    for i in range(int(w*h/2500)):
        gx = rnd.uniform(0, w); gy = rnd.uniform(hy+40, h)
        k = (gy-hy)/(h-hy); ln = 8 + 40*k
        shade = rnd.randint(-25, 25)
        col = (int(150+shade-60*k), int(150+shade-10*k), int(50+shade*0.5), 255)
        d.line([(gx, gy), (gx+rnd.uniform(-6, 6)*k*2, gy-ln)], fill=col, width=max(1, int(1+3*k)))
    return img

def make_jungle(w, h, seed=7):
    # jungle clearing: green tones with light rays
    rnd = random.Random(seed)
    y = np.linspace(0, 1, h)[:, None]
    top = np.array([255, 205, 120]); bot = np.array([90, 140, 50])
    col = top + (bot-top)*np.clip((y-0.2)/0.6, 0, 1)
    arr = np.broadcast_to(col[:, None, :], (h, w, 3)).astype(np.float32).copy()
    img = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    d = ImageDraw.Draw(img)
    # tree trunks
    for i in range(7):
        tx = rnd.uniform(0, w); tw = rnd.uniform(30, 90)
        d.rectangle([tx, 0, tx+tw, h*0.72], fill=(120, 80, 45))
    # canopy blobs
    for i in range(70):
        cx = rnd.uniform(-50, w+50); cy = rnd.uniform(-60, h*0.38); r = rnd.uniform(60, 170)
        g = rnd.randint(110, 170)
        d.ellipse([cx-r, cy-r*0.8, cx+r, cy+r*0.8], fill=(40, g, 50))
    # ground
    d.ellipse([-w*0.4, h*0.62, w*1.4, h*1.5], fill=(130, 170, 60))
    for i in range(1800):
        gx = rnd.uniform(0, w); gy = rnd.uniform(h*0.66, h)
        k = (gy-h*0.66)/(h*0.34); ln = 10+40*k
        d.line([(gx, gy), (gx+rnd.uniform(-8, 8), gy-ln)], fill=(70+rnd.randint(0,40), 150+rnd.randint(-20,30), 40, 255), width=max(1, int(1+3*k)))
    # light rays
    rays = Image.new('RGBA', (w, h), (0,0,0,0)); rd = ImageDraw.Draw(rays)
    for i in range(6):
        x = rnd.uniform(0.1, 0.9)*w
        rd.polygon([(x, 0), (x+80, 0), (x+380, h*0.8), (x+150, h*0.8)], fill=(255, 245, 190, 45))
    rays = rays.filter(ImageFilter.GaussianBlur(25))
    img.alpha_composite(rays)
    return img.filter(ImageFilter.GaussianBlur(1.2))

def foliage_frame(w, h, seed=3):
    rnd = random.Random(seed)
    im = Image.new('RGBA', (w, h), (0,0,0,0)); d = ImageDraw.Draw(im)
    for side in [0, 1]:
        for i in range(14):
            bx = (0 if side == 0 else w) + rnd.uniform(-40, 40)
            by = rnd.uniform(-50, h*0.35) if i < 7 else rnd.uniform(h*0.78, h+60)
            ang = rnd.uniform(0, math.pi*2); ln = rnd.uniform(160, 320); wd = ln*0.32
            g = rnd.randint(90, 160)
            ex, ey = bx+math.cos(ang)*ln, by+math.sin(ang)*ln
            nx, ny = -math.sin(ang)*wd, math.cos(ang)*wd
            mx, my = (bx+ex)/2, (by+ey)/2
            d.polygon([(bx, by), (mx+nx, my+ny), (ex, ey), (mx-nx, my-ny)], fill=(30, g, 45, 255))
            d.line([(bx, by), (ex, ey)], fill=(20, g-40, 30, 255), width=4)
    return im

print('building backgrounds', flush=True)
BG1 = make_bg(1300, 2300, seed=1)
BG2 = make_bg(2600, 1920, seed=4, horizon=0.5)   # wide for running pan
BGJ = make_jungle(1300, 2300)
BG5 = make_bg(1300, 2300, seed=9, horizon=0.5)
FOL = foliage_frame(W, H)

def bg_view(bg, t, z0=1.0, z1=1.12, ox=0.5, oy=0.5):
    z = z0 + (z1-z0)*t
    cw, ch = W/z*(bg.width/W if bg.width < W*1.3 else 1), H/z
    cw = min(bg.width, bg.width/z) if bg.width < 2000 else W/z
    ch = cw*H/W
    x = (bg.width-cw)*ox; y = (bg.height-ch)*oy
    return bg.crop((int(x), int(y), int(x+cw), int(y+ch))).resize((W, H), Image.BILINEAR)

# ---------- text ----------
GOLD_TOP = np.array([255, 245, 170]); GOLD_BOT = np.array([220, 140, 20])
_tcache = {}
def gold_text(txt, sz, stroke=10, wgt='ExtraBold', fnt='Baloo2.ttf', maxw=980, color=None):
    k = (txt, sz, stroke, wgt, fnt, color)
    if k in _tcache: return _tcache[k]
    f = font(sz, wgt, fnt)
    lines = txt.split('\n')
    while True:
        boxes = [f.getbbox(l, stroke_width=stroke) for l in lines]
        tw = max(b[2]-b[0] for b in boxes)
        if tw <= maxw or sz < 30: break
        sz = int(sz*0.93); f = font(sz, wgt, fnt)
    lh = int(sz*1.08)
    th = lh*len(lines) + stroke*2 + 20
    pad = 30
    mask = Image.new('L', (tw+pad*2, th+pad*2), 0); md = ImageDraw.Draw(mask)
    smask = Image.new('L', mask.size, 0); sd = ImageDraw.Draw(smask)
    for i, l in enumerate(lines):
        b = f.getbbox(l, stroke_width=stroke); lw = b[2]-b[0]
        x = pad + (tw-lw)//2 - b[0]; y = pad + i*lh
        sd.text((x, y), l, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        md.text((x, y), l, font=f, fill=255)
    w2, h2 = mask.size
    if color is None:
        yy = np.linspace(0, 1, h2)[:, None]
        grad = GOLD_TOP + (GOLD_BOT-GOLD_TOP)*yy
        fill = Image.fromarray(np.broadcast_to(grad[:, None, :], (h2, w2, 3)).astype(np.uint8)).convert('RGBA')
    else:
        fill = Image.new('RGBA', (w2, h2), color)
    out = Image.new('RGBA', (w2, h2+12), (0,0,0,0))
    sh = Image.new('RGBA', (w2, h2), (40, 15, 0, 0)); sh.putalpha(smask.filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v*0.7)))
    out.alpha_composite(sh, (0, 12))
    st = Image.new('RGBA', (w2, h2), (95, 40, 10, 255)); st.putalpha(smask)
    out.alpha_composite(st)
    fill.putalpha(mask); out.alpha_composite(fill)
    _tcache[k] = out
    return out

def put_text(c, img, cx, cy, s=1.0, alpha=1.0, rot=0):
    if s <= 0.02 or alpha <= 0.01: return
    im = img
    if s != 1.0: im = im.resize((max(2, int(im.width*s)), max(2, int(im.height*s))), Image.BICUBIC)
    if rot: im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1:
        im = im.copy(); im.putalpha(im.getchannel('A').point(lambda v: int(v*alpha)))
    _comp(c, im, int(cx-im.width/2), int(cy-im.height/2))

# ---------- particles ----------
PR = random.Random(42)
CONF = [(PR.uniform(0, W), PR.uniform(-H, H), PR.uniform(80, 220), PR.uniform(0, 6.28), PR.choice([(255,80,120),(80,170,255),(255,210,60),(120,220,100),(190,110,255),(255,140,40)])) for _ in range(110)]
BFLY = [(PR.uniform(0, W), PR.uniform(200, 1300), PR.uniform(0, 6.28), PR.uniform(50, 80)) for _ in range(5)]
FIRE = [(PR.uniform(0, W), PR.uniform(0, H), PR.uniform(0, 6.28)) for _ in range(40)]

def confetti(c, t, amt=1.0):
    d = ImageDraw.Draw(c)
    for i, (x, y, sp, ph, col) in enumerate(CONF):
        if i > len(CONF)*amt: break
        yy = (y + sp*t*2) % (H+200) - 100
        xx = x + 30*math.sin(t*2+ph)
        a = t*4+ph; s = 10
        dx, dy = math.cos(a)*s, abs(math.sin(a*1.3))*s*0.6+2
        d.polygon([(xx-dx, yy-dy), (xx+dx, yy-dy), (xx+dx, yy+dy), (xx-dx, yy+dy)], fill=col+(255,))

def butterflies(c, t, n=5):
    for i, (x, y, ph, sz) in enumerate(BFLY[:n]):
        xx = (x + 60*t + 120*math.sin(t*0.7+ph)) % (W+200) - 100
        yy = y + 80*math.sin(t*1.1+ph*2)
        flap = 0.55 + 0.45*abs(math.sin(t*12+ph))
        im = scaled(EM['butterfly'], sz)
        im = im.resize((max(2, int(im.width*flap)), im.height))
        _comp(c, im, int(xx-im.width/2), int(yy-im.height/2))

def sparkle_dust(c, t, amt=1.0):
    lay = Image.new('RGBA', (W, H), (0,0,0,0)); d = ImageDraw.Draw(lay)
    for i, (x, y, ph) in enumerate(FIRE):
        if i > len(FIRE)*amt: break
        a = 0.5+0.5*math.sin(t*3+ph)
        yy = (y - 40*t) % H
        r = 3+4*a
        d.ellipse([x-r, yy-r, x+r, yy+r], fill=(255, 245, 190, int(200*a)))
    c.alpha_composite(lay.filter(ImageFilter.GaussianBlur(2)))

def vignette():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx-W/2)/(W*0.75))**2 + ((yy-H/2)/(H*0.7))**2)
    a = np.clip((d-0.6)*1.4, 0, 1)*150
    v = np.zeros((H, W, 4), np.uint8); v[..., 0] = 60; v[..., 1] = 25; v[..., 3] = a.astype(np.uint8)
    return Image.fromarray(v)
VIG = vignette()

def wooden_sign(w, h):
    im = Image.new('RGBA', (w, h+380), (0,0,0,0)); d = ImageDraw.Draw(im)
    # posts
    d.rectangle([w*0.18, h-20, w*0.18+40, h+380], fill=(110, 65, 30))
    d.rectangle([w*0.82-40, h-20, w*0.82, h+380], fill=(110, 65, 30))
    rnd = random.Random(5)
    planks = 5; ph = h/planks
    for i in range(planks):
        base = (150+rnd.randint(-12, 12), 95+rnd.randint(-8, 8), 50)
        d.rounded_rectangle([0, i*ph+2, w, (i+1)*ph-2], radius=18, fill=base)
        for j in range(10):
            yy = i*ph + rnd.uniform(10, ph-10)
            d.line([(rnd.uniform(0, w*0.3), yy), (rnd.uniform(w*0.6, w), yy+rnd.uniform(-4, 4))], fill=(base[0]-25, base[1]-20, base[2]-15), width=2)
    d.rounded_rectangle([0, 0, w-1, h-1], radius=22, outline=(90, 50, 20), width=8)
    # flower garland
    for k in range(12):
        x = 20 + k*(w-40)/11
        y = -10 + 28*math.sin(k/11*math.pi)
        fl = scaled(EM['hibiscus'], 80 if k % 2 == 0 else 60)
        im.alpha_composite(fl, (int(x-fl.width/2), int(max(0, y))))
    return im
SIGN = wooden_sign(920, 760)

# ---------- scenes ----------
def miraya(c, idx, cx, by, h, t, bob=6, sway=1.2, alpha=1.0):
    paste(c, MIR[idx], cx, by + bob*math.sin(t*2.2), h*(1+0.012*math.sin(t*1.7)), alpha=alpha, rot=sway*math.sin(t*1.3))

def frame(i):
    t = i/FPS
    c = Image.new('RGBA', (W, H))

    if t < 6.0:  # S1 opening savanna
        lt = t/6
        c.paste(bg_view(BG1, lt, 1.0, 1.1, 0.5, 0.45))
        # animals gather from sides
        hz = H*0.66
        for k, (name, sx, ex, by, hh, fl, delay) in enumerate([
            ('giraffe', -300, 170, hz+40, 430, True, 0.0), ('elephant', W+300, 900, hz+60, 300, False, 0.3),
            ('zebra', -300, 330, hz+110, 230, True, 0.8), ('rhinoceros', W+300, 680, hz+100, 170, False, 1.2),
            ('monkey', W+250, 860, hz+260, 170, False, 1.6)]):
            p = ease(seg(t, delay, delay+3.2))
            x = sx + (ex-sx)*p
            hop = abs(math.sin(t*7+k))*14*(1-p) if p < 1 else 3*abs(math.sin(t*3+k))
            paste(c, EM[name], x, by-hop, hh, flip=fl)
        # rock + Miraya + lion cub
        rp = ease(seg(t, 0.6, 2.2))
        rock_y = H*0.9 + (1-rp)*400
        d = ImageDraw.Draw(c)
        d.ellipse([170, rock_y-140, 920, rock_y+90], fill=(150, 120, 95)); d.ellipse([210, rock_y-150, 870, rock_y-10], fill=(175, 145, 115))
        miraya(c, 0, 470, rock_y-50, 720*rp if rp > 0.05 else 0, t, sway=0.8)
        paste(c, EM['lion'], 780, rock_y-70 - 8*abs(math.sin(t*3)), 230*rp, rot=4*math.sin(t*2))
        butterflies(c, t, 3); sparkle_dust(c, t, 0.6)
        pass
        a = 1-seg(t, 5.4, 6.0)
    elif t < 11.5:  # S2 running title
        lt = (t-6)/5.5
        x = (BG2.width-W)*(lt)
        view = BG2.crop((int(x), 0, int(x)+W, H))
        c.paste(view)
        # speed lines in grass
        d = ImageDraw.Draw(c)
        rnd = random.Random(int(t*FPS))
        for _ in range(26):
            yy = rnd.uniform(H*0.62, H*0.98); xx = rnd.uniform(0, W); ln = rnd.uniform(80, 260)
            d.line([(xx, yy), (xx+ln, yy)], fill=(255, 240, 190, 140), width=3)
        gallop = abs(math.sin(t*9))
        base = H*0.92
        # lion (big) carrying Miraya: Miraya sits behind lion head on a drawn mane-coloured back
        bx = 560 + 20*math.sin(t*1.5)
        by = base - 40*gallop
        d.ellipse([bx-330, by-300, bx+150, by-60], fill=(222, 150, 60))         # body
        for lx in [-260, -170, 0, 80]:
            off = 30*math.sin(t*9 + lx)
            d.rounded_rectangle([bx+lx+off*0.3, by-130, bx+lx+60+off*0.3, by+10], radius=25, fill=(205, 135, 50))
        d.line([(bx-320, by-230), (bx-420, by-330+30*math.sin(t*6))], fill=(205, 135, 50), width=22)
        d.ellipse([bx-450, by-360+30*math.sin(t*6), bx-395, by-300+30*math.sin(t*6)], fill=(150, 80, 30))
        miraya(c, 1, bx-140, by-210, 560, t, bob=0, sway=2.0)
        paste(c, EM['lion'], bx+150, by-150, 360, rot=-5+4*math.sin(t*9), shadow=False)
        # title
        tp = back(seg(t, 6.4, 7.4))
        put_text(c, gold_text("Welcome to\nMiraya's\nJungle Party", 150), W/2, 420 + 10*math.sin(t*2), s=tp, alpha=min(1, tp*1.5))
        sparkle_dust(c, t, 0.8)
        a = min(seg(t, 6.0, 6.5), 1-seg(t, 11.0, 11.5))
    elif t < 16.5:  # S3 Miraya turns 3
        lt = (t-11.5)/5
        c.paste(bg_view(BGJ, lt, 1.05, 1.15, 0.5, 0.6))
        base = H*0.9
        miraya(c, 2, W/2, base, 980, t, sway=1.0)
        # cubs playing around
        for k, (name, ox, oy, hh) in enumerate([('lion', 160, base-30, 210), ('tiger', 900, base+10, 230), ('monkey_face', 150, base-640, 150), ('monkey', 930, base-560, 190)]):
            hop = abs(math.sin(t*5+k*1.3))*50
            p = back(seg(t, 11.8+k*0.25, 12.6+k*0.25))
            paste(c, EM[name], ox, oy-hop, hh*p, rot=8*math.sin(t*5+k), flip=(k % 2 == 1))
        paste(c, EM['paw_prints'], 300, base+40, 90, shadow=False)
        tp = back(seg(t, 12.0, 12.9))
        put_text(c, gold_text("Miraya", 170, wgt='ExtraBold'), W/2, 300, s=tp)
        tp2 = back(seg(t, 12.5, 13.4))
        big = gold_text("Turns 3!", 210)
        put_text(c, big, W/2, 500, s=tp2*(1+0.04*math.sin(t*6)), rot=3*math.sin(t*2))
        butterflies(c, t, 5); confetti(c, t-11.5, 0.4)
        a = min(seg(t, 11.5, 12.0), 1-seg(t, 16.0, 16.5))
    elif t < 24.0:  # S4 celebration + sign (long readable hold)
        lt = (t-16.5)/7.5
        c.paste(bg_view(BGJ, lt, 1.12, 1.18, 0.4, 0.55))
        sp = back(seg(t, 16.7, 17.6))
        sy = 120 + (1-sp)*-900
        _comp(c, SIGN, int(W/2-SIGN.width/2), int(sy))
        tx = gold_text("We invite you to\ncome and celebrate\nMiraya's\n3rd Birthday\nwith us", 112, stroke=8, maxw=840,
                       color=None)
        put_text(c, tx, W/2, sy+390, s=min(1, sp))
        # animals with hats around cake
        base = H*0.94
        cake_x = W/2
        party = [('elephant', 200, base-30, 300, False), ('giraffe', 920, base-40, 460, False), ('zebra', 850, base+40, 250, True),
                 ('lion', 300, base+60, 260, False), ('monkey', cake_x+170, base+40, 230, False), ('hippopotamus', 120, base+90, 190, True)]
        for k, (name, x, by, hh, fl) in enumerate(party):
            p = back(seg(t, 17.2+k*0.15, 18.0+k*0.15))
            bob = abs(math.sin(t*4+k))*12
            paste(c, EM[name], x, by-bob, hh*p, flip=fl)
            if p > 0.1:
                hm = scaled(HATS[k % 5], int(hh*0.45*p))
                hy = by-bob-hh*p*(0.98 if name not in ('giraffe',) else 1.0)
                hx = x + (hh*p*0.1 if name in ('elephant',) else 0) + (hh*p*0.28 if name == 'giraffe' else 0)
                _comp(c, hm.rotate(10*math.sin(t*3+k), Image.BICUBIC, expand=True), int(hx-hm.width/2), int(hy-hm.height*0.7))
        ck = back(seg(t, 17.6, 18.4))
        paste(c, EM['birthday_cake'], cake_x-40, base+20, 300*ck)
        # candle flames until monkey blows at 21.8s
        d = ImageDraw.Draw(c)
        if ck > 0.9:
            if t < 21.8:
                for fx in [-110, -40, 30]:
                    fl = 1 + 0.2*math.sin(t*25+fx)
                    d.ellipse([cake_x-40+fx*1.0-9, base-310-24*fl, cake_x-40+fx+9, base-302], fill=(255, 200, 60))
                    d.ellipse([cake_x-40+fx-4, base-306-14*fl, cake_x-40+fx+4, base-303], fill=(255, 250, 220))
            else:
                for fx in [-110, -40, 30]:
                    for s_ in range(3):
                        sy2 = base-320 - (t-21.8)*120 - s_*30
                        d.ellipse([cake_x-40+fx-8+10*math.sin(t*3+s_), sy2-8, cake_x-40+fx+8+10*math.sin(t*3+s_), sy2+8], fill=(235, 235, 235, 160))
        if 21.2 < t < 22.2:  # puff lines from monkey
            for k in range(4):
                d.line([(cake_x+90, base-260+k*16), (cake_x+40-40*seg(t, 21.2, 21.8), base-280+k*14)], fill=(255, 255, 255), width=5)
        if t > 21.8:
            confetti(c, t-21.8, 1.0)
            put_text(c, gold_text("Yaaay!", 90), cake_x+250, base-470, s=back(seg(t, 21.9, 22.5)))
        a = min(seg(t, 16.5, 17.0), 1-seg(t, 23.5, 24.0))
    elif t < 29.5:  # S5 details
        lt = (t-24)/5.5
        c.paste(bg_view(BG5, lt, 1.0, 1.12, 0.5, 0.4))
        dark = Image.new('RGBA', (W, H), (60, 25, 0, 70)); c.alpha_composite(dark)
        miraya(c, 3, W/2, H*0.97, 900, t, sway=0.8)
        for k, (x, y0) in enumerate([(130, 1400), (950, 1300), (220, 900), (880, 820)]):
            yy = y0 - 50*(t-24) + 20*math.sin(t*2+k)
            paste(c, EM['balloon'], x, yy, 220, shadow=False, anchor='center', rot=6*math.sin(t+k))
        tp = back(seg(t, 24.4, 25.2)); tp2 = back(seg(t, 24.9, 25.7)); tp3 = back(seg(t, 25.4, 26.2))
        put_text(c, gold_text("Save the Date", 100, wgt='Bold', color=(255, 255, 255, 255)), W/2, 210, s=tp)
        put_text(c, gold_text("26th November", 175), W/2, 400, s=tp2*(1+0.02*math.sin(t*4)))
        put_text(c, gold_text("5:30 PM", 210), W/2, 620, s=tp3*(1+0.02*math.sin(t*4+1)))
        sparkle_dust(c, t, 1.0)
        a = min(seg(t, 24.0, 24.5), 1-seg(t, 29.0, 29.5))
    else:  # S6 closing
        lt = (t-29.5)/6.5
        shake = 0
        if 32.0 < t < 33.3: shake = 14*math.sin(t*70)*(1-seg(t, 32.0, 33.3))
        c.paste(bg_view(BG1, lt, 1.08, 1.0, 0.5, 0.5))
        base = H*0.93
        for k, (name, x, by, hh, fl) in enumerate([('giraffe', 140, base-260, 420, True), ('elephant', 930, base-230, 290, False),
                                                    ('zebra', 120, base+20, 240, True), ('monkey', 960, base+10, 200, False), ('tiger', 860, base+60, 190, False)]):
            bob = abs(math.sin(t*4+k))*16
            paste(c, EM[name], x, by-bob, hh, flip=fl)
            hm = scaled(HATS[k % 5], int(hh*0.42))
            _comp(c, hm, int(x-hm.width/2 + (hh*0.28 if name == 'giraffe' and not fl else (-hh*0.28 if name == 'giraffe' else 0))), int(by-bob-hh-hm.height*0.6))
        # Miraya riding lion cub (bouncy)
        hop = abs(math.sin(t*5))*30
        miraya(c, 0, 470, base-90-hop, 760, t, bob=0, sway=2.5)
        paste(c, EM['lion'], 690, base-40-hop, 330, rot=6*math.sin(t*5))
        # roaring lion zoom
        rp = back(seg(t, 31.6, 32.4))
        if t > 31.6:
            put_text(c, scaled(EM['lion'], 600), W/2+shake, 820, s=rp*(1+0.06*math.sin(t*20)*(t < 33.3)))
            put_text(c, gold_text("ROAR!", 120, color=(255, 255, 255, 255)), W/2+shake, 470, s=back(seg(t, 32.0, 32.5))*(1 if t < 33.6 else max(0, 1-(t-33.6)*3)))
        tp = back(seg(t, 33.2, 34.0))
        put_text(c, gold_text("See You At\nThe Party!", 160), W/2, 330+8*math.sin(t*2), s=tp)
        confetti(c, t-29.5, 1.0); butterflies(c, t, 4); sparkle_dust(c, t, 0.8)
        a = min(seg(t, 29.5, 30.0), 1-seg(t, 35.3, 36.0))
    c.alpha_composite(VIG)
    if a < 1:
        blk = Image.new('RGBA', (W, H), (0, 0, 0, int(255*(1-a)))); c.alpha_composite(blk)
    return c.convert('RGB').tobytes()

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'out.mp4')
    only = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
    if only:
        for s in only:
            Image.frombytes('RGB', (W, H), frame(int(s*FPS))).save(os.path.join(D, f'prev_{s}.jpg'), quality=85)
        sys.exit()
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    with Pool(4) as p:
        for k, b in enumerate(p.imap(frame, range(N), chunksize=4)):
            ff.stdin.write(b)
            if k % 150 == 0: print(k, '/', N, flush=True)
    ff.stdin.close(); ff.wait(); print('done', out)
