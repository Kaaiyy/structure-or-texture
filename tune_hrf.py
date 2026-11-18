"""Compare pipeline variants on a few TUNING images (these are then held out
from the final scoring, so we don't grade ourselves on the images we tuned on).
usage: python tune_hrf.py data/hrf [stem stem ...]"""
import sys
import numpy as np
import hrf
from vessels import grow, pipeline_v2, pieces, dice, cl_dice

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
STEMS = sys.argv[2:] or ["01_h", "01_g", "01_dr"]
VARIANTS = {
    "v1 (current)":   dict(local=False, hyst=False),
    "+ local contrast": dict(local=True, hyst=False),
    "+ hysteresis":   dict(local=False, hyst=True),
    "both":           dict(local=True, hyst=True),
}

data = [hrf.load(ROOT, s) for s in STEMS]
print(f"tuning images: {STEMS}\n")
print(f"{'variant':18s} {'prec':>6s} {'recall':>7s} {'dice':>6s} {'clDice':>7s} {'share':>6s} {'pieces':>7s}")
for name, kw in VARIANTS.items():
    r = []
    for d in data:
        base = grow(1 - d["im"][:, :, 1], d["fov"])
        m = pipeline_v2(base, d["eye"], **kw)
        t = d["truth"]
        tp = (m & t).sum()
        n, big = pieces(m)
        r.append([tp / max(m.sum(), 1), tp / max(t.sum(), 1), dice(m, t), cl_dice(m, t), big, n])
    a = np.mean(r, 0)
    print(f"{name:18s} {a[0]:6.2f} {a[1]:7.2f} {a[2]:6.2f} {a[3]:7.2f} {a[4]:6.2f} {a[5]:7.0f}")
