"""Step 12: finer noise steps (where exactly does the test flip?) plus a BLUR
control (blur does not whiten the spectrum, so it is the fair test).
Same 3 images, same frozen pipeline, same IAAFT test as step 11.
Saves after every row (Ctrl+C safe, rerun resumes).

usage: python step12_sweep.py data/hrf [stems comma-separated] [n_surr=19]"""
import sys, csv, os, time
import numpy as np
from scipy import ndimage as ndi
import hrf
from vessels import grow, pipeline_v2, longest_piece, iaaft, cl_dice

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
STEMS = (sys.argv[2] if len(sys.argv) > 2 else "09_g,14_h,11_dr").split(",")
N_SURR = int(sys.argv[3]) if len(sys.argv) > 3 else 19
JOBS = [("noise", 0.1), ("noise", 0.2), ("noise", 0.3),
        ("blur", 1), ("blur", 2), ("blur", 4), ("blur", 8)]
OUT = "results/hrf_sweep.csv"
PIPE = dict(local=True, hyst=True)

fields = ["stem", "kind", "level", "cldice", "longest", "null_mean", "null_sd",
          "p_longest", "z_longest", "ratio"]
done = set()
if os.path.exists(OUT):
    done = {(r["stem"], r["kind"], r["level"]) for r in csv.DictReader(open(OUT))}
new = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
w = csv.DictWriter(f, fieldnames=fields)
if new:
    w.writeheader()

for stem in STEMS:
    d = hrf.load(ROOT, stem)
    green, fov = d["im"][:, :, 1], d["fov"]
    sd0 = green[fov].std()
    for kind, lev in JOBS:
        if (stem, kind, str(lev)) in done:
            continue
        t0 = time.time()
        if kind == "noise":
            x = green + np.random.default_rng(123).normal(0, lev * sd0, green.shape)
        else:
            x = ndi.gaussian_filter(green, lev)
        g = grow(1 - x, fov)
        m = pipeline_v2(g, d["eye"], **PIPE)
        real = longest_piece(m)
        null = np.array([longest_piece(pipeline_v2(iaaft(g, seed=k), d["eye"], **PIPE))
                         for k in range(N_SURR)], float)
        p = (1 + (null >= real).sum()) / (N_SURR + 1)
        sd = null.std() + 1e-12
        row = dict(stem=stem, kind=kind, level=lev, cldice=cl_dice(m, d["truth"]), longest=real,
                   null_mean=null.mean(), null_sd=sd, p_longest=p,
                   z_longest=(real - null.mean()) / sd, ratio=real / max(null.mean(), 1))
        w.writerow(row); f.flush()
        print(f"{stem:6s} {kind:5s} {lev:4} | clDice {row['cldice']:.2f} | longest {real:.0f} vs null "
              f"{null.mean():.0f}+-{sd:.0f}  p={p:.3f} ratio={row['ratio']:.1f} | {time.time()-t0:.0f}s",
              flush=True)
print("saved", OUT)
