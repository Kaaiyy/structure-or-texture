"""Step 11: degradation experiment. Add more and more noise to a real retina,
re-run the frozen pipeline, and track two things at each level:
  - clDice against the hand-drawn map (is the segmentation actually getting worse?)
  - the IAAFT test (does it flip to 'cannot tell from texture' as it gets worse?)
Saves after every row (Ctrl+C safe, rerun resumes).

usage: python step11_degrade.py data/hrf [stems comma-separated] [n_surr=19]"""
import sys, csv, os, time
import numpy as np
import hrf
from vessels import grow, pipeline_v2, pieces, longest_piece, iaaft, cl_dice

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
STEMS = (sys.argv[2] if len(sys.argv) > 2 else "09_g,14_h,11_dr").split(",")
N_SURR = int(sys.argv[3]) if len(sys.argv) > 3 else 19
LEVELS = [0, 0.5, 1, 2, 4]        # noise sd as a multiple of the image's own spread
OUT = "results/hrf_degrade.csv"
PIPE = dict(local=True, hyst=True)

fields = ["stem", "noise", "cldice", "longest", "null_mean", "null_sd", "p_longest", "z_longest", "ratio"]
done = set()
if os.path.exists(OUT):
    done = {(r["stem"], r["noise"]) for r in csv.DictReader(open(OUT))}
new = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
w = csv.DictWriter(f, fieldnames=fields)
if new:
    w.writeheader()

for stem in STEMS:
    d = hrf.load(ROOT, stem)
    green, fov = d["im"][:, :, 1], d["fov"]
    sd0 = green[fov].std()
    for lev in LEVELS:
        if (stem, str(lev)) in done:
            continue
        t0 = time.time()
        noisy = green + np.random.default_rng(123).normal(0, lev * sd0, green.shape)
        g = grow(1 - noisy, fov)
        m = pipeline_v2(g, d["eye"], **PIPE)
        real = longest_piece(m)
        null = np.array([longest_piece(pipeline_v2(iaaft(g, seed=k), d["eye"], **PIPE))
                         for k in range(N_SURR)], float)
        p = (1 + (null >= real).sum()) / (N_SURR + 1)
        sd = null.std() + 1e-12
        row = dict(stem=stem, noise=lev, cldice=cl_dice(m, d["truth"]), longest=real,
                   null_mean=null.mean(), null_sd=sd, p_longest=p,
                   z_longest=(real - null.mean()) / sd, ratio=real / max(null.mean(), 1))
        w.writerow(row); f.flush()
        print(f"{stem:6s} noise {lev:3} | clDice {row['cldice']:.2f} | longest {real:.0f} vs null "
              f"{null.mean():.0f}+-{sd:.0f}  p={p:.3f} z={row['z_longest']:.1f} "
              f"ratio={row['ratio']:.1f} | {time.time()-t0:.0f}s", flush=True)
print("saved", OUT)
