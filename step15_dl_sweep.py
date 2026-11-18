"""Step 15: head-to-head robustness. Noise and blur on ALL held-out HRF images,
scored against the expert map (clDice) for
  - our classical pipeline (frozen pipeline_v2)
  - the pretrained deep-learning model (Little W-Net, trained on DRIVE, never saw HRF)
No surrogates here (cheap): ~20 min. Saves after every image; rerun resumes.

usage: python step15_dl_sweep.py data/hrf"""
import sys, csv, os, time
import numpy as np
from scipy import ndimage as ndi
import hrf
from vessels import grow, pipeline_v2, dice, cl_dice
from dl_model import dl_mask

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
JOBS = [("noise", 0), ("noise", 0.3), ("noise", 0.5), ("noise", 1), ("noise", 2),
        ("blur", 1), ("blur", 2), ("blur", 4), ("blur", 8)]
OUT = "results/hrf_dl_sweep.csv"
fields = ["stem", "group", "kind", "level", "cl_classic", "cl_dl", "dice_classic", "dice_dl", "dl_vessel_pct"]

stems = [r["stem"] for r in csv.DictReader(open("results/hrf_heldout.csv"))]
done = set()
if os.path.exists(OUT):
    done = {r["stem"] for r in csv.DictReader(open(OUT))}
new = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
w = csv.DictWriter(f, fieldnames=fields)
if new:
    w.writeheader()

for k, stem in enumerate(stems, 1):
    if stem in done:
        continue
    t0 = time.time()
    d = hrf.load(ROOT, stem)
    im, fov, eye, truth = d["im"], d["fov"], d["eye"], d["truth"]
    sds = [im[:, :, c][fov].std() for c in range(3)]
    for kind, lev in JOBS:
        if kind == "noise":
            rs = np.random.default_rng(123)
            x = np.stack([im[:, :, c] + rs.normal(0, lev * sds[c], fov.shape) for c in range(3)], axis=2)
        else:
            x = np.stack([ndi.gaussian_filter(im[:, :, c], lev) for c in range(3)], axis=2)
        mc = pipeline_v2(grow(1 - x[:, :, 1], fov), eye, local=True, hyst=True)
        md = dl_mask(x, fov) & eye
        w.writerow(dict(stem=stem, group=d["group"], kind=kind, level=lev,
                        cl_classic=cl_dice(mc, truth), cl_dl=cl_dice(md, truth),
                        dice_classic=dice(mc, truth), dice_dl=dice(md, truth),
                        dl_vessel_pct=100 * md.sum() / max(eye.sum(), 1)))
    f.flush()
    print(f"{k:2d}/{len(stems)} {stem:6s} done in {time.time()-t0:.0f}s", flush=True)
print("saved", OUT)
