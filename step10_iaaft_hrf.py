"""Step 10: the real test. For a few held-out HRF images (spread from worst to
best clDice) run the IAAFT surrogate test with the FROZEN pipeline_v2.
Question: does the surrogate p-value / z-score separate good from bad
segmentations, better than the plain 'longest piece' number?
Saves after every image (safe to Ctrl+C and rerun: it resumes).

usage: python step10_iaaft_hrf.py data/hrf [n_images=6] [n_surr=19]"""
import sys, csv, os, time
import numpy as np
import hrf
from vessels import grow, pipeline_v2, pieces, longest_piece, iaaft

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
N_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 6
N_SURR = int(sys.argv[3]) if len(sys.argv) > 3 else 19
OUT = "results/hrf_iaaft.csv"
PIPE = dict(local=True, hyst=True)

held = list(csv.DictReader(open("results/hrf_heldout.csv")))
held.sort(key=lambda r: float(r["cldice"]))
pick = [held[i] for i in np.linspace(0, len(held) - 1, N_IMG).round().astype(int)]

done = set()
if os.path.exists(OUT):
    done = {r["stem"] for r in csv.DictReader(open(OUT))}
fields = ["stem", "group", "cldice", "longest", "null_mean", "null_sd", "p_longest", "z_longest",
          "share", "p_share", "n_surr"]
new = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
w = csv.DictWriter(f, fieldnames=fields)
if new:
    w.writeheader()

def stats(g, eye):
    m = pipeline_v2(g, eye, **PIPE)
    return longest_piece(m), pieces(m)[1]

for r in pick:
    if r["stem"] in done:
        continue
    t0 = time.time()
    d = hrf.load(ROOT, r["stem"])
    g = grow(1 - d["im"][:, :, 1], d["fov"])
    real = np.array(stats(g, d["eye"]), float)
    null = np.array([stats(iaaft(g, seed=k), d["eye"]) for k in range(N_SURR)], float)
    p = (1 + (null >= real).sum(0)) / (N_SURR + 1)
    sd = null[:, 0].std() + 1e-12
    row = dict(stem=r["stem"], group=r["group"], cldice=float(r["cldice"]),
               longest=real[0], null_mean=null[:, 0].mean(), null_sd=sd, p_longest=p[0],
               z_longest=(real[0] - null[:, 0].mean()) / sd, share=real[1], p_share=p[1], n_surr=N_SURR)
    w.writerow(row); f.flush()
    print(f"{r['stem']:7s} clDice {row['cldice']:.2f} | longest {real[0]:.0f} vs null "
          f"{row['null_mean']:.0f}+-{sd:.0f}  p={p[0]:.3f} z={row['z_longest']:.1f} | "
          f"share p={p[1]:.2f} | {time.time()-t0:.0f}s", flush=True)
print("saved", OUT)
