"""Step 07: run our pipeline on every HRF image and score it against the
hand-drawn vessel map. No surrogates yet (those are slow) - this is the cheap
first look: does 'longest piece' track real accuracy?

usage: python step07_hrf_baseline.py data/hrf [N]     (N = only first N images)"""
import sys, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from skimage.morphology import skeletonize
import hrf
from vessels import grow, run_pipeline, longest_piece

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
N = int(sys.argv[2]) if len(sys.argv) > 2 else None


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-12)


def cl_dice(pred, truth):
    """Centreline Dice (Shit et al. 2021): topology-aware overlap."""
    sp, st = skeletonize(pred), skeletonize(truth)
    tprec = (sp & truth).sum() / (sp.sum() + 1e-12)
    tsens = (st & pred).sum() / (st.sum() + 1e-12)
    return 2 * tprec * tsens / (tprec + tsens + 1e-12)


stems = hrf.list_stems(ROOT)[:N]
rows = []
for k, stem in enumerate(stems, 1):
    d = hrf.load(ROOT, stem)
    base = grow(1 - d["im"][:, :, 1], d["fov"])
    mask, length, n, share = run_pipeline(base, d["eye"])
    row = dict(stem=stem, group=d["group"], dice=dice(mask, d["truth"]),
               cldice=cl_dice(mask, d["truth"]), share=share, pieces=n,
               longest=longest_piece(mask), length=length)
    rows.append(row)
    print(f"{k:2d}/{len(stems)} {stem:7s} dice {row['dice']:.3f}  clDice {row['cldice']:.3f}  "
          f"share {share:.3f}  pieces {n:4d}  longest {row['longest']}")

with open("results/hrf_baseline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if len(rows) >= 6:
    col = {"h": "tab:green", "g": "tab:orange", "dr": "tab:red"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, stat in enumerate(["share", "longest"]):
        x = np.array([r[stat] for r in rows]); y = np.array([r["cldice"] for r in rows])
        ax[j].scatter(x, y, c=[col[r["group"]] for r in rows])
        rho = spearmanr(x, y)[0]
        ax[j].set_xlabel(stat); ax[j].set_ylabel("clDice vs hand-drawn map")
        ax[j].set_title(f"{stat} vs clDice  (Spearman rho = {rho:.2f})")
        print(f"Spearman {stat} vs clDice: {rho:.2f}   vs dice: {spearmanr(x, [r['dice'] for r in rows])[0]:.2f}")
    fig.tight_layout(); fig.savefig("figures/07_hrf_baseline.png", dpi=120, bbox_inches="tight")
    print("saved figures/07_hrf_baseline.png")
