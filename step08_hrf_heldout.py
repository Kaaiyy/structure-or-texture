"""Step 08: score the FROZEN pipeline (local contrast + hysteresis) on the
held-out HRF images. Patient 01 was used for tuning, so it is excluded.
Question: do the 'share' and 'longest piece' statistics track how good the
segmentation really is (clDice against the expert's map)?

usage: python step08_hrf_heldout.py data/hrf [exclude_prefix]"""
import sys, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import hrf
from vessels import grow, pipeline_v2, pieces, longest_piece, dice, cl_dice

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
EXCLUDE = sys.argv[2] if len(sys.argv) > 2 else "01"       # tuning patient
PIPE = dict(local=True, hyst=True)                          # FROZEN

stems = [s for s in hrf.list_stems(ROOT) if not s.startswith(EXCLUDE + "_")]
rows = []
for k, stem in enumerate(stems, 1):
    d = hrf.load(ROOT, stem)
    m = pipeline_v2(grow(1 - d["im"][:, :, 1], d["fov"]), d["eye"], **PIPE)
    n, share = pieces(m)
    t = d["truth"]
    tp = (m & t).sum()
    row = dict(stem=stem, group=d["group"], precision=tp / max(m.sum(), 1),
               recall=tp / max(t.sum(), 1), dice=dice(m, t), cldice=cl_dice(m, t),
               share=share, pieces=n, longest=longest_piece(m))
    rows.append(row)
    print(f"{k:2d}/{len(stems)} {stem:7s} dice {row['dice']:.3f} clDice {row['cldice']:.3f} "
          f"share {share:.3f} pieces {n:4d} longest {row['longest']}")

with open("results/hrf_heldout.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

print("\nmean by group (precision / recall / dice / clDice):")
for g in ("h", "g", "dr"):
    r = [x for x in rows if x["group"] == g]
    if r:
        print(f"  {hrf.GROUPS[g]:22s} n={len(r):2d}  " + " / ".join(
            f"{np.mean([x[c] for x in r]):.2f}" for c in ("precision", "recall", "dice", "cldice")))

if len(rows) >= 6:
    print(f"\nSpearman over n={len(rows)} images (images from one patient are not independent):")
    col = {"h": "tab:green", "g": "tab:orange", "dr": "tab:red"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, stat in enumerate(["share", "longest"]):
        x = np.array([r[stat] for r in rows]); y = np.array([r["cldice"] for r in rows])
        rho, p = spearmanr(x, y)
        print(f"  {stat:8s} vs clDice: rho = {rho:.2f}, p = {p:.3f}")
        ax[j].scatter(x, y, c=[col[r["group"]] for r in rows])
        ax[j].set_xlabel(stat); ax[j].set_ylabel("clDice vs hand-drawn map")
        ax[j].set_title(f"{stat} vs clDice (rho = {rho:.2f}, p = {p:.2f})")
    fig.tight_layout(); fig.savefig("figures/08_hrf_heldout.png", dpi=120, bbox_inches="tight")
    print("saved figures/08_hrf_heldout.png")
