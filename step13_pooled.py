"""Step 13: pool every degradation run (steps 11 + 12) into one picture.
Question: how does the test's edge over texture (real / surrogate) relate to
how good the segmentation really is (clDice), for noise vs blur?
No heavy compute: reads results/hrf_degrade.csv and results/hrf_sweep.csv."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

pts = []   # (stem, kind, level, cldice, ratio, p)
for r in csv.DictReader(open("results/hrf_degrade.csv")):
    k = float(r["noise"])
    pts.append((r["stem"], "noise", k, float(r["cldice"]), float(r["ratio"]), float(r["p_longest"])))
    if k == 0:   # clean image is also the blur baseline
        pts.append((r["stem"], "blur", 0.0, float(r["cldice"]), float(r["ratio"]), float(r["p_longest"])))
for r in csv.DictReader(open("results/hrf_sweep.csv")):
    pts.append((r["stem"], r["kind"], float(r["level"]), float(r["cldice"]),
                float(r["ratio"]), float(r["p_longest"])))

kinds = {"noise": ("o", "tab:blue"), "blur": ("^", "tab:orange")}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for kind, (mk, c) in kinds.items():
    q = [p for p in pts if p[1] == kind]
    ax[0].scatter([p[3] for p in q], [p[4] for p in q], marker=mk, c=c, label=kind, alpha=0.8)
ax[0].axhline(1, c="grey", ls="--")
ax[0].set_xlabel("clDice vs hand-drawn map (segmentation quality)")
ax[0].set_ylabel("real / surrogate (test's edge over texture)")
ax[0].set_title("Edge vs quality"); ax[0].legend()
for kind, (mk, c) in kinds.items():
    for stem in sorted({p[0] for p in pts}):
        q = sorted([p for p in pts if p[1] == kind and p[0] == stem], key=lambda p: p[2])
        ax[1].plot([p[3] for p in q], [p[5] for p in q], marker=mk, c=c, alpha=0.5)
ax[1].axhline(0.05, c="grey", ls="--"); ax[1].text(0.2, 0.08, "p = 0.05 (best possible)", color="grey")
ax[1].set_xlabel("clDice"); ax[1].set_ylabel("p-value (lower = structure)")
ax[1].set_title("Each line = one image, increasing degradation")
fig.tight_layout(); fig.savefig("figures/13_pooled.png", dpi=120, bbox_inches="tight")
print("saved figures/13_pooled.png\n")

for kind in kinds:
    q = [p for p in pts if p[1] == kind]
    rho, pv = spearmanr([p[3] for p in q], [p[4] for p in q])
    bad = [p for p in q if p[3] < 0.5]
    print(f"{kind:5s}: n={len(q)}  rho(ratio, clDice) = {rho:.2f} (p={pv:.3f}, images not independent)")
    print(f"       runs with clDice < 0.5: {len(bad)};  still p = 0.05 in {sum(p[5] <= 0.05 for p in bad)} of them")
    ok = [p for p in q if p[3] >= 0.6]
    print(f"       runs with clDice >= 0.6: {len(ok)};  lost p = 0.05 in {sum(p[5] > 0.05 for p in ok)} of them\n")
