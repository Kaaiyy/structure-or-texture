"""Step 09: is the 'longest piece' vs clDice relationship real, or is it just
'healthy eyes score higher on everything'? Uses results/hrf_heldout.csv only
(no images needed, runs instantly).
  1. Spearman WITHIN each group.
  2. Partial Spearman, removing the group effect.
  3. Patient-level bootstrap CI (images from one patient are not independent)."""
import csv
import numpy as np
from scipy.stats import spearmanr, rankdata

rows = list(csv.DictReader(open("results/hrf_heldout.csv")))
grp = np.array([r["group"] for r in rows])
pat = np.array([r["stem"].split("_")[0] for r in rows])
Y = np.array([float(r["cldice"]) for r in rows])
STATS = {s: np.array([float(r[s]) for r in rows]) for s in ("longest", "share", "pieces")}


def partial_spearman(x, y, g):
    """Rank both variables, remove group means, correlate what is left."""
    rx, ry = rankdata(x), rankdata(y)
    for lab in np.unique(g):
        rx[g == lab] -= rx[g == lab].mean()
        ry[g == lab] -= ry[g == lab].mean()
    return np.corrcoef(rx, ry)[0, 1]


def boot_ci(x, y, pat, n=2000, seed=0):
    """Resample whole PATIENTS, so the 3 images of a patient stay together."""
    rs = np.random.default_rng(seed)
    ids = np.unique(pat)
    out = []
    for _ in range(n):
        pick = rs.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.nonzero(pat == p)[0] for p in pick])
        out.append(spearmanr(x[idx], y[idx])[0])
    return np.nanpercentile(out, [2.5, 97.5])


print(f"n = {len(rows)} images, {len(np.unique(pat))} patients\n")
for name, x in STATS.items():
    rho, p = spearmanr(x, Y)
    lo, hi = boot_ci(x, Y, pat)
    print(f"{name} vs clDice")
    print(f"  all images:         rho = {rho:5.2f}   (p = {p:.3f}, naive)")
    print(f"  patient bootstrap:  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  group effect removed: rho = {partial_spearman(x, Y, grp):5.2f}")
    for g, label in (("h", "healthy"), ("g", "glaucoma"), ("dr", "diabetic retinopathy")):
        m = grp == g
        r, pp = spearmanr(x[m], Y[m])
        print(f"  within {label:21s} n={m.sum():2d}  rho = {r:5.2f}  (p = {pp:.2f})")
    print()
