"""Step 15b: summarise the head-to-head sweep (classical vs deep learning).
Mean clDice per setting, paired difference (DL minus classical) with a
patient-level bootstrap 95% CI, and a figure. Reads results/hrf_dl_sweep.csv."""
import csv
import numpy as np
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/hrf_dl_sweep.csv")))
pat = np.array([r["stem"].split("_")[0] for r in rows])
kind = np.array([r["kind"] for r in rows]); lev = np.array([float(r["level"]) for r in rows])
cc = np.array([float(r["cl_classic"]) for r in rows]); cd = np.array([float(r["cl_dl"]) for r in rows])
ids = np.unique(pat)
rs = np.random.default_rng(0)

def boot(mask, n=2000):
    """Resample whole patients; return 95% CI of mean(DL - classical)."""
    diff = (cd - cc)
    per = {p: diff[mask & (pat == p)] for p in ids}
    out = []
    for _ in range(n):
        pick = rs.choice(ids, len(ids))
        out.append(np.concatenate([per[p] for p in pick]).mean())
    return np.percentile(out, [2.5, 97.5])

print(f"{len(set(r['stem'] for r in rows))} images, {len(ids)} patients\n")
print("setting      classical  deep-learning   DL minus classical  (95% CI, patient bootstrap)")
setups = [("noise", l) for l in sorted(set(lev[kind == "noise"]))] + \
         [("blur", l) for l in sorted(set(lev[kind == "blur"]))]
res = {}
for k, l in setups:
    m = (kind == k) & (lev == l)
    lo, hi = boot(m)
    res[(k, l)] = (cc[m].mean(), cd[m].mean())
    print(f"{k:5s} {l:4g}     {cc[m].mean():.2f}       {cd[m].mean():.2f}         {cd[m].mean()-cc[m].mean():+.2f}   [{lo:+.2f}, {hi:+.2f}]")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for a, k in zip(ax, ("noise", "blur")):
    xs = [l for kk, l in setups if kk == k]
    a.plot(xs, [res[(k, l)][0] for l in xs], "o-", c="tab:blue", label="classical pipeline")
    a.plot(xs, [res[(k, l)][1] for l in xs], "s-", c="tab:red", label="deep learning (DRIVE-trained)")
    a.set_xlabel("added noise (x image spread)" if k == "noise" else "blur sigma (px)")
    a.set_title(f"{k}: mean clDice over held-out HRF"); a.legend()
ax[0].set_ylabel("clDice vs hand-drawn map")
fig.tight_layout(); fig.savefig("figures/15_dl_vs_classic.png", dpi=120, bbox_inches="tight")
print("\nsaved figures/15_dl_vs_classic.png")
