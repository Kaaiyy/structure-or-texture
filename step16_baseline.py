"""Step 16: the free baseline. Does 'how many pixels did the model call vessel?'
already tell us when the deep-learning model has failed (clDice low)?
No heavy compute: reads results/hrf_dl_sweep.csv.
Failure = the model's clDice < FAIL (default 0.5).
AUC: 1.0 = perfectly separates failures from good runs, 0.5 = coin flip."""
import csv, sys
import numpy as np
from scipy.stats import spearmanr, rankdata

FAIL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
rows = list(csv.DictReader(open("results/hrf_dl_sweep.csv")))
stem = np.array([r["stem"] for r in rows]); pat = np.array([r["stem"].split("_")[0] for r in rows])
kind = np.array([r["kind"] for r in rows]); lev = np.array([float(r["level"]) for r in rows])
cl = np.array([float(r["cl_dl"]) for r in rows]); pct = np.array([float(r["dl_vessel_pct"]) for r in rows])
clean = {s: pct[(stem == s) & (kind == "noise") & (lev == 0)][0] for s in set(stem)}
rel = pct / np.array([clean[s] for s in stem])     # vs the SAME image's clean output (oracle)

def auc(score, fail):
    """P(score of a failure < score of a good run); ties count half."""
    r = rankdata(score); n1 = fail.sum(); n0 = (~fail).sum()
    return (r[~fail].sum() - n0 * (n0 + 1) / 2) / (n0 * n1) if n1 and n0 else np.nan

def boot_auc(score, fail, n=1000, seed=0):
    rs = np.random.default_rng(seed); ids = np.unique(pat); out = []
    for _ in range(n):
        idx = np.concatenate([np.nonzero(pat == p)[0] for p in rs.choice(ids, len(ids))])
        out.append(auc(score[idx], fail[idx]))
    return np.nanpercentile(out, [2.5, 97.5])

fail = cl < FAIL
print(f"{len(rows)} runs, {fail.sum()} failures (clDice < {FAIL}), {(~fail).sum()} good\n")
print("setting      mean clDice   mean vessel pixels (% of eye)")
for k, l in sorted({(a, b) for a, b in zip(kind, lev)}, key=lambda t: (t[0] != "noise", t[1])):
    m = (kind == k) & (lev == l)
    print(f"{k:5s} {l:4g}      {cl[m].mean():.2f}          {pct[m].mean():5.2f}")
print()
for name, sc in (("vessel pixels alone (usable on a new image)", pct),
                 ("vessel pixels / that image's clean output (needs a clean reference)", rel)):
    lo, hi = boot_auc(sc, fail)
    print(f"AUC {auc(sc, fail):.2f}  95% CI [{lo:.2f}, {hi:.2f}]  <- {name}")
print(f"\nSpearman(vessel pixels, clDice) = {spearmanr(pct, cl)[0]:.2f}  (all runs pooled)")
