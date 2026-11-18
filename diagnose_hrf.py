"""Diagnose why the pipeline scores low on HRF.
Overlay: green = correct (TP), red = false alarm (FP), blue = missed (FN).
usage: python diagnose_hrf.py data/hrf [stem stem ...]"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import hrf
from vessels import grow, tubes, segment

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
STEMS = sys.argv[2:] or ["01_h", "01_g", "01_dr"]

fig, ax = plt.subplots(1, len(STEMS), figsize=(6 * len(STEMS), 5))
ax = np.atleast_1d(ax)
for a, stem in zip(ax, STEMS):
    d = hrf.load(ROOT, stem)
    v = tubes(grow(1 - d["im"][:, :, 1], d["fov"])) * d["eye"]
    th, m = segment(v, d["eye"])
    t = d["truth"]
    tp, fp, fn = m & t, m & ~t, ~m & t
    prec = tp.sum() / max(m.sum(), 1)
    rec = tp.sum() / max(t.sum(), 1)
    print(f"{stem}: Otsu cut {th:.3f} | found {m.mean()*100:.1f}% of image, "
          f"truth {t.mean()*100:.1f}% | precision {prec:.2f} recall {rec:.2f}")
    rgb = np.dstack([d["im"][:, :, 1]] * 3) * 0.6
    rgb[tp] = [0, 1, 0]
    rgb[fp] = [1, 0, 0]
    rgb[fn] = [0.2, 0.4, 1]
    a.imshow(rgb)
    a.set_title(f"{stem}: precision {prec:.2f}, recall {rec:.2f}")
    a.axis("off")
fig.tight_layout()
fig.savefig("figures/07b_diagnose.png", dpi=100, bbox_inches="tight")
print("saved figures/07b_diagnose.png")
