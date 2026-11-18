"""Figure 11: as noise increases, clDice falls (left) and the IAAFT test
loses its edge over texture (right). One line per image."""
import csv
import numpy as np
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/hrf_degrade.csv")))
stems = sorted({r["stem"] for r in rows})
col = dict(zip(stems, ["tab:orange", "tab:green", "tab:red"]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for s in stems:
    r = [x for x in rows if x["stem"] == s]
    n = [float(x["noise"]) for x in r]
    ax[0].plot(n, [float(x["cldice"]) for x in r], "o-", c=col[s], label=s)
    ax[1].plot(n, [float(x["ratio"]) for x in r], "o-", c=col[s], label=s)
ax[0].set_ylabel("clDice vs hand-drawn map")
ax[1].set_ylabel("real / surrogate (longest piece)")
ax[1].axhline(1, c="grey", ls="--"); ax[1].text(2.1, 1.1, "= texture level", color="grey")
for a in ax:
    a.set_xlabel("added noise (x the image's own spread)"); a.legend()
ax[0].set_title("Segmentation quality falls"); ax[1].set_title("Test edge over texture vanishes")
fig.tight_layout(); fig.savefig("figures/11_degrade.png", dpi=120, bbox_inches="tight")
print("saved figures/11_degrade.png")
