"""Step 04: thin the vessel mask to a 1-pixel centreline, then count.
Length, branch points, ends, calibre - and how much connects to the optic disc."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from vessels import load_retina, find_eye, tubes, grow, segment, pieces

IM = load_retina()
FOV, EYE = find_eye(IM)
V = tubes(grow(1 - IM[:, :, 1], FOV)) * EYE
TH, MASK = segment(V, EYE)

SK = skeletonize(MASK)                                  # centreline
NB = ndi.convolve(SK.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") * SK
LEN = int(SK.sum())                                     # centreline pixels
BR = int(((NB >= 4) & SK).sum())                        # 3+ neighbours = junction
EN = int(((NB == 2) & SK).sum())                        # 1 neighbour = end
CAL = float(MASK.sum()) / LEN                           # mean width in px

# The optic disc is the brightest part of the eye. Vessels leave from it,
# so a real tree should be reachable from there.
B = ndi.gaussian_filter(IM.sum(2), 25) * EYE
DY, DX = np.unravel_index(np.argmax(B), B.shape)
lab, n = ndi.label(MASK, np.ones((3, 3)))
yy, xx = np.ogrid[:MASK.shape[0], :MASK.shape[1]]
near = (yy - DY) ** 2 + (xx - DX) ** 2 < 120 ** 2
ids = np.unique(lab[near & MASK])
CONN = np.isin(lab, ids) & MASK
SHARE = float(CONN.sum() / MASK.sum())

print("centreline length (px):", LEN)
print("branch points:", BR, " end points:", EN)
print("mean calibre (px):", round(CAL, 2))
print("optic disc at (row, col):", (int(DY), int(DX)))
print("share connected to disc:", round(SHARE, 3))

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(IM)
ax.imshow(np.ma.masked_where(~CONN, CONN), cmap="spring", alpha=0.9)
ax.imshow(np.ma.masked_where(~(MASK & ~CONN), MASK), cmap="autumn_r", alpha=0.9)
ax.plot(DX, DY, "wo", ms=14, mfc="none", mew=2)
ax.set_title(f"Pink: reaches the disc ({SHARE:.0%})   Yellow: orphaned")
ax.axis("off")
fig.savefig("figures/04_measure.png", dpi=120, bbox_inches="tight")
print("saved figures/04_measure.png")
