"""Step 03: let the data pick the threshold (Otsu), not a slider."""
import numpy as np
import matplotlib.pyplot as plt
from vessels import load_retina, find_eye, tubes, grow, segment, pieces

IM = load_retina()
FOV, EYE = find_eye(IM)
V = tubes(grow(1 - IM[:, :, 1], FOV)) * EYE

TH, MASK = segment(V, EYE)
N, BIG = pieces(MASK)
AREA = float(MASK.sum() / EYE.sum())

print("Otsu threshold:", round(TH, 4))
print("vessel fraction of eye:", round(AREA, 4))
print("separate pieces:", N, "(a real vessel tree has 1)")
print("largest piece share:", round(BIG, 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(V[EYE], bins=256, log=True, color="gray")
ax[0].axvline(TH, color="crimson"); ax[0].set_title("Filter response + Otsu cut")
ax[1].imshow(IM); ax[1].imshow(np.ma.masked_where(~MASK, MASK), cmap="spring")
ax[1].set_title(f"Vessels: {N} pieces"); ax[1].axis("off")
fig.savefig("figures/03_threshold.png", dpi=120, bbox_inches="tight")
print("saved figures/03_threshold.png")
