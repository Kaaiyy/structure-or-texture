"""Step 05: destroy the picture and run it again.
Same spectrum, same contrast, phases rotated towards random.
If the pipeline finds STRUCTURE, connectivity should collapse.
If it only finds TEXTURE, it will keep 'finding vessels'."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from vessels import load_retina, find_eye, grow, surrogate, run_pipeline

IM = load_retina()
FOV, EYE = find_eye(IM)
BASE = grow(1 - IM[:, :, 1], FOV)

DOSES = np.linspace(0, 1, 9)
rows = []
for f in DOSES:
    mask, length, n, big = run_pipeline(surrogate(BASE, f), EYE)
    rows.append((f, length, n, big))
    print(f"dose {f:.3f}  length {length:6d}  pieces {n:5d}  largest share {big:.3f}")
    if f == 1:
        noise_img, noise_mask = surrogate(BASE, f), mask

d, L, N, B = map(np.array, zip(*rows))
print("\nfully scrambled vs real: length x", round(L[-1] / L[0], 1),
      " | largest share", round(B[0], 3), "->", round(B[-1], 3))
print("Spearman rho, dose vs length:", round(spearmanr(d, L)[0], 2))
print("Spearman rho, dose vs largest share:", round(spearmanr(d, B)[0], 2))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].imshow(noise_img * EYE, cmap="gray")
ax[0].imshow(np.ma.masked_where(~noise_mask, noise_mask), cmap="spring")
ax[0].set_title("Fully scrambled: 'vessels' found"); ax[0].axis("off")
ax[1].plot(d, L, "o-", color="darkorange"); ax[1].set_xlabel("phase randomised")
ax[1].set_title("Amount of 'vessel' found"); ax[1].set_ylabel("centreline px")
ax[2].plot(d, B, "o-", color="crimson"); ax[2].set_xlabel("phase randomised")
ax[2].set_title("Connectivity (largest piece share)")
fig.tight_layout()
fig.savefig("figures/05_surrogate.png", dpi=120, bbox_inches="tight")
print("saved figures/05_surrogate.png")
