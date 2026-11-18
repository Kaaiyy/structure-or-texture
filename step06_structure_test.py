"""Step 06: a proper per-image structure test.
Null = IAAFT surrogates: same brightness histogram AND same spectrum, arrangement random.
Compares the post's 'largest-piece share' with an absolute 'longest piece' statistic."""
import numpy as np
import matplotlib.pyplot as plt
from vessels import load_retina, find_eye, grow, structure_test

N_SURR = 19            # 19 surrogates -> smallest possible p = 1/20 = 0.05

IM = load_retina()
FOV, EYE = find_eye(IM)
BASE = grow(1 - IM[:, :, 1], FOV)

print(f"running {N_SURR} IAAFT surrogates (a few minutes)...")
real, null, p, z = structure_test(BASE, EYE, n_surr=N_SURR)

for i, name in enumerate(["largest-piece share", "longest piece (px)"]):
    print(f"\n{name}")
    print("  real:", round(float(real[i]), 3))
    print("  surrogates:", np.round(np.sort(null[:, i]), 3))
    print(f"  p = {p[i]:.2f}   z = {z[i]:.1f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, name in enumerate(["Largest-piece share (post's measure)", "Longest piece, centreline px"]):
    ax[i].hist(null[:, i], bins=10, color="gray", label="IAAFT surrogates")
    ax[i].axvline(real[i], color="crimson", lw=2, label="real retina")
    ax[i].set_title(f"{name}\np = {p[i]:.2f}, z = {z[i]:.1f}")
    ax[i].legend()
fig.tight_layout()
fig.savefig("figures/06_structure_test.png", dpi=120, bbox_inches="tight")
print("\nsaved figures/06_structure_test.png")
