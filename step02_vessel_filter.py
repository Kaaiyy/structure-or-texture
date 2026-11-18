"""Step 02: a vessel is a dark tube. The Hessian (second derivative) says
what a tube is: one curvature large, the other near zero.
We also measure which colour channel carries the vessels."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data

IM = data.retina().astype(float) / 255
FOV = IM.sum(2) > 0.18
EYE = ndi.binary_erosion(FOV, np.ones((3, 3)), iterations=14)


def tubes(im, sig=(1, 2, 3, 4, 6), beta=0.5):
    """Multi-scale Hessian tube filter (Frangi-style), bright ridges only."""
    out = np.zeros_like(im)
    for s in sig:
        a = ndi.gaussian_filter(im, s, order=(0, 2)) * s * s
        b = ndi.gaussian_filter(im, s, order=(2, 0)) * s * s
        c = ndi.gaussian_filter(im, s, order=(1, 1)) * s * s
        t = np.sqrt((a - b) ** 2 + 4 * c * c)
        hi, lo = (a + b + t) / 2, (a + b - t) / 2      # Hessian eigenvalues
        k = np.abs(hi) > np.abs(lo)
        big, sml = np.where(k, hi, lo), np.where(k, lo, hi)
        S = np.sqrt(sml ** 2 + big ** 2)               # overall curvature
        rb = sml / (big + 1e-12)                       # blob vs line
        v = np.exp(-rb ** 2 / (2 * beta ** 2))
        v *= 1 - np.exp(-S * S / (S.max() ** 2 / 2))
        v[big > 0] = 0                                 # keep bright ridges only
        out = np.maximum(out, v)
    return out


def grow(im, n=24):
    """Extend the picture past the aperture so the edge is not a 'vessel'."""
    o = im.copy()
    o[~FOV] = im[FOV].mean()
    for _ in range(n):
        o[~FOV] = ndi.uniform_filter(o, 11)[~FOV]
    return o


# Which colour carries the vessels? Measure it instead of looking it up.
# Invert (1 - channel) so dark vessels become bright ridges.
def peak(c):
    t = tubes(grow(1 - IM[:, :, c])) * EYE
    return float(np.percentile(t[EYE], 99.5))

R = [peak(c) for c in (0, 1, 2)]
print("99.5th pct response R, G, B:", [round(v, 3) for v in R])
print("green / red:", round(R[1] / R[0], 2))

V = tubes(grow(1 - IM[:, :, 1])) * EYE

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(IM[:, :, 1], cmap="gray"); ax[0].set_title("Green channel")
ax[1].imshow(V, cmap="magma"); ax[1].set_title("Tube filter response")
for a in ax: a.axis("off")
fig.savefig("figures/02_vessel_filter.png", dpi=120, bbox_inches="tight")
print("saved figures/02_vessel_filter.png")
