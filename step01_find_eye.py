"""Step 01: the lit circle is the camera aperture, not the eye.
A ridge filter will report the aperture rim as the biggest 'vessel',
so we shrink the circle inward before any vessel finding."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data

IM = data.retina().astype(float) / 255          # real fundus photo, 1411 x 1411
FOV = IM.sum(2) > 0.18                           # lit circle = camera aperture
EYE = ndi.binary_erosion(FOV, np.ones((3, 3)), iterations=14)  # pull the edge in
RIM = FOV & ~EYE                                 # the band we throw away

print("shape:", IM.shape)
print("aperture fraction:", round(float(FOV.mean()), 3))
print("eye fraction:", round(float(EYE.mean()), 3))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(IM); ax[0].set_title("Photograph")
ax[1].imshow(FOV, cmap="gray"); ax[1].set_title("Aperture (not the eye)")
ax[2].imshow(IM); ax[2].imshow(np.ma.masked_where(~RIM, RIM), cmap="autumn")
ax[2].set_title("Rim removed (red)")
for a in ax: a.axis("off")
fig.savefig("figures/01_find_eye.png", dpi=120, bbox_inches="tight")
print("saved figures/01_find_eye.png")
