"""Step 14a: make three versions of ONE held-out HRF image for the deep-learning audit:
  real.png       the image as is (1400 px wide, outside the eye set to black)
  scrambled.png  same amplitude spectrum, random phases (all vessel structure destroyed)
  noisy.png      the real image plus strong noise
plus mask.png (field of view). Output goes OUTSIDE the repo: ../lwnet_inputs/
usage: python step14a_make_inputs.py data/hrf [stem=09_g]"""
import sys, os
import numpy as np
from PIL import Image
import hrf

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hrf"
STEM = sys.argv[2] if len(sys.argv) > 2 else "09_g"
OUT = os.path.abspath(os.path.join("..", "lwnet_inputs"))
os.makedirs(OUT, exist_ok=True)

d = hrf.load(ROOT, STEM)
im, fov = d["im"], d["fov"]
rs = np.random.default_rng(0)

def save(name, a):
    a = np.clip(a, 0, 1) * fov[:, :, None]
    Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(OUT, name))

# phase scramble: SAME random phases for all 3 colour channels, amplitudes kept
theta = np.angle(np.fft.rfft2(rs.normal(size=fov.shape)))
scr = np.stack([np.fft.irfft2(np.abs(np.fft.rfft2(im[:, :, c])) * np.exp(1j * theta),
                              s=fov.shape) for c in range(3)], axis=2)
sd = im[fov].std()
save("real.png", im)
save("scrambled.png", scr)
save("noisy.png", im + rs.normal(0, 1.0 * sd, im.shape))
Image.fromarray((fov * 255).astype(np.uint8)).save(os.path.join(OUT, "mask.png"))
print("saved real / scrambled / noisy / mask in", OUT)
