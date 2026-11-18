"""Step 14b: look at what the pretrained deep-learning model (Little W-Net, trained
on DRIVE) draws on the real, scrambled and noisy versions of one image.
Reads ../lwnet_inputs and ../lwnet_outputs. Writes figures/14_dl_look.png"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from vessels import pieces, longest_piece

names = ["real", "scrambled", "noisy"]
fig, ax = plt.subplots(3, 3, figsize=(12, 12))
for j, n in enumerate(names):
    inp = np.asarray(Image.open(f"../lwnet_inputs/{n}.png").convert("RGB"))
    prob = np.asarray(Image.open(f"../lwnet_outputs/{n}/{n}_seg.png").convert("L"))
    b = np.asarray(Image.open(f"../lwnet_outputs/{n}/{n}_bin_seg.png").convert("L")) > 0
    npc, share = pieces(b)
    print(f"{n:10s} vessel pixels {b.mean()*100:5.2f}% of image | pieces {npc:5d} | "
          f"largest-piece share {share:.2f} | longest {longest_piece(b)}")
    ax[0, j].imshow(inp); ax[0, j].set_title(f"input: {n}")
    ax[1, j].imshow(prob, cmap="gray"); ax[1, j].set_title("model probability")
    ax[2, j].imshow(b, cmap="gray"); ax[2, j].set_title("model's vessels (binary)")
for a in ax.ravel():
    a.axis("off")
fig.tight_layout(); fig.savefig("figures/14_dl_look.png", dpi=90, bbox_inches="tight")
print("saved figures/14_dl_look.png")
