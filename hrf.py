"""Loader for the HRF fundus database (Budai et al. 2013, CC BY 4.0).
Layout: images/NN_{h,g,dr}.jpg|JPG, mask/NN_x_mask.tif (field of view),
manual1/NN_x.* (hand-drawn vessels). Matches files by name prefix, so it does
not depend on the exact extension of the hand-drawn maps."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GROUPS = {"h": "healthy", "g": "glaucoma", "dr": "diabetic retinopathy"}


def list_stems(root):
    """Sorted image stems like '01_h', '01_dr'."""
    stems = {p.stem for p in Path(root, "images").iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    return sorted(stems)


def group_of(stem):
    return stem.split("_", 1)[1]


def _find(folder, stem):
    """File in folder whose name starts with stem (case-insensitive)."""
    hits = [p for p in Path(folder).iterdir()
            if p.stem.lower().startswith(stem.lower())]
    if not hits:
        raise FileNotFoundError(f"no file starting with '{stem}' in {folder}")
    hits.sort(key=lambda p: len(p.name))        # shortest = exact match
    return hits[0]


def _binary(path, size):
    a = Image.open(path)
    if a.mode not in ("L", "1"):
        a = a.convert("L")
    a = a.resize(size, Image.NEAREST)
    return np.asarray(a) > 0


def load(root, stem, width=1400, shrink=14):
    """Return dict: im (0-1 RGB), fov, eye, truth - all resized to `width` px wide."""
    img = Image.open(_find(Path(root, "images"), stem)).convert("RGB")
    w0, h0 = img.size
    size = (width, round(h0 * width / w0))
    im = np.asarray(img.resize(size, Image.LANCZOS), float) / 255
    fov = _binary(_find(Path(root, "mask"), stem), size)
    truth = _binary(_find(Path(root, "manual1"), stem), size)
    eye = ndi.binary_erosion(fov, np.ones((3, 3)), iterations=shrink)
    return dict(im=im, fov=fov, eye=eye, truth=truth & eye, stem=stem,
                group=group_of(stem), scale=width / w0)
