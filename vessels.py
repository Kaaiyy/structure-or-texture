"""Shared code for the structure-or-texture project.
Every step script imports from here, so each function exists in one place."""
import numpy as np
from scipy import ndimage as ndi
from skimage import data


def load_retina():
    """Real fundus photo shipped with scikit-image, scaled to 0-1."""
    return data.retina().astype(float) / 255


def find_eye(im, thresh=0.18, shrink=14):
    """Lit circle = camera aperture (FOV). Shrink it so the rim is not a 'vessel'."""
    fov = im.sum(2) > thresh
    eye = ndi.binary_erosion(fov, np.ones((3, 3)), iterations=shrink)
    return fov, eye


def tubes(im, sig=(1, 2, 3, 4, 6), beta=0.5):
    """Multi-scale Hessian tube filter (Frangi-style), bright ridges only."""
    out = np.zeros_like(im)
    for s in sig:
        a = ndi.gaussian_filter(im, s, order=(0, 2)) * s * s
        b = ndi.gaussian_filter(im, s, order=(2, 0)) * s * s
        c = ndi.gaussian_filter(im, s, order=(1, 1)) * s * s
        t = np.sqrt((a - b) ** 2 + 4 * c * c)
        hi, lo = (a + b + t) / 2, (a + b - t) / 2
        k = np.abs(hi) > np.abs(lo)
        big, sml = np.where(k, hi, lo), np.where(k, lo, hi)
        S = np.sqrt(sml ** 2 + big ** 2)
        rb = sml / (big + 1e-12)
        v = np.exp(-rb ** 2 / (2 * beta ** 2))
        v *= 1 - np.exp(-S * S / (S.max() ** 2 / 2))
        v[big > 0] = 0
        out = np.maximum(out, v)
    return out


def grow(im, fov, n=24):
    """Extend the picture past the aperture so its edge is not a 'vessel'."""
    o = im.copy()
    o[~fov] = im[fov].mean()
    for _ in range(n):
        o[~fov] = ndi.uniform_filter(o, 11)[~fov]
    return o


def otsu(x, bins=256):
    """Otsu's threshold by hand: the cut that leaves the least spread
    inside the two groups it makes. Chosen by the data, not by eye."""
    h, e = np.histogram(x, bins)
    p = h / h.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * (e[:-1] + e[1:]) / 2)
    good = (w > 0) & (w < 1)
    b = (m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12)
    return float(e[1:][np.argmax(np.where(good, b, -1))])


def segment(v, eye, min_size=100):
    """Threshold with Otsu, then drop specks smaller than min_size pixels."""
    th = otsu(v[eye])
    mask = (v > th) & eye
    lab, _ = ndi.label(mask, np.ones((3, 3)))
    sz = np.bincount(lab.ravel())[1:]
    mask &= np.isin(lab, np.nonzero(sz >= min_size)[0] + 1)
    return th, mask


def pieces(mask):
    """Number of connected pieces, and the share held by the largest one."""
    lab, n = ndi.label(mask, np.ones((3, 3)))
    sz = np.bincount(lab.ravel())[1:]
    return n, float(sz.max() / sz.sum()) if n else 0.0


def surrogate(im, f, seed=0):
    """Phase surrogate. Keep every spatial frequency at its measured amplitude,
    rotate the phases a fraction f of the way towards random.
    f=0 is the original image; f=1 keeps texture but destroys arrangement."""
    rs = np.random.default_rng(seed)
    F = np.fft.fft2(im)
    P = np.angle(np.fft.fft2(rs.normal(size=im.shape)))   # symmetric random phases
    g = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * ((1 - f) * np.angle(F) + f * P))))
    return (g - g.mean()) / g.std() * im.std() + im.mean()


def run_pipeline(im, eye):
    """Filter -> Otsu -> clean. Returns (mask, centreline length, pieces, largest share)."""
    from skimage.morphology import skeletonize
    v = tubes(im) * eye
    _, mask = segment(v, eye)
    n, big = pieces(mask)
    return mask, int(skeletonize(mask).sum()), n, big
