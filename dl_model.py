"""Run the pretrained Little W-Net (Galdran et al., trained on DRIVE) from our own
scripts. Expects the lwnet repo in ~/Documents/lwnet (or set LWNET_DIR).
Same steps as lwnet's predict_one_image.py: crop to the eye, resize to 512x512,
average 4 flips, sigmoid, resize back, zero outside the eye."""
import os, sys, warnings
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from skimage.measure import regionprops

LW = os.environ.get("LWNET_DIR", os.path.expanduser("~/Documents/lwnet"))
sys.path.insert(0, LW)
warnings.filterwarnings("ignore")
from models.get_model import get_arch            # noqa: E402
from utils.model_saving_loading import load_model  # noqa: E402
from utils import paired_transforms_tv04 as p_tr   # noqa: E402

THRESH = 0.4196          # the repo's default binarising threshold
_model = None


def _get():
    global _model
    if _model is None:
        m = get_arch("wnet")
        m.mode = "eval"
        m, _ = load_model(m, os.path.join(LW, "experiments", "wnet_drive"), "cpu")
        m.eval()
        _model = m
    return _model


def dl_prob(im, fov, size=512):
    """im: HxWx3 floats in 0-1, fov: bool mask. Returns full-size vessel probability."""
    model = _get()
    minr, minc, maxr, maxc = regionprops(fov.astype(int))[0].bbox
    arr = (np.clip(im, 0, 1) * fov[:, :, None] * 255).astype(np.uint8)
    crop = Image.fromarray(arr[minr:maxr, minc:maxc])
    tens = p_tr.Compose([p_tr.Resize((size, size)), p_tr.ToTensor()])(crop)
    outs = []
    with torch.no_grad():
        for dims in ([], [-1], [-2], [-1, -2]):
            x = tens.flip(*dims) if dims else tens
            p = torch.sigmoid(model(x.unsqueeze(0)).squeeze(0))
            outs.append(p.flip(*dims) if dims else p)
    pred = torch.stack(outs).mean(0).numpy()[-1]
    pred = resize(pred, crop.size[::-1], order=3)
    full = np.zeros(fov.shape)
    full[minr:maxr, minc:maxc] = pred
    full[~fov] = 0
    return full


def dl_mask(im, fov, thresh=THRESH):
    return dl_prob(im, fov) > thresh
