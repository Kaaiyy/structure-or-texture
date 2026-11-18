# Structure or Texture?

**Does a vessel segmentation find real anatomy, or just texture that looks like it?**

Most vessel-segmentation methods are judged against hand-drawn maps. Most real images don't have one.
This project asks whether a check that needs no ground truth is possible: scramble the image so it keeps
the same texture (same brightness histogram and spectrum) but loses its arrangement, run the identical
pipeline on the scrambles, and see whether the real image beats them.

![The pipeline finds 7.5x more "vessel" in scrambled noise than in a real retina](figures/05_surrogate.png)

## What I found (short version)

1. **How much "vessel" a method finds is not evidence.** On one retina it finds 7.5x more centreline in a
   scrambled copy than in the real image. Connectivity separates them, not amount.
2. **The popular "largest connected piece / all pieces" measure is weak.** Against a plain phase scramble it looks
   great (share 0.503 vs 0.009). Against a stronger null (IAAFT, which keeps the exact histogram too) it does not
   pass (p = 0.10 on that image), and on 42 held-out HRF images it correlates only weakly with real quality.
3. **The absolute length of the longest connected piece is a decent free baseline.** On 42 held-out HRF images
   it tracks clDice against the expert map (Spearman 0.47, patient-level 95% CI 0.21 to 0.68;
   0.43 after removing the healthy/glaucoma/DR group effect).
4. **The surrogate test detects structure; it does not grade quality.** Every real image passed, including one with
   clDice 0.45. It loses its edge under added noise before the segmentation actually gets worse, and it barely notices blur.
5. **A pretrained deep-learning model did not hallucinate vessels here.** It draws almost nothing on scrambled or
   noisy input and fails by omission. Counting how many pixels it calls vessel already separates its failures
   from its good runs (AUC 0.99), so a surrogate test has nothing to add for that kind of failure.

What is still open: failures where a model draws confident, plausible but wrong structure (unfamiliar image types,
artefacts). That has not been tested.

## Results

**Classical pipeline vs a pretrained model under noise and blur** (42 held-out HRF images, 14 patients, mean clDice):

| Degradation | Classical (tuned on 1 patient) | Little W-Net (trained on DRIVE only) |
|---|---|---|
| none | 0.69 | 0.78 |
| noise 0.3 / 0.5 / 1 / 2 (x image spread) | 0.66 / 0.63 / 0.44 / 0.25 | 0.75 / 0.71 / 0.54 / 0.22 |
| blur sigma 1 / 2 / 4 / 8 (px) | 0.68 / 0.65 / 0.56 / 0.44 | 0.77 / 0.73 / 0.56 / **0.04** |

The deep-learning model is better until things get severe, then collapses under heavy blur (it draws almost nothing).
The classical pipeline degrades gradually. See `figures/15_dl_vs_classic.png`. The classical pipeline was tuned on
one patient and the model was trained on thousands of labelled pixels, so the clean-image gap is expected;
the robustness curves are the useful part.

**IAAFT structure test on a few images** (19 surrogates, so the smallest possible p is 0.05):
5 of 6 held-out images reached p = 0.05, one reached 0.10. In the noise sweep (3 images) the ratio of real to surrogate
fell from about 6 to about 1 by noise level 0.5, while clDice only fell by about 0.05 to 0.1
(`figures/11_degrade.png`). Under blur, 2 of 3 images still passed at sigma 8 while clDice fell to about 0.4.

## Steps

| Step | What it does | Output |
|---|---|---|
| 01 | Remove the camera aperture rim, which a vessel filter mistakes for a vessel | `figures/01_find_eye.png` |
| 02 | Multi-scale Hessian tube filter; green carries vessels (3.2x red) | `figures/02_vessel_filter.png` |
| 03 | Otsu threshold chosen by the data | `figures/03_threshold.png` |
| 04 | Centreline length, branch points, calibre, connectivity to the optic disc | `figures/04_measure.png` |
| 05 | Phase-surrogate dose sweep: same spectrum, structure destroyed | `figures/05_surrogate.png` |
| 06 | IAAFT surrogates and a per-image rank p-value | `figures/06_structure_test.png` |
| 07 | HRF dataset baseline, and a diagnosis of what the first pipeline misses | `figures/07_hrf_baseline.png`, `07b_diagnose.png` |
| 08 | Frozen improved pipeline (local contrast + hysteresis) on 42 held-out images | `figures/08_hrf_heldout.png` |
| 09 | Confound checks: group effect, patient-level bootstrap | `step09_confounds.py` |
| 10 | IAAFT test on 6 held-out images | `results/hrf_iaaft.csv` |
| 11 to 13 | Noise and blur sweeps, finer steps, pooled view | `figures/11_degrade.png`, `13_pooled.png` |
| 14 | First look at a pretrained deep-learning model on real, scrambled, noisy input | `figures/14_dl_look.png` |
| 15 | Classical vs deep learning under noise and blur, all 42 images | `figures/15_dl_vs_classic.png` |
| 16 | Free baseline: does the vessel-pixel fraction detect deep-learning failure? | `step16_baseline.py` |

## Run it

The retina image in steps 01 to 06 ships with scikit-image. Steps 07 onward need the HRF dataset, which is not
included here (it has its own licence; download it yourself). Steps 14 to 16 need the Little W-Net repository
cloned next to this one (MIT licence) and PyTorch.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python step05_surrogate.py

    # for steps 14-16
    pip install torch torchvision "setuptools<81"
    git clone https://github.com/agaldran/lwnet.git ../lwnet

## Honest limits

- **One dataset.** HRF only: 45 images, 15 patients. One patient was used for tuning and excluded from scoring,
  leaving 42 images from 14 patients. Images from one patient are not independent, so confidence intervals
  resample whole patients.
- **Small samples in the key experiments.** The IAAFT test ran on 6 images, the degradation sweeps on 3, with only
  19 surrogates (p cannot go below 0.05). The classical-vs-deep-learning sweep and the free baseline use all 42 images.
- **Unfair on purpose, noted.** The classical pipeline was tuned on one patient; the deep-learning model was trained
  on a different dataset with many labelled images. Do not read the clean-image gap as "AI beats classical".
- **Images are downscaled** to 1400 px wide before everything. Absolute clDice values depend on that.
- **"Failure" is a cutoff** (clDice below 0.5), chosen by me. The AUC is threshold-free in the score, not in that cutoff.
- **Only one deep-learning model** and only degradations that remove signal. Hallucination is untested.
- **Pixels, not millimetres.** The images carry no scale.
- **Not a diagnosis.** Nothing here says anything about anyone's health.
- **Citations still to verify** before this goes into a paper: Frangi (1998), Otsu (1979), Schreiber and Schmitz (1996),
  Theiler et al. (1992).

## Roadmap

- [x] Steps 01 to 06 on one image
- [x] Calibrate against hand-drawn maps (HRF, held-out, with confound checks)
- [x] Degradation experiments (noise, blur)
- [x] Audit a pretrained deep-learning vessel model on scrambled and degraded images
- [ ] Test confident-but-wrong failures (images the model was not built for)
- [ ] DRIVE and CHASE_DB1 (after checking their terms of use)
- [ ] More surrogates (99) on all 42 images, on a computing cluster
- [ ] A second kind of tubular image

## Credits and references

Steps 01 to 05 were inspired by a public post, "5 medical imaging projects you can do at home with Claude".
The retina image in steps 01 to 06 ships with [scikit-image](https://scikit-image.org) (`skimage.data.retina()`).

- HRF (High-Resolution Fundus) database, CC BY 4.0, used for steps 07 onward.
- Galdran, Anjos, Dolz, Chakor, Lombaert, Ben Ayed. "State-of-the-art retinal vessel segmentation with minimalistic
  models." Scientific Reports, 2022. Code and weights: https://github.com/agaldran/lwnet (MIT).
- Shit et al. "clDice, a novel topology-preserving loss function for tubular structure segmentation." CVPR 2021.
- Methods used by name: IAAFT surrogates (Schreiber and Schmitz), Hessian tube filter (Frangi-style), Otsu thresholding.
