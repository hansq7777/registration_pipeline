# Histology Registration Preprocessing Experiment Design

Date: 2026-03-26

Branch:
- `experiment/histology_registration_preproc_20260326`

## Current Local Strategy

Current Step 5 / evaluation defaults in code:
- registration working long edge: `1024`
- optional draft working long edge: `512`
- mask mode used in the recent epoch evaluation harness: `tissue_only`
- image preprocessing used in the recent full fake-myelin and raw-vs-fake experiments:
  - convert to grayscale
  - percentile clipping inside tissue mask
  - background fixed to white
- optional pre-blur exists in Step 5 UI, default `0.0`

Relevant code:
- [pair_registration.py](/mnt/c/work/registration_pipeline/histology/gui_mvp/hitl_gui/application/pair_registration.py)
- [workflow_window.py](/mnt/c/work/registration_pipeline/histology/gui_mvp/hitl_gui/windows/workflow_window.py)
- [prepare_registration_preprocessing_and_eval.py](/mnt/c/work/registration_pipeline/histology/tools/prepare_registration_preprocessing_and_eval.py)
- [evaluate_fake_myelin_epochs_registration.py](/mnt/c/work/registration_pipeline/histology/tools/evaluate_fake_myelin_epochs_registration.py)

Local variants already implemented:
- `baseline`
- `clip`
- `clip_norm`
- `clip_norm_clahe`

Local conclusion so far:
- `clip` was the most stable among the tested grayscale intensity variants.
- `clip_norm` can help on some cases.
- `clip_norm_clahe` often raises MI/CC while worsening Dice/HD95.

## What The Literature Suggests

Common themes across differently stained histology and multi-modal registration:
- coarse but robust prealignment, then multiresolution refinement
- grayscale or structurally simplified representations are often preferred over raw color
- background removal / tissue masking is important
- downsampling is standard for coarse registration
- edge or modality-robust descriptors often outperform raw intensity similarity when intensity semantics differ across modalities

Useful sources:
- ANHIR challenge summary:
  - best methods used coarse robust initial alignment, then non-rigid registration, multiresolution, and careful data-specific tuning
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC7584382/
- 3-step NGF histology registration:
  - prealignment + coarse parametric + nonlinear
  - uses NGF specifically because it matches edges rather than absolute intensity
  - https://arxiv.org/abs/1903.12063
- Zooming / NGF for differently stained histology:
  - affine on coarse resolution, then higher-resolution refinement
  - NGF explicitly chosen for differently stained images
  - https://www.mic.uni-luebeck.de/fileadmin/mic/publications/2014/Lotz_Berger_-_Zooming_in__High_Resolution_3D_Reconstruction_of_Differently_Stained_Histological_Whole_Slide_Images_SPIE-submitted.pdf
- Survey of histology section registration:
  - differently stained registration often relies on comparable clusters / probability maps rather than raw intensity identity
  - successive Gaussian smoothing and neighborhood color features have been used to stabilize cross-stain matching
  - https://www.nmr.mgh.harvard.edu/~iglesias/pdf/survey_histo_recon.pdf
- Macenko stain normalization:
  - common histology color normalization baseline
  - https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
- Vahadane stain normalization:
  - structure-preserving stain normalization
  - https://pubmed.ncbi.nlm.nih.gov/27164577/
- MIND descriptor:
  - modality-independent local self-similarity descriptor for multimodal registration
  - useful when direct intensity relation is non-functional
  - https://doi.org/10.1016/j.media.2012.05.008

## Interpretation For This Project

For `nissl <-> myelin` coarse registration:
- raw intensity similarity is unreliable because dark and bright regions do not carry equivalent biological meaning across stains
- aggressive local contrast enhancement can amplify non-corresponding texture
- therefore the next experiments should prioritize:
  - modest smoothing
  - structural simplification
  - tissue-only metric masks
  - multiresolution
  - edge / descriptor representations over color harmonization alone

Color normalization is still worth testing, but should not be the first priority:
- Macenko / Vahadane / Reinhard are most relevant when scanner/stain variation is a major issue within a stain family
- they are less likely than NGF/MIND-style structure emphasis to solve cross-stain semantics directly

## Proposed Experiment Program

Keep the harness fixed:
- usable pairs only
- same pair set
- same evaluation geometry
- same mask mode: `tissue_only`
- same coarse registration flow:
  - `input-only`
  - `rigid-only`
  - `rigid + affine`
- same working long edge: `1024`
- optional draft reruns at `512`

### Phase 1: Intensity-Side Preprocessing

These are cheap and closest to the current codebase.

1. `clip_white`
- current baseline
- grayscale
- 1-99 percentile clip inside tissue
- outside tissue fixed to white

2. `clip_white_blur_0.5`
3. `clip_white_blur_1.0`
4. `clip_white_blur_1.5`
5. `clip_white_blur_2.0`
- purpose:
  - suppress stain-specific microtexture
  - let MI focus more on coarse morphology

6. `clip_norm_white`
- current `clip_norm`

7. `clip_norm_white_blur_1.0`

8. `clip_norm_clahe_white`
- keep only as a control because local results already suggest instability

9. `clip_norm_clahe_white_blur_1.0`
- test whether blur can rescue CLAHE by damping over-enhanced texture

### Phase 2: Structure-Emphasis Representations

These are still preprocess-like, but closer to cross-modal registration practice.

10. `gradient_mag_blur_1.0`
- grayscale after clip
- Gaussian blur
- Sobel gradient magnitude
- background fixed to zero or white consistently

11. `gradient_mag_blur_1.5`

12. `laplacian_of_gaussian`
- LoG-style edge emphasis on clipped grayscale

13. `ngf_like_proxy`
- practical proxy representation:
  - blur
  - normalized gradient components or gradient magnitude
- purpose:
  - approximate the literature trend that edge alignment is more robust across stains

### Phase 3: Color / Stain Normalization Branch

Run only if Phase 1-2 does not clearly help.

14. `reinhard_then_gray_clip`
15. `macenko_then_gray_clip`
16. `vahadane_then_gray_clip`

Important:
- these should be evaluated on the RGB input before grayscale conversion
- this phase is more engineering-heavy and should be done after the simpler grayscale/blur/edge branch

### Phase 4: Descriptor Branch

This is no longer just preprocessing, but it is the most literature-aligned path if intensity preprocessing saturates.

17. `mind_descriptor + rigid`
18. `mind_descriptor + affine`

This should be treated as a separate branch of work because it changes the similarity representation, not only the image normalization.

## Recommended Order

1. Phase 1 first
2. Then Phase 2
3. Only then Phase 3
4. Descriptor branch last

Reason:
- Phase 1 is cheapest and easiest to integrate with current harness
- Phase 2 is still lightweight and more directly motivated by cross-stain registration literature
- Phase 3 is common in pathology but less likely to solve the main semantic mismatch by itself
- Phase 4 is promising but is a metric/representation change rather than a simple preprocessing tweak

## Minimum Concrete Matrix To Run First

If we want a focused first batch instead of the full list, use:
- `clip_white`
- `clip_white_blur_0.5`
- `clip_white_blur_1.0`
- `clip_white_blur_1.5`
- `clip_white_blur_2.0`
- `clip_norm_white`
- `clip_norm_white_blur_1.0`
- `clip_norm_clahe_white`
- `gradient_mag_blur_1.0`
- `gradient_mag_blur_1.5`

For each condition:
- evaluate `input-only`
- evaluate `rigid-only`
- evaluate `rigid + affine`

Primary ranking:
- mean / median Dice
- mean / median HD95
- count of cases where rigid improves over input
- count of cases where affine improves over rigid

Secondary ranking:
- MI
- CC
- runtime

## Expected Outcomes

Most likely winners:
- `clip_white_blur_0.5` or `clip_white_blur_1.0`
- possibly `gradient_mag_blur_1.0`

Most likely failure modes:
- CLAHE variants improving MI/CC while worsening Dice/HD95
- too much blur (`>= 2.0`) washing out hemisphere boundaries
- stain normalization changing appearance while not solving cross-stain structural mismatch

## Practical Recommendation

Before broadening the fake-myelin work further, the next best experiment is:
- run the focused Phase 1 + Phase 2 matrix on `raw_nissl`
- keep `epoch30 fake myelin` only as a secondary comparator after a winner emerges on `raw_nissl`

That keeps the experiment interpretable:
- first improve preprocessing on the real cross-stain task
- then ask whether fake-myelin still adds anything under the better preprocessing regime
