<div align="center">

<h1>MOCHI</h1>

<h3>Registration-Free Learnable Multi-View Capture of Faces<br>in Dense Semantic Correspondence</h3>

<a href="https://filby89.github.io/mochi/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2605.01450" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2605.01450-b31b1b" alt="arXiv"></a>
<a href="https://www.youtube.com/watch?v=-dicD0PMbC8" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Video-YouTube-red?logo=youtube&logoColor=red" alt="Video"></a>

<p>
  <a href="https://filby89.github.io" target="_blank" rel="noopener noreferrer">Panagiotis P. Filntisis</a> &nbsp;·&nbsp;
  <a href="https://georgeretsi.github.io" target="_blank" rel="noopener noreferrer">George Retsinas</a> &nbsp;·&nbsp;
  <a href="https://radekd91.github.io" target="_blank" rel="noopener noreferrer">Radek Danecek</a> &nbsp;·&nbsp;
  <a href="https://vanessik.github.io" target="_blank" rel="noopener noreferrer">Vanessa Sklyarova</a> &nbsp;·&nbsp;
  <a href="https://robotics.ntua.gr/members/maragos/" target="_blank" rel="noopener noreferrer">Petros Maragos</a> &nbsp;·&nbsp;
  <a href="https://sites.google.com/site/bolkartt/" target="_blank" rel="noopener noreferrer">Timo Bolkart</a>
</p>

<h4>CVPR 2026</h4>

</div>

<p align="center">
  <img src="media/teaser.png" alt="MOCHI teaser" width="100%">
  <br>
  <em>MOCHI predicts topologically consistent 3D face meshes in dense semantic correspondence directly from calibrated multi-view images, and a test-time optimization (MOCHI-TTO) pass further sharpens the geometry.</em>
</p>


---

## 1. Installation

The code targets **Python 3.10** and **CUDA 12.4**. Create a fresh environment and run the
installer:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash install.sh
```

`install.sh` pulls PyTorch 2.5.1 (cu124), pytorch3d 0.7.8, MPI-IS `mesh`, kaolin, kornia,
pyrender, trimesh, wandb, etc., and builds the vendored `liegroups` package under
`modules/liegroups`. It also downloads the **FLAME 2023** head model
(`flame2023_no_jaw.pkl`), which is license-restricted: register at
<https://flame.is.tue.mpg.de/> and agree to the license first — the script will prompt for
your FLAME username and password.


## 2. Data

MOCHI uses the **FaMoS** dataset, released as part of
[TEMPEH](https://tempeh.is.tue.mpg.de/). Follow the following steps to prepare the data.

**a) Download FaMoS.** Register at <https://tempeh.is.tue.mpg.de/> and agree to the license,
then use the (TEMPEH-provided) fetch scripts under [`famos_download/`](famos_download/) — see
[`famos_download/README.md`](famos_download/README.md) for details:

```bash
cd famos_download
bash fetch_test_subset.sh     # quick start: small paper test subset
bash fetch_training_data.sh   # full training set (images, scans, FLAME registrations)
bash fetch_test_data.sh       # full test set
cd ..
```

**b) Preprocess into multi-view grids.** Follow [`datasets/preprocess.md`](datasets/preprocess.md),
which uses [`datasets/build_grids.py`](datasets/build_grids.py) to render the
ground-truth normal/depth maps from the scans and pack them, alongside the multi-view RGB,
cameras, and dense landmarks, into the grid layout the trainer reads.

## 3. Training

Training also needs the **dense-landmark predictions** for FaMoS. These come from a
synthetic-data-trained landmark detector that is not shipped here, so we release the
precomputed predictions as two Drive archives:

```bash
mkdir -p famos_dense_landmarks && cd famos_dense_landmarks
gdown 19F8IdfmxZw4aXqSvvYlp7Z3R_Ek9vRvQ -O color_dense_landmarks.zip          # ~30 GB
gdown 1UOtmoTGXdFV4dP9TtmRUHEG9XACc2O_8 -O color_dense_semantic_landmarks.zip # ~1.2 GB
unzip color_dense_landmarks.zip
unzip color_dense_semantic_landmarks.zip
cd ..
```

This yields `famos_dense_landmarks/{color_dense_landmarks,color_dense_semantic_landmarks}/<subject>/<sequence>/<frame>/…`.
Point the trainer at them by editing
[`scripts/_data_paths.sh`](scripts/_data_paths.sh) (`--dense-landmarks-dir` /
`--dense-semantic-landmarks-dir`).

The model then trains in three sequential stages, with an optional fourth test-time-optimization
pass. After editing the data paths in `scripts/_data_paths.sh`, run the stage launchers in
order:

```bash
bash scripts/stage1_pretrain.sh   # coarse, no differentiable rendering
bash scripts/stage2_coarse.sh     # coarse + differentiable rendering   (needs stage 1)
bash scripts/stage3_local.sh      # local refinement                    (needs stage 2)
bash scripts/stage4_refine.sh     # optional per-scene TTO              (needs stage 3)
```

## 4. Pretrained models

We release the trained MOCHI checkpoints so you can run the test-time optimization (stage 4) —
or start local refinement (stage 3) — without retraining from scratch. Download them with
`gdown` into `pretrained_models/` (the default paths the stage scripts expect):

```bash
mkdir -p pretrained_models
gdown 1YFs_CUUyzwtjwrO-sbBZ3FWOXZdMjLGJ -O pretrained_models/global.pth   # coarse global model (stages 1-2)
gdown 1d0cCVz344RtKcB4X5DfverupZtHKumCS -O pretrained_models/local.pth    # local refinement model (stage 3)
```

`scripts/stage3_local.sh` and `scripts/stage4_refine.sh` default to
`pretrained_models/global.pth` and `pretrained_models/local.pth`; override them via the
`PRETRAINED_CKPT` / `PRETRAINED_LOCAL_CKPT` environment variables if you store the checkpoints
elsewhere.

## Acknowledgements

This work builds directly on **[TEMPEH](https://tempeh.is.tue.mpg.de/)** (MPI-IS, 2023); much
of the multi-view volumetric backbone and data tooling derives from it. We also use
**[FLAME](https://flame.is.tue.mpg.de/)** and **[pytorch3d](https://pytorch3d.org/)** /
**[kaolin](https://github.com/NVIDIAGameWorks/kaolin)** for differentiable rendering.

## License

See the [LICENSE](./LICENSE) file. This repository builds on the TEMPEH codebase; please respect
the upstream license terms at <https://tempeh.is.tue.mpg.de/license.html>.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{filntisis2026mochi,
    title     = {Registration-Free Learnable Multi-View Capture of Faces in Dense Semantic Correspondence},
    author    = {Filntisis, Panagiotis P. and Retsinas, George and Daněček, Radek and Sklyarova, Vanessa and Maragos, Petros and Bolkart, Timo},
    booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}
```
