# MoCHI: Multi-view Coarse-to-fine Head Inference

Official implementation accompanying the CVPR submission. This release reproduces
the four-stage training and test-time-optimization pipeline on FaMoS data.

---

## Repository layout

```
mochi_rebuttal_public/
├── trainer/                # entry points + training/refinement loops
│   ├── train_global.py     # CLI for stages 1–3 (calls trainer.global_trainer.run)
│   ├── train_refine.py     # CLI for stage 4 / TTO  (calls trainer.refiner.run)
│   ├── global_trainer.py   # main training class
│   ├── refiner.py          # test-time optimization class
│   └── base_trainer.py     # shared base class
├── option_handler/         # argparse-based CLI options
├── models/                 # FLAME, model_aligner, perceptual losses
├── modules/                # volumetric samplers, V2V networks, transformer,
│                           #   resnet feature backbone (uresnet2)
├── datasets/               # FaMoS multi-view dataset class
├── utils/                  # rendering, mesh, losses, cameras
├── assets/                 # FLAME models, masks, head template, data lists
├── scripts/                # one-line stage launchers
├── render_normals_undistorted_test.py   # FaMoS preprocessing script
├── install.sh              # environment install (Python 3.10 + CUDA 12.4)
├── requirements.txt
└── README.md
```

---

## 1. Environment setup

The code targets Python 3.10 and CUDA 12.4. Create a fresh virtual environment
and run the installer:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The installer pulls PyTorch 2.5.1 (cu124), pytorch3d 0.7.8, MPI-IS `mesh`,
`liegroups` (vendored under `modules/liegroups`), kaolin, kornia, pyrender,
trimesh, wandb, etc. The script clones MPI-IS `mesh` into `assets/software`
and builds it; you need a working compiler toolchain.

If you need a headless render (e.g. on a cluster compute node without an X
server), `pyrender` defaults to EGL when `PYOPENGL_PLATFORM=egl` is exported
before launch.

---

## 2. Asset download

The repo ships with these small project-specific assets under `assets/`:

* `assets/template/sampling_template.obj`, `vertex_masks2.npz` — 5023-vertex
  FLAME template + our region masks.
* `assets/head_template.obj`, `assets/landmark_embedding.npy`,
  `assets/{l,r}_eyelid.npy`, `assets/mediapipe_landmark_embedding/` — landmark
  embeddings used by FLAME and pliks.
* `assets/FLAME_masks/FLAME_masks_triangles.npy` — triangle-region masks.
* `assets/FaMoS_fiveteen_test_subjects.json`,
  `assets/fiveteen_subj__all_seq_frames_per_seq_10_test.json`,
  `assets/meshes_list_test.json` — example data lists for refine / evaluation.

The FLAME model files are **not** redistributed in this repo. Download them
yourself from <https://flame.is.tue.mpg.de/> after accepting the license,
then place them as follows:

```
assets/FLAME2023/flame2023.pkl           # FLAME 2023 generic
assets/FLAME2023/flame2023_no_jaw.pkl    # FLAME 2023 with jaw locked (used by default)
assets/FLAME_masks/FLAME_masks.pkl       # FLAME region masks
```

(`flame2023_no_jaw.pkl` is the model the trainer loads by default; see
`base_trainer.py` and `pliks_flame_2.py`.)

---

## 3. Data preparation

The training pipeline consumes pre-rendered, undistorted multi-view RGB,
normal-map and depth-map grids from the FaMoS dataset. See
[`datasets/preprocess.md`](datasets/preprocess.md) for the step-by-step
procedure (downloading FaMoS, running `render_normals_undistorted_test.py`,
expected output layout).

The trainer expects the following directory hierarchy after preprocessing:

```
<grid_root>/
├── color_images_v2/        # input RGB grids (one per view)
├── color_normals_numpy/    # ground-truth normal maps (npy)
├── color_depth/            # ground-truth depth maps
├── color_cameras/          # per-frame camera intrinsics / extrinsics
└── color_dense_landmarks/  # dense landmark predictions
```

---

## 4. Training stages

The pipeline trains in three sequential stages, with an optional fourth
test-time-optimization (TTO) pass. Convenience launchers live in
[`scripts/`](scripts/) — edit the data paths at the top of each script and
run:

```bash
bash scripts/stage1_pretrain.sh
bash scripts/stage2_coarse.sh    # depends on stage 1 checkpoint
bash scripts/stage3_local.sh     # depends on stage 2 checkpoint
bash scripts/stage4_refine.sh    # optional TTO; depends on stage 3 checkpoint
```

The exact CLI arguments reproduced by each script:

### Stage 1 — pretraining (no differentiable rendering)

```bash
python -m trainer.train_global \
  -eid restart/coarse_nodifrs \
  -print-freq 100 -val-freq 20000 -vis-freq 100 \
  -b 2 -irf 2 -wandb True --gradient-max-norm 1 \
  --input-image-type color_images --num-iterations 300000 \
  -wpointr 0 -wp2s 0 -wlandd 1 \
  -wshapereg 1e-3 -wexpreg 1e-3 \
  -wvertregpliks 1e-3 -wvertregpliks-edge 0.1 \
  -pliks True
```

### Stage 2 — coarse with differentiable rendering

```bash
python -m trainer.train_global \
  -eid restart/coarse_difrs \
  -print-freq 100 -val-freq 20000 -vis-freq 1000 \
  -b 2 -irf 2 -wandb True --gradient-max-norm 1 \
  --input-image-type color_images --num-iterations 300000 \
  -wpointr 0 -wp2s 0 -wlandd 0.5 \
  -wshapereg 1e-3 -wexpreg 1e-3 \
  -wvertregpliks 1e-3 -wvertregpliks-edge 0.1 \
  -pliks True \
  -difr True -wpointmaps 10 -wnorm 10 \
  --pretrained-path runs/coarse/restart/coarse_nodifrs/checkpoints/model_00309000.pth
```

### Stage 3 — local refinement

```bash
python -m trainer.train_global \
  -eid restart/local_wland005 \
  -print-freq 100 -val-freq 20000 -vis-freq 300 \
  -b 2 -irf 1 -wandb True --gradient-max-norm 1 \
  --input-image-type color_images --num-iterations 300000 \
  -wpointr 0.01 -wp2s 0 -wlandd 0.0 --weight-landmarks 0.05 \
  -wshapereg 1e-6 -wexpreg 1e-5 \
  -wvertregpliks 1e-5 -wvertregpliks-edge 0.1 \
  -difr True -wpointmaps 10 -wnorm 4 -wedge 0.01 \
  --enable-local True \
  --point-mask-weights '{"w_point_face":0.0,"w_point_ears":0.0,"w_point_eyeballs":1.0,"w_point_eye_region":0.0,"w_point_lips":0.0,"w_point_neck":0.0,"w_point_nostrils":0.0,"w_point_scalp":0.0,"w_point_boundary":0.0}' \
  --pretrained-path runs/coarse/restart/coarse_difrs/checkpoints/model_00309000.pth
```

### Stage 4 — test-time optimization (optional)

```bash
python -m trainer.train_refine \
  -eid tto/$(REF_START)_$(REF_END) \
  --gradient-max-norm 1 --input-image-type color_images \
  -b 1 -irf 1 -difr True -wpointmaps 10 -wnorm 4 \
  -wp2s 0 -wlandd 0 -wpointr 0.01 -wedge 0.01 \
  --weight-landmarks 0.5 \
  --enable-local True \
  -tdl <path-to-train-data-list>.json \
  -vdl assets/meshes_list_test.json \
  -refine-lr 1e-3 -refine-steps 105 \
  --refine-vis True -refine-vis-freq 1 -vis-freq 1 \
  --refine-start-index $(REF_START) --refine-end-index $(REF_END) \
  --pretrained-path runs/coarse/<your-stage3-run>/checkpoints/model_00300000.pth
```

---

## License

See [`LICENSE`](LICENSE). The repository builds on the TEMPEH codebase
(MPI-IS, 2023); please respect the upstream license terms.
