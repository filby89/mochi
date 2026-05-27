# FaMoS preprocessing

The training pipeline operates on **undistorted, sub-sampled multi-view image
grids** rather than the raw FaMoS captures. This document describes how to
generate the directory layout the training code expects.

## 1. Obtain the FaMoS dataset

Register at the [FaMoS / TEMPEH project page](https://tempeh.is.tue.mpg.de), agree to the
license, and download the data with the fetch scripts in
[`../famos_download/`](../famos_download/) (see `famos_download/README.md`).
After unpacking you should have:

```
<famos_root>/
├── downsampled_images_4_no_grid/
│   ├── downsampled_images_4/        # 4×-downsampled per-view RGB
│   └── calibrations/                # per-view camera calibrations
├── meshes_npz/                      # ground-truth scan meshes (.npz)
└── registrations/                   # FLAME registrations
```

You also need the dense-landmark predictions used as supervision; we use the
predictions produced by our companion landmark network (see
`assets/landmark_embedding.npy` for the FLAME embedding).

## 2. Generate the undistorted multi-view grids

`render_normals_undistorted_test.py` (top of the repo) loads each frame, runs
multi-view rendering of the ground-truth scan to produce normal-map and
depth-map grids, and writes the per-frame inputs in the layout the trainer
consumes.

Edit the hard-coded paths near the top of the file to match your machine:

```python
in_image_path        = "<famos_root>/downsampled_images_4_no_grid/downsampled_images_4"
in_calibration_path  = "<famos_root>/downsampled_images_4_no_grid/calibrations"
in_dense_path        = "<dense_landmark_predictions>"
in_registration_path = "<famos_root>/registrations"
in_meshes_path       = "<famos_root>/meshes_npz"
```

Then run:

```bash
python render_normals_undistorted_test.py <START> <END> <OUTPUT_ROOT> <UNDISTORT>
```

Arguments:

* `<START>`, `<END>` — inclusive/exclusive indices into
  `assets/meshes_list.json` (or another JSON list of frames). Useful for
  sharding across nodes.
* `<OUTPUT_ROOT>` — where the rendered grids are written. The trainer's
  `image-directory` etc. must point at sub-directories of this root.
* `<UNDISTORT>` — `1` to undistort the input RGB / re-derive intrinsics,
  `0` to skip undistortion.

The script writes:

```
<OUTPUT_ROOT>/
├── color_images/            # undistorted RGB grids
├── color_normals/           # rendered normal-map grids
├── color_depth/             # rendered depth-map grids (.npy)
├── color_cameras/           # undistorted intrinsics + extrinsics + centers
└── color_dense_landmarks/   # dense landmark predictions reprojected
```

## 3. Wire the trainer to your output

In each stage script under `scripts/`, set the data-related CLI flags to
your output paths:

```bash
-tdl  /path/to/your_train_data_list.json
-vdl  /path/to/your_val_data_list.json
--scan-directory      <famos_root>/meshes_npz
--processed-directory <famos_root>/registrations
--image-directory       <OUTPUT_ROOT>/color_images_v2
--normals-image-directory  <OUTPUT_ROOT>/color_normals_numpy
--depths-image-directory   <OUTPUT_ROOT>/color_depth
--calibration-directory    <OUTPUT_ROOT>/color_cameras
--dense-landmarks-dir      <OUTPUT_ROOT>/color_dense_landmarks
--dense-semantic-landmarks-dir <OUTPUT_ROOT>/color_dense_semantic_landmarks
```

## 4. Sanity check

Before launching a long training run, verify that the dataset can iterate one
batch:

```bash
python -c "
from datasets.face_align_dataset_mpi_grid import FaceAlignDatasetMPI
ds = FaceAlignDatasetMPI(
    data_list_fname='<your_train_data_list>.json',
    image_dir='<OUTPUT_ROOT>/color_images_v2',
    calibration_dir='<OUTPUT_ROOT>/color_cameras',
    scan_dir='<famos_root>/meshes_npz',
    registration_root_dir='<famos_root>/registrations',
    normals_dir='<OUTPUT_ROOT>/color_normals_numpy',
    depths_dir='<OUTPUT_ROOT>/color_depth',
    dense_landmarks_dir='<OUTPUT_ROOT>/color_dense_landmarks',
    dense_semantic_landmarks_dir='<OUTPUT_ROOT>/color_dense_semantic_landmarks',
    image_resize_factor=2,
    image_file_ext='png',
)
print('dataset size:', len(ds))
print('sample keys:', list(ds[0].keys())[:10])
"
```
