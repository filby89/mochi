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

## 2. Generate the multi-view grids

`build_grids.py` (in this `datasets/` folder) loads each frame, renders the
ground-truth normal and depth maps from the scan through each view's
intrinsics + radial distortion, and packs everything into the multi-view grid
layout the trainer reads.

Run it from the repo root:

```bash
python -m datasets.build_grids \
    --data-list assets/meshes_list.json \
    --image-dir            <famos_root>/downsampled_images_4_no_grid/downsampled_images_4 \
    --calibration-dir      <famos_root>/downsampled_images_4_no_grid/calibrations \
    --scan-dir             <famos_root>/meshes_npz \
    --registration-dir     <famos_root>/registrations \
    --dense-landmarks-dir  <dense_landmark_predictions> \
    --out-root             <OUTPUT_ROOT>
```

Use `--start <i>` and `--end <j>` to process a slice of the data list (handy
for sharding across nodes); see `python -m datasets.build_grids --help` for
every flag.

The script writes:

```
<OUTPUT_ROOT>/
├── color_images/            # multi-view RGB grids
├── color_normals/           # rendered normal-map grids
├── color_depth/             # rendered depth-map grids (.npy)
├── color_cameras/           # per-view intrinsics + extrinsics + centers + radial distortions
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
--image-directory       <OUTPUT_ROOT>/color_images
--normals-image-directory  <OUTPUT_ROOT>/color_normals
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
    image_dir='<OUTPUT_ROOT>/color_images',
    calibration_dir='<OUTPUT_ROOT>/color_cameras',
    scan_dir='<famos_root>/meshes_npz',
    registration_root_dir='<famos_root>/registrations',
    normals_dir='<OUTPUT_ROOT>/color_normals',
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
