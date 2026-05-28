"""
Build the multi-view image / normal / depth grids that the MOCHI trainer reads.

For each frame in the data list this script:
  * loads the multi-view RGB and per-view calibrations,
  * reads the GT scan + FLAME registration,
  * renders the GT normal and depth maps by projecting the scan through each
    view's intrinsics + radial distortion (so the rendered maps line up with the
    raw RGB views the trainer consumes),
  * writes everything as multi-view grids under ``<out-root>/``:

      color_images/<subject>/<sequence>/<frame>/<sequence>.<frame>.png
      color_normals/<subject>/<sequence>/<frame>/<sequence>.<frame>.png
      color_depth/<subject>/<sequence>/<frame>/<sequence>.<frame>.npy
      color_cameras/<subject>/<sequence>/<frame>/<sequence>.<frame>_intrinsics.npz
      color_dense_landmarks/<subject>/<sequence>/<frame>/<sequence>.<frame>.npy

Run from the repo root, e.g.::

    python -m datasets.build_grids \\
        --data-list assets/meshes_list.json \\
        --image-dir <famos>/downsampled_images_4 \\
        --calibration-dir <famos>/calibrations \\
        --scan-dir <famos>/meshes_npz \\
        --registration-dir <famos>/registrations \\
        --dense-landmarks-dir <dense_landmark_predictions> \\
        --out-root <out-root>
"""

import argparse
import json
import os
import tempfile

import imageio
import numpy as np
import torch
from tqdm import tqdm

from datasets.face_align_dataset_mpi import FaceAlignDatasetMPI
from utils.mesh_helper import MeshHelper


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--data-list',           required=True, help='JSON list of [subject, sequence, frame] triples.')
    p.add_argument('--image-dir',           required=True, help='Per-view RGB root (subject/sequence/frame/<seq>.<frame>.<view>.png).')
    p.add_argument('--calibration-dir',     required=True, help='Per-view .tka calibrations root.')
    p.add_argument('--scan-dir',            required=True, help='Per-frame .npz scan meshes root.')
    p.add_argument('--registration-dir',    required=True, help='Per-frame .ply FLAME registrations root.')
    p.add_argument('--dense-landmarks-dir', required=True, help='Per-frame dense-landmark predictions root.')
    p.add_argument('--out-root',            required=True, help='Destination root for the multi-view grids.')
    p.add_argument('--image-resize-factor', type=int, default=4)
    p.add_argument('--start',               type=int, default=0,    help='Inclusive start index into the data list (for sharding).')
    p.add_argument('--end',                 type=int, default=None, help='Exclusive end index. Default: full list.')
    p.add_argument('--num-workers',         type=int, default=8)
    return p.parse_args()


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def make_out_dirs(out_root):
    dirs = {
        'rgb':              os.path.join(out_root, 'color_images'),
        'normals':          os.path.join(out_root, 'color_normals'),
        'depth':            os.path.join(out_root, 'color_depth'),
        'cameras':          os.path.join(out_root, 'color_cameras'),
        'dense_landmarks':  os.path.join(out_root, 'color_dense_landmarks'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def save_frame(dirs, subject, sequence, frame,
               color_grid, normals_grid, depth_grid,
               intrinsics, extrinsics, centers, radial_distortions,
               dense_landmarks):
    tag = f'{sequence}.{frame}'
    rel = os.path.join(subject, sequence, frame)
    for d in dirs.values():
        os.makedirs(os.path.join(d, rel), exist_ok=True)

    imageio.imwrite(os.path.join(dirs['rgb'],             rel, f'{tag}.png'), color_grid)
    imageio.imwrite(os.path.join(dirs['normals'],         rel, f'{tag}.png'), normals_grid)
    np.save(        os.path.join(dirs['depth'],           rel, f'{tag}.npy'), depth_grid)
    np.savez(       os.path.join(dirs['cameras'],         rel, f'{tag}_intrinsics.npz'),
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    centers=centers,
                    radial_distortions=radial_distortions)
    np.save(        os.path.join(dirs['dense_landmarks'], rel, f'{tag}.npy'), dense_landmarks)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.data_list) as f:
        items = json.load(f)
    end = args.end if args.end is not None else len(items)
    items = items[args.start:end]

    # The dataset expects a JSON file path, so write the requested shard to a
    # temp file rather than mutating the user's data list.
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tf:
        json.dump(items, tf)
        slice_path = tf.name

    print(f'Building grids for {len(items)} frames ({args.start}..{end}) -> {args.out_root}')

    try:
        dataset = FaceAlignDatasetMPI(
            data_list_fname=slice_path,
            image_dir=args.image_dir,
            calibration_dir=args.calibration_dir,
            scan_dir=args.scan_dir,
            registration_root_dir=args.registration_dir,
            dense_landmarks_dir=args.dense_landmarks_dir,
            image_resize_factor=args.image_resize_factor,
            image_file_ext='png',
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=False,
        )

        dirs = make_out_dirs(args.out_root)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for sample in tqdm(loader, desc='Building grids'):
            subject  = sample['subject'][0]
            sequence = sample['sequence'][0]
            frame    = sample['frame'][0]

            # ---- render GT normal + depth from the scan, in each view's frame ----
            vertices = sample['v_scan'][0].numpy()
            faces    = sample['f_scan'][0].numpy()
            mesh_helper = MeshHelper(num_vertices=vertices.shape[0], faces=faces)
            verts_gpu = torch.from_numpy(vertices).unsqueeze(0).float().to(device)

            K = sample['color_camera_intrinsics'][0].unsqueeze(0).to(device)
            E = sample['color_camera_extrinsics'][0].unsqueeze(0).to(device)
            D = sample['color_camera_distortions'][0].unsqueeze(0).to(device)
            centers_np = sample['color_camera_centers'][0].cpu().numpy()
            color_views = sample['color_images'][0]  # (V, 3, H, W)
            h, w = color_views.shape[-2:]

            render_out = mesh_helper.render_normals_and_depth(
                verts_gpu, K, E, radial_distortions=D,
                depth_rendering_height=h, depth_rendering_width=w,
                return_depth=True,
            )
            normals = render_out['normal_images']  # (V, H, W, 3) in [0, 1]
            depth = render_out['depth_images']     # (V, H, W, 1) raw depth

            # ---- stack views horizontally into a single grid per modality ----
            color_grid = np.hstack([
                (dataset.denormalize_image(color_views[i].cpu().numpy().transpose(1, 2, 0)) * 255).astype(np.uint8)
                for i in range(color_views.shape[0])
            ])
            normals_grid = np.hstack([
                (normals[i].cpu().numpy() * 255).astype(np.uint8) for i in range(normals.shape[0])
            ])
            depth_grid = np.hstack([
                depth[i].cpu().numpy() for i in range(depth.shape[0])
            ])

            save_frame(
                dirs, subject, sequence, frame,
                color_grid, normals_grid, depth_grid,
                intrinsics=K[0].cpu().numpy(),
                extrinsics=E[0].cpu().numpy(),
                centers=centers_np,
                radial_distortions=D[0].cpu().numpy(),
                dense_landmarks=sample['color_camera_dense_landmarks'][0].cpu().numpy(),
            )
    finally:
        if os.path.exists(slice_path):
            os.unlink(slice_path)


if __name__ == '__main__':
    main()
