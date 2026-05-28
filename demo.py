"""
MOCHI demo — forward-only inference.

Runs the trained coarse (global) + local models on calibrated multi-view images
and writes the predicted FLAME-topology mesh per frame. No ground-truth scans,
registrations or dense landmarks are needed (those are only used for training /
test-time optimization).

It consumes the *raw* FaMoS test subset directly (per-view images + .tka
calibrations) via the per-view dataset, mirroring the configuration used to build
the training grids — so no separate preprocessing step is required.

Example (after `cd famos_download && bash fetch_test_subset.sh && cd ..`):

    python demo.py \
        -local True \
        --pretrained-path pretrained_models/global.pth \
        --pretrained-local-path pretrained_models/local.pth \
        -tdl famos_download/data/test_data_subset/paper_test_frames.json \
        --image-directory famos_download/data/test_data_subset/test_subset_images_4 \
        --calibration-directory famos_download/data/test_data_subset/test_subset_calibrations \
        -eid demo

Predicted meshes are written to `runs/demo/demo_meshes/*.ply`.
"""

import os
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from option_handler.train_options_global import TrainOptions
from trainer.global_trainer import Trainer
from datasets.face_align_dataset_mpi import FaceAlignDatasetMPI

# The training grids were built from the 4x-downsampled per-view captures with
# this resize factor; the model consumes the raw views plus their distortion
# coefficients (no undistortion). We mirror that here so the demo input matches
# what the model was trained on.
PREPROCESS_RESIZE_FACTOR = 4


def run(config_fname=''):
    parser = TrainOptions()
    args = parser.parse(config_filename=config_fname)
    args.enable_local = True  # the demo always runs coarse + local

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Build the models exactly as training does and load the pretrained checkpoints
    # (-pretr-path / global.pth for coarse, -pretr-path-local / local.pth for local).
    trainer = Trainer(args, device)
    trainer.mkdirs()
    trainer.register_mesh_sampler()
    trainer.register_model()
    trainer.model.eval()
    trainer.local_model.eval()

    faces = trainer.faces.cpu().numpy()

    # Per-view dataset straight from the raw test subset (images + calibrations only).
    dataset = FaceAlignDatasetMPI(
        data_list_fname=args.train_data_list_fname,
        image_dir=args.image_directory,
        calibration_dir=args.calibration_directory,
        image_resize_factor=PREPROCESS_RESIZE_FACTOR,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    out_dir = os.path.join(trainer.directory_output, 'demo_meshes')
    os.makedirs(out_dir, exist_ok=True)

    for sample in tqdm(loader, desc='MOCHI demo'):
        subject, sequence, frame = sample['subject'][0], sample['sequence'][0], sample['frame'][0]

        images = sample['color_images'].to(device)                 # (B, V, 3, H, W)
        camera_intrinsics = sample['color_camera_intrinsics'].to(device)
        camera_extrinsics = sample['color_camera_extrinsics'].to(device)
        camera_distortions = sample['color_camera_distortions'].to(device)
        camera_centers = sample['color_camera_centers'].to(device)

        # Coarse stage runs on half-resolution images (mirrors trainer.feed_data).
        B, V, C, H, W = images.shape
        images_coarse = F.interpolate(
            images.view(B * V, C, H, W), scale_factor=0.5, mode='bilinear', align_corners=False
        ).view(B, V, C, H // 2, W // 2)
        intrinsics_coarse = camera_intrinsics.clone()
        intrinsics_coarse[..., :2, :] *= 0.5

        with torch.inference_mode():
            coarse = trainer.model(
                images=images_coarse, camera_intrinsics=intrinsics_coarse,
                camera_extrinsics=camera_extrinsics, camera_distortions=camera_distortions,
                random_grid=False,
            )
            coarse_points = coarse['vertices']

            results = trainer.local_model(
                images=images, camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics, camera_distortions=camera_distortions,
                camera_centers=camera_centers, global_points=coarse_points, random_grid=False,
            )
            vertices = results[-1]  # (B, 5023, 3), in millimetres

        for b in range(vertices.shape[0]):
            v = vertices[b].detach().cpu().numpy() * 0.001  # mm -> m
            out_fname = os.path.join(out_dir, f'{subject}_{sequence}_{frame}.ply')
            trimesh.Trimesh(vertices=v, faces=faces, process=False).export(out_fname)

    print('Done. Meshes written to', out_dir)


if __name__ == '__main__':
    run()
