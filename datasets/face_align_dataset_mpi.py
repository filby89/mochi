"""
Per-view multi-view FaMoS dataset used by:

  * `datasets.build_grids` (preprocessing: per-view RGB + cameras + scan/registration
    + dense landmarks -> multi-view grids that the trainer reads)
  * `demo.py`               (inference: per-view RGB + cameras -> mesh)

A single sample yields, for each `_C` (color) view of the requested
(subject, sequence, frame):

  - normalized RGB,
  - camera intrinsics, extrinsics, radial distortion, camera centre,

optionally (when the corresponding directory is provided):

  - the dense-landmark predictions per view,
  - the ground-truth scan (vertices + faces),
  - the FLAME registration (vertices + faces, converted to millimetres).

Notes
-----
- The model consumes the views *as captured*; no undistortion happens here. The
  intrinsics + radial distortions are passed through and applied later by the
  projection / rasterisation code.
- Image normalisation uses ImageNet statistics (matching the feature backbone).
"""

import glob
import os

import imageio
import numpy as np
import torch
import torch.utils.data as data
from psbody.mesh import Mesh
from skimage.transform import resize

from utils.camera import load_mpi_camera
from utils.utils import get_filename


# ImageNet RGB normalisation — matches the feature backbone.
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceAlignDatasetMPI(data.Dataset):
    """Per-view multi-view loader for FaMoS captures."""

    def __init__(
        self,
        data_list_fname,
        image_dir,
        calibration_dir,
        scan_dir='',
        registration_root_dir='',
        dense_landmarks_dir='',
        image_resize_factor=1,
        image_file_ext='png',
    ):
        super().__init__()

        if not os.path.exists(data_list_fname):
            raise RuntimeError(f'Invalid data list path: {data_list_fname}')
        # Plain JSON list of [subject, sequence, frame] triples.
        import json
        with open(data_list_fname) as f:
            self.split_list = json.load(f)

        if not os.path.exists(image_dir):
            raise RuntimeError(f'Invalid image directory: {image_dir}')
        if not os.path.exists(calibration_dir):
            raise RuntimeError(f'Invalid calibration directory: {calibration_dir}')

        self.image_resize_factor = image_resize_factor

        # ---- path templates ----------------------------------------------------
        self._img_fname = lambda subject, sequence, frame, view: os.path.join(
            image_dir, subject, sequence, frame, f'{sequence}.{frame}.{view}.{image_file_ext}')
        self._calib_dir = lambda subject, sequence: os.path.join(
            calibration_dir, subject, sequence)

        if dense_landmarks_dir and os.path.exists(dense_landmarks_dir):
            self._dense_landmarks_fname = lambda subject, sequence, frame, view: os.path.join(
                dense_landmarks_dir, subject, sequence, frame, f'{sequence}.{frame}.{view}.npy')
        else:
            self._dense_landmarks_fname = None

        if scan_dir and os.path.exists(scan_dir):
            self._scan_fname = lambda subject, sequence, frame: os.path.join(
                scan_dir, subject, sequence, f'{sequence}.{frame}.npz')
        else:
            self._scan_fname = None

        if registration_root_dir and os.path.exists(registration_root_dir):
            self._registration_fname = lambda subject, sequence, frame: os.path.join(
                registration_root_dir, subject, sequence, f'{sequence}.{frame}.ply')
        else:
            self._registration_fname = None

        # Tensor copies of the mean/std are handy for the tensor path of normalize.
        self._mean = torch.from_numpy(_RGB_MEAN)
        self._std  = torch.from_numpy(_RGB_STD)

    # ----------------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ----------------------------------------------------------------------------

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, index):
        return self.read(index % len(self.split_list))

    # ----------------------------------------------------------------------------
    # Sample assembly
    # ----------------------------------------------------------------------------

    def read(self, index):
        subject, sequence, frame = self.split_list[index]

        # FaMoS labels color cameras with a `_C` suffix in the calibration file
        # name (e.g. 23_C.tka). Other suffixes (e.g. stereo `_A`/`_B`) are not
        # used in this codebase.
        calib_fnames = sorted(glob.glob(os.path.join(self._calib_dir(subject, sequence), '*.tka')))
        calib_fnames = [f for f in calib_fnames if '_C' in get_filename(f)]
        if not calib_fnames:
            raise RuntimeError(f'No `_C` calibrations for {subject}/{sequence}/{frame}')

        images, intrinsics, extrinsics, distortions, centers = [], [], [], [], []
        dense_landmarks_per_view = []
        camera_names = []

        for calib_fname in calib_fnames:
            view = get_filename(calib_fname)
            view_data = self._load_view(subject, sequence, frame, view, calib_fname)
            if view_data is None:
                continue
            images.append(view_data['image'])
            intrinsics.append(view_data['intrinsics'])
            extrinsics.append(view_data['extrinsics'])
            distortions.append(view_data['radial_distortion'])
            centers.append(view_data['camera_center'])
            camera_names.append(view)
            if view_data.get('dense_landmarks') is not None:
                dense_landmarks_per_view.append(view_data['dense_landmarks'])

        sample = {
            'color_images':            torch.stack(images, dim=0),
            'color_camera_intrinsics': torch.stack(intrinsics, dim=0),
            'color_camera_extrinsics': torch.stack(extrinsics, dim=0),
            'color_camera_distortions': torch.stack(distortions, dim=0),
            'color_camera_centers':    torch.stack(centers, dim=0),
            'subject': subject,
            'sequence': sequence,
            'frame': frame,
            'camera_names': camera_names,
        }
        if dense_landmarks_per_view:
            sample['color_camera_dense_landmarks'] = torch.stack(dense_landmarks_per_view, dim=0)

        # Optional GT: scan (.npz with 'vertices' + 'faces').
        if self._scan_fname is not None:
            scan_fname = self._scan_fname(subject, sequence, frame)
            try:
                v_scan, f_scan = self._load_scan(scan_fname)
                sample['v_scan'] = torch.from_numpy(v_scan.astype(np.float32))
                sample['f_scan'] = torch.from_numpy(f_scan.astype(np.int64))
            except Exception as e:
                print(f'Unable to load scan {scan_fname}: {e}')

        # Optional GT: FLAME registration (.ply, in metres -> millimetres).
        if self._registration_fname is not None:
            reg_fname = self._registration_fname(subject, sequence, frame)
            if os.path.exists(reg_fname):
                try:
                    reg = Mesh(filename=reg_fname)
                    reg.v[:] *= 1000.0  # FLAME registrations are stored in metres.
                    sample['v_registration'] = torch.from_numpy(reg.v.astype(np.float32))
                    sample['f_registration'] = torch.from_numpy(reg.f.astype(np.int64))
                except Exception as e:
                    print(f'Unable to load registration {reg_fname}: {e}')

        return sample

    # ----------------------------------------------------------------------------
    # Per-view loading
    # ----------------------------------------------------------------------------

    def _load_view(self, subject, sequence, frame, view, calib_fname):
        """Load (image, camera) for a single view; returns a dict or None."""
        image_fname = self._img_fname(subject, sequence, frame, view)
        if not os.path.exists(image_fname):
            print(f'Image file not found - {image_fname}')
            return None
        try:
            image = imageio.imread(image_fname, pilmode='RGB')
        except Exception as e:
            raise RuntimeError(f'Error loading image - {image_fname}: {e}')

        camera = load_mpi_camera(calib_fname, self.image_resize_factor, to_meters=False)
        if camera is None:
            return None

        image = image.astype(np.float32) / 255.0
        target_h, target_w = camera['image_size'][0], camera['image_size'][1]
        if image.shape[0] != target_h or image.shape[1] != target_w:
            image = resize(image, (target_h, target_w), anti_aliasing=True)
        image = self.normalize_image(image)

        out = {
            'image':             torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).contiguous(),
            'intrinsics':        torch.from_numpy(camera['intrinsics'].astype(np.float32)),
            'extrinsics':        torch.from_numpy(camera['extrinsics'].astype(np.float32)),
            'radial_distortion': torch.from_numpy(camera['radial_distortion'].astype(np.float32)),
            'camera_center':     torch.from_numpy(camera['camera_center'].astype(np.float32)),
        }

        if self._dense_landmarks_fname is not None:
            dl_fname = self._dense_landmarks_fname(subject, sequence, frame, view)
            if os.path.exists(dl_fname):
                out['dense_landmarks'] = torch.from_numpy(np.load(dl_fname).astype(np.float32))
            else:
                print(f'No dense landmarks at: {dl_fname}')

        return out

    # ----------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------

    @staticmethod
    def _load_scan(scan_fname):
        if not os.path.exists(scan_fname):
            raise RuntimeError(f'Scan not found: {scan_fname}')
        ext = os.path.splitext(scan_fname)[1].lower()
        if ext == '.npz':
            scan = np.load(scan_fname)
            return scan['vertices'], scan['faces']
        if ext in ('.obj', '.ply'):
            scan = Mesh(filename=scan_fname)
            return scan.v, scan.f
        raise RuntimeError(f'Unknown scan file extension: {ext}')

    def normalize_image(self, image):
        """ImageNet-normalise an RGB image.

        Accepts either a (H, W, 3) NumPy array or a (B, 3, H, W) torch tensor.
        """
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            return (image - _RGB_MEAN.reshape(1, 1, 3)) / _RGB_STD.reshape(1, 1, 3)
        if torch.is_tensor(image):
            if image.ndim != 4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            mean = self._mean.view(1, 3, 1, 1).to(image.device)
            std  = self._std.view(1, 3, 1, 1).to(image.device)
            return (image - mean) / std
        raise RuntimeError(f'unrecognised image type {type(image)}')

    def denormalize_image(self, image):
        """Inverse of `normalize_image`."""
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            return image * _RGB_STD.reshape(1, 1, 3) + _RGB_MEAN.reshape(1, 1, 3)
        if torch.is_tensor(image):
            if image.ndim != 4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            mean = self._mean.view(1, 3, 1, 1).to(image.device)
            std  = self._std.view(1, 3, 1, 1).to(image.device)
            return image * std + mean
        raise RuntimeError(f'unrecognised image type {type(image)}')
