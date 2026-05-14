"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms 
in the LICENSE file included with this software distribution. 
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

import os
import imageio

import numpy as np
import torch
import torch.utils.data as data

from psbody.mesh import Mesh
from utils import utils
from utils.data_augment import get_random_crop_offsets, scale_crop
import cv2

class FaceAlignDatasetMPI(data.Dataset):
    def __init__(self, 
                data_list_fname,
                image_dir='',
                image_resize_factor = 1,
                calibration_dir='',
                scan_dir='',
                normals_dir='',
                registration_root_dir='',   
                depths_dir='',
                # data augmentation parameters
                scale_min=0.9, # random scaling
                scale_max=1.1,
                brightness_sigma=0.1 / 3.0, # random brightness perturbation  
                image_file_ext='png',
                dense_landmarks_dir='',
                dense_semantic_landmarks_dir='',
                to_meters=False
                ):
        super().__init__()
       
        if os.path.exists(data_list_fname):
            self.split_list = utils.load_json(data_list_fname)
        else:
            raise RuntimeError('Invalid data path - %s' % data_list_fname)

        self.to_meters = to_meters

        if to_meters:
            print('Scaling all data to meters')
            self.to_meters_scale_factor = 1000
        else:
            self.to_meters_scale_factor = 1

        # augmentation
        self.scale_min = scale_min # random scaling
        self.scale_max = scale_max
        self.brightness_sigma = brightness_sigma # random brightness perturbation

        if os.path.exists(image_dir):
            self.img_dir = lambda subject, sequence, frame : os.path.join(image_dir, subject, sequence, frame)
            self.img_fname_grid = lambda subject, sequence, frame : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s' % (sequence, frame, image_file_ext))
        else:
            raise RuntimeError('Invalid image directory')

        if os.path.exists(normals_dir):
            self.normals_img_dir = lambda subject, sequence, frame : os.path.join(normals_dir, subject, sequence, frame)
            self.normals_img_fname_grid = lambda subject, sequence, frame : os.path.join(self.normals_img_dir(subject, sequence, frame), '%s.%s.%s' % (sequence, frame, 'npy'))
        else:
            self.normals_img_dir = None
            self.normals_img_fname_grid = None

        if os.path.exists(depths_dir):
            self.depths_dir = lambda subject, sequence, frame : os.path.join(depths_dir, subject, sequence, frame)
            self.depths_fname = lambda subject, sequence, frame : os.path.join(self.depths_dir(subject, sequence, frame), '%s.%s.npy' % (sequence, frame))
        else:
            self.depths_dir = None
            self.depths_fname = None

        if os.path.exists(calibration_dir):
            self.calibration_numpy_dir = lambda subject, sequence, frame : os.path.join(calibration_dir, subject, sequence, frame)
            self.calibration_img_fname_grid = lambda subject, sequence, frame : os.path.join(self.calibration_numpy_dir(subject, sequence, frame), '%s.%s_intrinsics.npz' % (sequence, frame))
        else:
            raise RuntimeError('Invalid calibration directory')

        if os.path.exists(dense_landmarks_dir):
            self.dense_landmarks_dir = lambda subject, sequence, frame : os.path.join(dense_landmarks_dir, subject, sequence, frame)
            self.dense_landmarks_fname = lambda subject, sequence, frame : os.path.join(self.dense_landmarks_dir(subject, sequence, frame), '%s.%s.npy' % (sequence, frame))
        else:
            self.dense_landmarks_dir = None
            self.dense_landmarks_fname = None
        
        if os.path.exists(dense_semantic_landmarks_dir):
            self.dense_semantic_landmarks_dir = lambda subject, sequence, frame : os.path.join(dense_semantic_landmarks_dir, subject, sequence, frame)
            self.dense_semantic_landmarks_fname = lambda subject, sequence, frame : os.path.join(self.dense_semantic_landmarks_dir(subject, sequence, frame), '%s.%s.npz' % (sequence, frame))
        else:
            self.dense_semantic_landmarks_dir = None
            self.dense_semantic_landmarks_fname = None

        self.scan_fname = ''
        if os.path.exists(scan_dir):
            if "scan" in scan_dir.lower():
                self.scan_fname = lambda subject, sequence, frame : os.path.join(scan_dir, subject, sequence, '%s.%s.obj' % (sequence, frame))
            else:
                self.scan_fname = lambda subject, sequence, frame : os.path.join(scan_dir, subject, sequence, '%s.%s.npz' % (sequence, frame))
             
        self.registration_fname = ''
        if os.path.exists(registration_root_dir):
            self.registration_fname = lambda subject, sequence, frame : os.path.join(registration_root_dir, subject, sequence, '%s.%s.ply' % (sequence, frame))


        self.image_resize_factor = image_resize_factor

        self.data_size = len(self.split_list)

        # normalization
        # standard values from resnet:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202
        self.mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


        self.mean = torch.from_numpy(self.mean_np)
        self.std  = torch.from_numpy(self.std_np)


    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.read(index % self.data_size)

    def read(self, index):
        to_meters = self.to_meters
        subject, sequence, frame = self.split_list[index]

        color_images = []
        color_images_normals = []
        color_camera_intrinsics = []
        color_camera_extrinsics = []
        color_camera_distortions = []
        color_camera_centers = []
        color_images_augmented = []
        color_images_normals_augmented = []
        color_camera_intrinsics_augmented = []

        color_images_depth = []
        color_images_depth_augmented = []

        color_camera_dense_landmarks = []
        color_camera_dense_landmarks_augmented = []
        color_camera_dense_landmarks_masks = []
        color_camera_dense_landmarks_masks_augmented = []

        color_camera_dense_mediapipe_landmarks = []
        color_camera_dense_mediapipe_landmarks_augmented = []

        calib = np.load(self.calibration_img_fname_grid(subject, sequence, frame))
        
        frame_grid = imageio.imread(self.img_fname_grid(subject, sequence, frame), pilmode='RGB')
        normals_grid = np.load(self.normals_img_fname_grid(subject, sequence, frame)) if self.normals_img_fname_grid is not None else None

        if getattr(self, 'dense_landmarks_fname', None) is not None:
            try:
                dense_path = self.dense_landmarks_fname(subject, sequence, frame)
                dense_landmarks = np.load(dense_path) if os.path.exists(dense_path) else None
                dense_mask_all = torch.ones((len(calib['intrinsics']), 1), dtype=torch.long)
            except Exception as e:
                print('Error loading dense landmarks:', e)
                dense_landmarks = np.zeros((len(calib['intrinsics']), 5023, 2), dtype=np.float32)
                dense_mask_all = torch.zeros((len(calib['intrinsics']), 1), dtype=torch.long)
        else:
            dense_landmarks = None

        dense_mediapipe_landmarks = None
        if getattr(self, 'dense_semantic_landmarks_fname', None) is not None:
            try:
                dense_semantic_path = self.dense_semantic_landmarks_fname(subject, sequence, frame)
                dense_semantic_data = np.load(dense_semantic_path) if os.path.exists(dense_semantic_path) else None
                dense_mediapipe_landmarks = dense_semantic_data['mediapipe_landmarks']
            except Exception as e:
                print('Error loading dense semantic landmarks:', e)
                raise e

        depths = np.load(self.depths_fname(subject, sequence, frame)) if self.depths_fname is not None else None

        h_grid, w_grid = frame_grid.shape[:2]
        h_one, w_one = h_grid, w_grid // len(calib['intrinsics'])

        perturbation = None
        for i in range(len(calib['intrinsics'])):
            if depths is not None:
                depth_img = depths[:, i * w_one:(i + 1) * w_one]
            else:
                depth_img = None
            
            frame_img = frame_grid[:, i * w_one:(i + 1) * w_one, :]
            normals_img = normals_grid[:, i * w_one:(i + 1) * w_one, :] if normals_grid is not None else None
            calib_img = {
                'intrinsics': calib['intrinsics'][i],
                'extrinsics': calib['extrinsics'][i],
                'radial_distortion': calib['radial_distortions'][i],
                'centers': calib['centers'][i]
            }

            dense_landmarks_view = dense_landmarks[i] if dense_landmarks is not None else None
            dense_mediapipe_landmarks_view = dense_mediapipe_landmarks[i] if dense_mediapipe_landmarks is not None else None

            img_with_camera = self.augment_img_with_camera(
                subject,
                sequence,
                frame_img,
                normals_img,
                calib_img,
                to_meters=to_meters,
                perturbation=perturbation,
                depth_img=depth_img,
                dense_landmarks=dense_landmarks_view,
                dense_mediapipe_landmarks=dense_mediapipe_landmarks_view
            )
            perturbation = img_with_camera['perturbation'] 

            if img_with_camera is not None:
                color_images.append(img_with_camera['image'])
                color_camera_intrinsics.append(img_with_camera['intrinsics'])
                color_camera_extrinsics.append(img_with_camera['extrinsics'])
                color_camera_distortions.append(img_with_camera['radial_distortion'])
                color_camera_centers.append(img_with_camera['camera_center'])
                color_images_augmented.append(img_with_camera['image_augmented'])
                color_camera_intrinsics_augmented.append(img_with_camera['intrinsics_augmented'])

                if img_with_camera.get('dense_landmarks', None) is not None:
                    color_camera_dense_landmarks.append(img_with_camera['dense_landmarks'])
                    color_camera_dense_landmarks_augmented.append(img_with_camera['dense_landmarks_augmented'])
                    color_camera_dense_landmarks_masks.append(dense_mask_all[i])
                    color_camera_dense_landmarks_masks_augmented.append(dense_mask_all[i])

                    color_camera_dense_mediapipe_landmarks.append(img_with_camera['dense_mediapipe_landmarks'])
                    color_camera_dense_mediapipe_landmarks_augmented.append(img_with_camera['dense_mediapipe_landmarks_augmented'])

                if img_with_camera['normals_image'] is not None:
                    color_images_normals.append(img_with_camera['normals_image'])
                    color_images_normals_augmented.append(img_with_camera['normals_image_augmented'])

                if img_with_camera['depth_map'] is not None:
                    color_images_depth.append(img_with_camera['depth_map'])
                    color_images_depth_augmented.append(img_with_camera['depth_map_augmented'])

        if len(color_images) > 0:
            color_images = torch.stack(color_images, dim=0)
            color_camera_intrinsics = torch.stack(color_camera_intrinsics, dim=0)
            color_camera_extrinsics = torch.stack(color_camera_extrinsics, dim=0)

            color_camera_extrinsics[:, :3, 3] /= self.to_meters_scale_factor

            color_camera_distortions = torch.stack(color_camera_distortions, dim=0)
            color_camera_centers = torch.stack(color_camera_centers, dim=0)
            color_images_augmented = torch.stack(color_images_augmented, dim=0)
            color_camera_intrinsics_augmented = torch.stack(color_camera_intrinsics_augmented, dim=0)

            if len(color_camera_dense_landmarks) > 0:
                color_images_normals = torch.stack(color_images_normals, dim=0) if len(color_images_normals) > 0 else None
                color_images_normals_augmented = torch.stack(color_images_normals_augmented, dim=0) if len(color_images_normals_augmented) > 0 else None

            if len(color_images_depth) > 0:
                color_images_depth = torch.stack(color_images_depth, dim=0) if len(color_images_depth) > 0 else None
                color_images_depth_augmented = torch.stack(color_images_depth_augmented, dim=0) if len(color_images_depth_augmented) > 0 else None

            if len(color_camera_dense_landmarks) > 0:
                color_camera_dense_landmarks = torch.stack(color_camera_dense_landmarks, dim=0) if len(color_camera_dense_landmarks) > 0 else None
                color_camera_dense_landmarks_augmented = torch.stack(color_camera_dense_landmarks_augmented, dim=0) if len(color_camera_dense_landmarks_augmented) > 0 else None
                color_camera_dense_landmarks_masks = torch.stack(color_camera_dense_landmarks_masks, dim=0) if len(color_camera_dense_landmarks_masks) > 0 else None
                color_camera_dense_landmarks_masks_augmented = torch.stack(color_camera_dense_landmarks_masks_augmented, dim=0) if len(color_camera_dense_landmarks_masks_augmented) > 0 else None

                color_camera_dense_mediapipe_landmarks = torch.stack(color_camera_dense_mediapipe_landmarks, dim=0) if len(color_camera_dense_mediapipe_landmarks) > 0 else None
                color_camera_dense_mediapipe_landmarks_augmented = torch.stack(color_camera_dense_mediapipe_landmarks_augmented, dim=0) if len(color_camera_dense_mediapipe_landmarks_augmented) > 0 else None

        data = {
            # img
            'color_images': color_images,
            'color_images_augmented': color_images_augmented,

            # normals
            'color_images_normals': color_images_normals,
            'color_images_normals_augmented': color_images_normals_augmented,

            # camera
            'color_camera_intrinsics': color_camera_intrinsics,
            'color_camera_extrinsics': color_camera_extrinsics,
            'color_camera_distortions': color_camera_distortions,
            'color_camera_centers': color_camera_centers,
            
            'color_camera_intrinsics_augmented': color_camera_intrinsics_augmented,

            'color_camera_dense_landmarks': color_camera_dense_landmarks,
            'color_camera_dense_landmarks_augmented': color_camera_dense_landmarks_augmented,

            'color_camera_dense_landmarks_masks': color_camera_dense_landmarks_masks,

            'color_images_depth': color_images_depth,
            'color_images_depth_augmented': color_images_depth_augmented,

            'color_camera_dense_landmarks_masks_augmented': color_camera_dense_landmarks_masks_augmented,
            'color_camera_dense_mediapipe_landmarks': color_camera_dense_mediapipe_landmarks,
            'color_camera_dense_mediapipe_landmarks_augmented': color_camera_dense_mediapipe_landmarks_augmented,


            # meta
            'index': index,
            'subject': subject,
            'sequence': sequence,
            'frame': frame,   
        }

        # remove all None values
        data = {k: v for k, v in data.items() if v is not None}

        if self.scan_fname != '':
            scan_fname = self.scan_fname(subject, sequence, frame)
            try:
                data['v_scan'], data['f_scan'] = self.load_scan(scan_fname)
                data['v_scan'] = torch.from_numpy(data['v_scan'].astype(np.float32))
                if self.to_meters:
                    data['v_scan'] /= 1000
                data['f_scan'] = torch.from_numpy(data['f_scan'].astype(np.int64))
            except Exception as e:
                print(f'Unable to load scan {scan_fname}: {e}')

        # Load registration
        if self.registration_fname != '':
            registration_fname = self.registration_fname(subject, sequence, frame)
        
            if os.path.exists(registration_fname):
                data['registration_fname'] = registration_fname
                try:
                    registration = Mesh(filename=registration_fname)
                    if not self.to_meters:
                        registration.v[:] *= 1000 # FLAME registrations are in meters, if to_meters is false, convert them to milimeters
                except:
                    print(f'Unable to load registration {registration_fname}')

                v_registration, f_registration = registration.v, registration.f
                data['v_registration'] = torch.from_numpy(v_registration.astype(np.float32))
                data['f_registration'] = torch.from_numpy(f_registration.astype(np.int64))

                data['v_reg_sampled'] = torch.from_numpy(v_registration.astype(np.float32))
                data['f_reg_sampled'] = torch.from_numpy(f_registration.astype(np.int64))

        if 'v_reg_sampled' in data:
            data['v_reg_global'] = data['v_reg_sampled']
            data['f_reg_global'] = data['f_reg_sampled']
        elif 'v_registration' in data:
            data['v_reg_global'] = data['v_registration']
            data['f_reg_global'] = data['f_registration']




        return data


    def load_scan(self, scan_fname):
        if not os.path.exists(scan_fname):
            raise RuntimeError(f'Scan not found {scan_fname}')

        file_extension = utils.get_extension(scan_fname)
        if file_extension.lower() in ['.obj', '.ply']:
            try:
                scan = Mesh(filename=scan_fname)
            except:
                raise RuntimeError(f'Unable to load scan {scan_fname}')

            return scan.v, scan.f
        elif file_extension.lower() in ['.npz']:
            try:
                scan = np.load(scan_fname)
            except:
                raise RuntimeError(f'Unable to load scan {scan_fname}')

            return scan['vertices'], scan['faces']

    def augment_img_with_camera(
        self,
        subject,
        sequence,
        image,
        normals_image,
        calib,
        to_meters=False,
        perturbation=None,
        depth_img=None,
        dense_landmarks=None,
        dense_mediapipe_landmarks=None,
    ):
        # ---- inputs to float [0,1] for geometric ops ----
        image = image.astype(np.float32) / 255.0
        normals_image = normals_image.astype(np.float32) if normals_image is not None else None
        h, w = image.shape[:2]

        camera = {
            'intrinsics': calib['intrinsics'],
            'extrinsics': calib['extrinsics'],
            'radial_distortion': calib['radial_distortion'],
            'camera_center': calib['centers'],
        }
        np.random.seed()
        crop_size = (h, w)  # full-res crop window
        scale_factor = self.scale_min + (self.scale_max - self.scale_min) * np.random.random()
        h_offset, w_offset = get_random_crop_offsets(crop_size, height=h, width=w)

        normals_image = normals_image * 2.0 - 1.0 if normals_image is not None else None # bring to [-1,1] for warp ops
        # print("WAHG!")

        sc = scale_crop(
            image, crop_size, h_offset, w_offset, scale_factor,
            K=camera['intrinsics'], normals_image=normals_image, debug=False,
            debug_root='debug_/', landmarks=None, depth_map=depth_img
        )
        image_augmented = sc['image']
        intrinsics_augmented = sc['K']
        normals_augmented = sc['normals_image']
        depth_augmented = sc['depth_map']

        dense_landmarks_augmented = None
        if dense_landmarks is not None:
            sc_dense = scale_crop(
                image, crop_size, h_offset, w_offset, scale_factor,
                K=None, normals_image=None, debug=False, debug_root=None,
                landmarks=dense_landmarks, depth_map=None
            )
            dense_landmarks_augmented = sc_dense['landmarks']

        dense_mediapipe_landmarks_augmented = None
        if dense_mediapipe_landmarks is not None:
            sc_dense_mp = scale_crop(
                image, crop_size, h_offset, w_offset, scale_factor,
                K=None, normals_image=None, debug=False, debug_root=None,
                landmarks=dense_mediapipe_landmarks, depth_map=None
            )
            dense_mediapipe_landmarks_augmented = sc_dense_mp['landmarks']

        # brightness jitter AFTER geometry
        if perturbation is None:
            perturb = 1.0 + self.brightness_sigma * np.random.randn(1, 1, 3)
        else:
            perturb = perturbation
        image_augmented = np.clip(image_augmented * perturb, 0.0, 1.0)

        # normalize RGB to your stats
        image_norm = self.normalize_image(image)
        image_aug_norm = self.normalize_image(image_augmented)

        # optional downscale when NOT using fixed-size path
        if self.image_resize_factor != 1:
            image_norm = cv2.resize(image_norm, (image_norm.shape[1]//self.image_resize_factor,
                                                image_norm.shape[0]//self.image_resize_factor), interpolation=cv2.INTER_LINEAR)
            image_aug_norm = cv2.resize(image_aug_norm, (image_aug_norm.shape[1]//self.image_resize_factor,
                                                        image_aug_norm.shape[0]//self.image_resize_factor), interpolation=cv2.INTER_LINEAR)
            if normals_image is not None:
                normals_image = cv2.resize(normals_image, (normals_image.shape[1]//self.image_resize_factor,
                                                        normals_image.shape[0]//self.image_resize_factor), interpolation=cv2.INTER_LINEAR)
            if normals_augmented is not None:
                normals_augmented = cv2.resize(normals_augmented, (normals_augmented.shape[1]//self.image_resize_factor,
                                                                normals_augmented.shape[0]//self.image_resize_factor), interpolation=cv2.INTER_LINEAR)
            if depth_img is not None:
                depth_img = cv2.resize(depth_img, (depth_img.shape[1]//self.image_resize_factor,
                                                depth_img.shape[0]//self.image_resize_factor), interpolation=cv2.INTER_NEAREST)
                depth_img = depth_img[:, :, None]
            if depth_augmented is not None:
                depth_augmented = cv2.resize(depth_augmented, (depth_augmented.shape[1]//self.image_resize_factor,
                                                            depth_augmented.shape[0]//self.image_resize_factor), interpolation=cv2.INTER_NEAREST)
                depth_augmented = depth_augmented[:, :, None]

            # adjust intrinsics
            camera['intrinsics'][0, :] /= self.image_resize_factor
            camera['intrinsics'][1, :] /= self.image_resize_factor
            intrinsics_augmented /= self.image_resize_factor

            if dense_landmarks is not None:
                dense_landmarks[:, 0] /= self.image_resize_factor
                dense_landmarks[:, 1] /= self.image_resize_factor
            if dense_landmarks_augmented is not None:
                dense_landmarks_augmented[:, 0] /= self.image_resize_factor
                dense_landmarks_augmented[:, 1] /= self.image_resize_factor

            if dense_mediapipe_landmarks is not None:
                dense_mediapipe_landmarks[:, 0] /= self.image_resize_factor
                dense_mediapipe_landmarks[:, 1] /= self.image_resize_factor
            if dense_mediapipe_landmarks_augmented is not None:
                dense_mediapipe_landmarks_augmented[:, 0] /= self.image_resize_factor
                dense_mediapipe_landmarks_augmented[:, 1] /= self.image_resize_factor

        # tensors
        image_t = torch.from_numpy(image_norm.astype(np.float32)).permute(2, 0, 1).contiguous()
        intrinsics_t = torch.from_numpy(camera['intrinsics'].astype(np.float32))
        extrinsics_t = torch.from_numpy(camera['extrinsics'].astype(np.float32))
        radial_t = torch.from_numpy(camera['radial_distortion'].astype(np.float32))
        center_t = torch.from_numpy(camera['camera_center'].astype(np.float32))

        dense_landmarks_t = torch.from_numpy(dense_landmarks.astype(np.float32)) if dense_landmarks is not None else None
        dense_landmarks_aug_t = torch.from_numpy(dense_landmarks_augmented.astype(np.float32)) if dense_landmarks_augmented is not None else dense_landmarks_t
        dense_mediapipe_landmarks_t = torch.from_numpy(dense_mediapipe_landmarks.astype(np.float32)) if dense_mediapipe_landmarks is not None else None
        dense_mediapipe_landmarks_aug_t = torch.from_numpy(dense_mediapipe_landmarks_augmented.astype(np.float32)) if dense_mediapipe_landmarks_augmented is not None else dense_mediapipe_landmarks_t

        normals_t = torch.from_numpy(normals_image.astype(np.float32)).permute(2, 0, 1).contiguous() if normals_image is not None else None

        image_aug_t = torch.from_numpy(image_aug_norm.astype(np.float32)).permute(2, 0, 1).contiguous()
        intrinsics_aug_t = torch.from_numpy(intrinsics_augmented.astype(np.float32))

        normals_aug_t = torch.from_numpy(normals_augmented.astype(np.float32)).permute(2, 0, 1).contiguous() if normals_augmented is not None else None

        depth_aug_t = torch.from_numpy(depth_augmented.astype(np.float32)) if depth_augmented is not None else None
        depth_img_t = torch.from_numpy(depth_img.astype(np.float32)) if depth_img is not None else None
        

        out = {
            'image': image_t,
            'intrinsics': intrinsics_t,
            'extrinsics': extrinsics_t,
            'radial_distortion': radial_t,
            'camera_center': center_t,

            'image_augmented': image_aug_t,
            'intrinsics_augmented': intrinsics_aug_t,

            'normals_image': normals_t,
            'normals_image_augmented': normals_aug_t,

            'perturbation': perturb,
            'depth_map_augmented': depth_aug_t,
            'depth_map': depth_img_t,
        }

        if dense_landmarks_t is not None:
            out.update({
                'dense_landmarks': dense_landmarks_t,
                'dense_landmarks_augmented': dense_landmarks_aug_t,
            })

        if dense_mediapipe_landmarks_t is not None:
            out.update({
                'dense_mediapipe_landmarks': dense_mediapipe_landmarks_t,
                'dense_mediapipe_landmarks_augmented': dense_mediapipe_landmarks_aug_t,
            })

        return out

    # -----------------------
    # normalize input

    def normalize_image(self, image):
        # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
        if isinstance(image, np.ndarray):
            if image.ndim !=3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return ( image - self.mean_np.reshape((1,1,3)) ) / self.std_np.reshape((1,1,3))
        elif torch.is_tensor(image):
            if image.ndimension() !=4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return ( image - self.mean.view(1,3,1,1).to(image.device) ) / self.std.view(1,3,1,1).to(image.device)
        else:
            raise RuntimeError(f"unrecognizable image type {type(image)}")

    def denormalize_image(self, image):
        # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
        if isinstance(image, np.ndarray):
            if image.ndim !=3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return image * self.std_np.reshape((1,1,3)) + self.mean_np.reshape((1,1,3))
        elif torch.is_tensor(image):
            if image.ndimension() !=4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return image * self.std.view(1,3,1,1).to(image.device) + self.mean.view(1,3,1,1).to(image.device)
        else:
            raise RuntimeError(f"unrecognizable image type {type(image)}")
