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
import math
import glob
import random
import imageio
from skimage.transform import rescale, resize

import numpy as np
import torch
import torch.utils.data as data

from psbody.mesh import Mesh
from utils import mesh_sampling, utils
from utils.camera import load_mpi_camera, rotate_image
from utils.data_augment import get_random_crop_offsets, scale_crop, pad_img_and_intrinsics
from utils.utils import get_filename
import cv2

DISABLE_AUGM = False

PAD_TO_VIT = False

class FaceAlignDatasetMPI(data.Dataset):
    def __init__(self, 
                data_list_fname,
                dataset_root_dir='',
                image_dir='',
                image_resize_factor = 1,
                pad_to_vit=False,
                pad_multiple=14,
                output_image_height=None,     # ← NEW
                output_image_width=None,      # ← NEW
                calibration_dir='',
                scan_dir='',
                normals_dir='',
                registration_root_dir='',   
                global_registration_root_dir='',
                depths_dir='',
                mesh_sampler=None,
                # data augmentation parameters
                scale_min=0.9, # random scaling
                scale_max=1.1,
                brightness_sigma=0.1 / 3.0, # random brightness perturbation  
                scan_vertex_count=10000,
                # parameters to specify the type of images being loaded
                load_stereo_images=True,
                load_color_images=False,
                calibration_blacklist=[],
                image_file_ext='png',
                fan_landmarks_dir='',
                mediapipe_landmarks_dir='',
                dense_landmarks_dir='',
                dense_semantic_landmarks_dir='',
                return_full_scan=False,
                undistort_images=False,
                undistorted_K_dir='',
                # MAE masking & view-dropout parameters
                masking_ratio=0.0,    # fraction of pixels to mask in augmented image
                view_drop_prob=1.0,     # probability to drop an entire view
                to_meters=False,
                camera_names=None
                ):
        super().__init__()

       
        if os.path.exists(data_list_fname):
            self.split_list = utils.load_json(data_list_fname)
        else:
            raise RuntimeError('Invalid data path - %s' % data_list_fname)
        print()
        # f = 0
        # for i in self.split_list:
        #     if i[1] == 'mouth_side':
        #         break
        #     f += 1
        # self.split_list = self.split_list[f:f+1]
        self.load_stereo_images = load_stereo_images
        self.load_color_images = load_color_images
        self.calibration_blacklist = calibration_blacklist
        self.return_full_scan = return_full_scan

        self.pad_to_vit = bool(pad_to_vit)
        self.pad_multiple = int(pad_multiple) if pad_multiple is not None else 14

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

        self.mesh_sampler = mesh_sampler
        self.scan_vertex_count = scan_vertex_count
        self.registration_root_dir = registration_root_dir
        self.undistort_images = undistort_images

        self.masking_ratio = masking_ratio
        self.view_drop_prob = view_drop_prob
        self._camera_names_override = list(camera_names) if camera_names is not None else None

        if os.path.exists(image_dir):
            self.img_dir = lambda subject, sequence, frame : os.path.join(image_dir, subject, sequence, frame)
            # self.img_fname = lambda subject, sequence, frame, view : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s.%s' % (sequence, frame, view, image_file_ext))
            self.img_fname_grid = lambda subject, sequence, frame : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s' % (sequence, frame, image_file_ext))

        if os.path.exists(normals_dir):
            self.normals_img_dir = lambda subject, sequence, frame : os.path.join(normals_dir, subject, sequence, frame)
            # self.normals_img_fname = lambda subject, sequence, frame, view : os.path.join(self.normals_img_dir(subject, sequence, frame), '%s.%s.%s.%s' % (sequence, frame, view, image_file_ext))
            self.normals_img_fname_grid = lambda subject, sequence, frame : os.path.join(self.normals_img_dir(subject, sequence, frame), '%s.%s.%s' % (sequence, frame, 'npy'))
        else:
            self.normals_img_dir = None
            self.normals_img_fname_grid = None
            # raise RuntimeError('Invalid normals directory')
        print(self.normals_img_dir, self.normals_img_fname_grid)

        # print('Depths directory:', depths_dir)
        if os.path.exists(depths_dir):
            self.depths_dir = lambda subject, sequence, frame : os.path.join(depths_dir, subject, sequence, frame)
            self.depths_fname = lambda subject, sequence, frame : os.path.join(self.depths_dir(subject, sequence, frame), '%s.%s.npy' % (sequence, frame))
        else:
            self.depths_dir = None
            self.depths_fname = None
            # raise RuntimeError('Invalid depths directory')

        if os.path.exists(calibration_dir):
            self.calibration_numpy_dir = lambda subject, sequence, frame : os.path.join(calibration_dir, subject, sequence, frame)
            self.calibration_img_fname_grid = lambda subject, sequence, frame : os.path.join(self.calibration_numpy_dir(subject, sequence, frame), '%s.%s_intrinsics.npz' % (sequence, frame))
        else:
            raise RuntimeError('Invalid calibration directory')

        if os.path.exists(fan_landmarks_dir):
            self.fan_landmarks_dir = lambda subject, sequence, frame : os.path.join(fan_landmarks_dir, subject, sequence, frame)
            self.fan_landmarks_fname = lambda subject, sequence, frame : os.path.join(self.fan_landmarks_dir(subject, sequence, frame), '%s.%s.npy' % (sequence, frame))
            ff = '/fast/pfilntisis/TEMPEH_data/data/training_data/downsampled_images_4_fan_landmarks_hhj1897'

            self.fan_landmarks_orig_fnam = lambda subject, sequence, frame, view : os.path.join(ff, subject, sequence, frame, '%s.%s.%s.npz' % (sequence, frame, view))
        else:
            self.fan_landmarks_dir = None
            self.fan_landmarks_fname = None
            self.fan_landmarks_orig_fnam = None

        # Optional MediaPipe landmarks
        if os.path.exists(mediapipe_landmarks_dir):
            self.mediapipe_landmarks_dir = lambda subject, sequence, frame : os.path.join(mediapipe_landmarks_dir, subject, sequence, frame)
            self.mediapipe_landmarks_fname = lambda subject, sequence, frame : os.path.join(self.mediapipe_landmarks_dir(subject, sequence, frame), '%s.%s.npy' % (sequence, frame))
            nn = '/fast/pfilntisis/TEMPEH_data/data/training_data/downsampled_images_4_mediapipe_landmarks_fp32'
            self.mediapipe_landmarks_orig_fnam = lambda subject, sequence, frame, view : os.path.join(nn, subject, sequence, frame, '%s.%s.%s.npz' % (sequence, frame, view))
        else:
            self.mediapipe_landmarks_dir = None
            self.mediapipe_landmarks_fname = None
            self.mediapipe_landmarks_orig_fnam = None

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
            if return_full_scan:
                if "scan" in scan_dir.lower():
                    self.scan_fname = lambda subject, sequence, frame : os.path.join(scan_dir, subject, sequence, '%s.%s.obj' % (sequence, frame))
                else:
                    self.scan_fname = lambda subject, sequence, frame : os.path.join(scan_dir, subject, sequence, '%s.%s.npz' % (sequence, frame))
            else:
                self.scan_fname = lambda subject, sequence, frame : os.path.join(scan_dir, subject, sequence, '%s.%s.npy' % (sequence, frame))
        elif os.path.exists(dataset_root_dir):
            self.scan_fname = lambda subject, sequence, frame : os.path.join(dataset_root_dir, subject, sequence, 'meshes', '%s.%s.obj' % (sequence, frame))
             
        self.registration_fname = ''
        if os.path.exists(registration_root_dir):
            self.registration_fname = lambda subject, sequence, frame : os.path.join(registration_root_dir, subject, sequence, '%s.%s.ply' % (sequence, frame))

        self.global_mesh_fname = ''
        if os.path.exists(global_registration_root_dir):
            self.global_mesh_fname = lambda subject, sequence, frame : os.path.join(global_registration_root_dir, subject, sequence, '%s.%s.ply' % (sequence, frame))


        self.image_resize_factor = image_resize_factor
        self.output_image_height = output_image_height
        self.output_image_width  = output_image_width

        self.data_size = len(self.split_list)

        # normalization
        # standard values from resnet:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202
        self.mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # self.mean_np = np.array([0, 0, 0], dtype=np.float32)
        # self.std_np  = np.array([1, 1, 1], dtype=np.float32)

        self.mean = torch.from_numpy(self.mean_np)
        self.std  = torch.from_numpy(self.std_np)


        # find unique subjects

        self.subjects = set()
        for split in self.split_list:
            subject, sequence, frame = split
            self.subjects.add(subject)
        self.subjects = sorted(list(self.subjects))
        print('Subjects', self.subjects)
        print('Found %d subjects in the dataset' % len(self.subjects))
            

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.read(index % self.data_size)

    def read(self, index):
        to_meters = self.to_meters
        subject, sequence, frame = self.split_list[index]

        # # Read calibration files
        # # print(self.calibration_dir(subject, sequence))
        # calib_fnames = sorted(glob.glob(os.path.join(self.calibration_dir(subject, sequence), '*.tka')))
        # # print('Num calib fnames: %d' % len(calib_fnames))
        # # print(calib_fnames)

        color_images = []
        color_images_normals = []
        color_camera_intrinsics = []
        color_camera_extrinsics = []
        color_camera_distortions = []
        color_camera_centers = []
        color_images_augmented = []
        color_images_normals_augmented = []
        color_camera_intrinsics_augmented = []

        stereo_images = []
        stereo_images_normals = []
        stereo_camera_intrinsics = []
        stereo_camera_extrinsics = []
        stereo_camera_distortions = []
        stereo_camera_centers = []
        stereo_images_augmented = []
        stereo_images_normals_augmented = []
        stereo_camera_intrinsics_augmented = []

        color_camera_landmarks = []
        stereo_camera_landmarks = []
        color_camera_landmarks_augmented = []
        stereo_camera_landmarks_augmented = []

        stereo_camera_landmarks_masks = []

        color_images_depth = []
        color_images_depth_augmented = []

        color_camera_landmarks_masks = []
        color_camera_landmarks_masks_augmented = []

        color_camera_landmarks_hull_masks = []
        color_camera_landmarks_hull_masks_augmented = []

        color_camera_dense_landmarks = []
        color_camera_dense_landmarks_augmented = []
        color_camera_dense_landmarks_masks = []
        color_camera_dense_landmarks_masks_augmented = []

        color_camera_dense_fan_landmarks = []
        color_camera_dense_fan_landmarks_augmented = []
        color_camera_dense_mediapipe_landmarks = []
        color_camera_dense_mediapipe_landmarks_augmented = []

        camera_names =[]

        if self._camera_names_override is not None:
            color_cameras = self._camera_names_override
        elif self.load_stereo_images:
            color_cameras = [
                '23_A', '23_B', '24_A', '24_B', '25_A', '25_B', '26_A', '26_B', '27_A', '27_B',
                '28_A', '28_B', '29_A', '29_B', '30_A', '30_B'
            ]
        else:
            color_cameras = [
                '23_C','24_C','25_C','26_C','27_C',
                '28_C','29_C','30_C'
            ]

        calib = np.load(self.calibration_img_fname_grid(subject, sequence, frame))
        
        frame_grid = imageio.imread(self.img_fname_grid(subject, sequence, frame), pilmode='RGB')
        # normals_grid = imageio.imread(self.normals_img_fname_grid(subject, sequence, frame), pilmode='RGB') if self.normals_img_fname_grid is not None else None
        normals_grid = np.load(self.normals_img_fname_grid(subject, sequence, frame)) if self.normals_img_fname_grid is not None else None

        fan_landmarks = np.load(self.fan_landmarks_fname(subject, sequence, frame)) if self.fan_landmarks_fname is not None else None
        mediapipe_landmarks = np.load(self.mediapipe_landmarks_fname(subject, sequence, frame)) if getattr(self, 'mediapipe_landmarks_fname', None) is not None else None

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

        if getattr(self, 'dense_semantic_landmarks_fname', None) is not None:
            try:
                dense_semantic_path = self.dense_semantic_landmarks_fname(subject, sequence, frame)
                dense_semantic_data = np.load(dense_semantic_path) if os.path.exists(dense_semantic_path) else None
                # print(dense_semantic_data, 'dense semantic path')
                # raise
                dense_fan_landmarks = dense_semantic_data['fan_landmarks']
                dense_mediapipe_landmarks = dense_semantic_data['mediapipe_landmarks']
            except Exception as e:
                print('Error loading dense semantic landmarks:', e)
                raise e
                # dense_landmarks = np.zeros((len(calib['intrinsics']), 5023, 2), dtype=np.float32)
                # dense_mask_all = torch.zeros((len(calib['intrinsics']), 1), dtype=torch.long)

        # print(mediapipe_landmarks.shape)
        # raise

        depths = np.load(self.depths_fname(subject, sequence, frame)) if self.depths_fname is not None else None
        # print(depths, 'depths shape')
        # print(depths.shape, frame_grid.shape, normals_grid.shape)

        h_grid, w_grid = frame_grid.shape[:2]
        h_one, w_one = h_grid, w_grid // len(calib['intrinsics'])

        perturbation = None
        for i in range(len(calib['intrinsics'])):
            try:
                fan_landmarks_scores_masks = np.load(self.fan_landmarks_orig_fnam(subject, sequence, frame, color_cameras[i]))
            except Exception as e:
                fan_landmarks_scores_masks = {
                    'mask': np.array([0], dtype=bool),
                    'scores': np.zeros((1, 68), dtype=np.float32),
                }



            fan_landmarks_mask = fan_landmarks_scores_masks['mask']

            fan_landmarks_scores_avg = fan_landmarks_scores_masks['scores'][0].mean()

            if fan_landmarks_scores_avg < 0.2:
                fan_landmarks_mask = torch.zeros_like(torch.from_numpy(fan_landmarks_mask), dtype=torch.long)
            else:
                fan_landmarks_mask = torch.from_numpy(fan_landmarks_mask).long()

            if fan_landmarks is not None:
                if fan_landmarks[i][:,0].std() < 1e-3: 
                    fan_landmarks_mask = torch.zeros_like(fan_landmarks_mask).long()

            if depths is not None:
                depth_img = depths[:, i * w_one:(i + 1) * w_one]
            else:
                depth_img = None

            # print(depth_img.max(), depth_img.min(), 'depth img max min')
            
            frame_img = frame_grid[:, i * w_one:(i + 1) * w_one, :]
            normals_img = normals_grid[:, i * w_one:(i + 1) * w_one, :] if normals_grid is not None else None
            calib_img = {
                'intrinsics': calib['intrinsics'][i],
                'extrinsics': calib['extrinsics'][i],
                'radial_distortion': calib['radial_distortions'][i],
                'centers': calib['centers'][i]
            }

            # pad the image and intrinsics to match ViT patch size
            if self.pad_to_vit or PAD_TO_VIT:
                multiple = max(1, int(self.pad_multiple))
                resize_factor = max(1, int(self.image_resize_factor))
                target_h = int(math.ceil(frame_img.shape[0] / (multiple * resize_factor)) * (multiple * resize_factor))
                target_w = int(math.ceil(frame_img.shape[1] / (multiple * resize_factor)) * (multiple * resize_factor))
                pad_results = pad_img_and_intrinsics(
                    frame_img,
                    calib_img['intrinsics'],
                    normals_cropped=normals_img,
                    depth_img=depth_img,
                    dense_landmarks=dense_landmarks[i] if dense_landmarks is not None else None,
                    out_size=(target_h, target_w),
                )

                  #fan_landmarks_scores_masks['landmarks'][0])
                # raise "Mediapipe Landmarks not padded yet"
                # print(pad_results['image'].shape, frame_img.shape)
                frame_img = pad_results['image']
                calib_img['intrinsics'] = pad_results['K']
                normals_img = pad_results['normals_image']
                landmarks = pad_results['landmarks']
                dense_landmarks_view = pad_results['dense_landmarks']
                depth_img = pad_results['depth_img']
            else:
                landmarks = fan_landmarks[i] if fan_landmarks is not None else None
                dense_landmarks_view = dense_landmarks[i] if dense_landmarks is not None else None

            try:
                mediapipe_landmarks_scores_masks = np.load(self.mediapipe_landmarks_orig_fnam(subject, sequence, frame, color_cameras[i])) if getattr(self, 'mediapipe_landmarks_orig_fnam', None) is not None else None
            except Exception as e:
                    mediapipe_landmarks_scores_masks = {
                    'mask': np.array([0], dtype=bool),
                    # 'scores': np.zeros((1, 105), dtype=np.float32),
                }

            # Prepare per-view MediaPipe landmarks (no padding used)
            mp_landmarks_view = mediapipe_landmarks[i] if mediapipe_landmarks is not None else None


            dense_fan_landmarks_view = dense_fan_landmarks[i] if dense_landmarks is not None else None
            dense_mediapipe_landmarks_view = dense_mediapipe_landmarks[i] if dense_landmarks is not None else None

            # dense_landmarks_view = dense_landmarks[i] if dense_landmarks is not None else None
            # print(mediapipe_landmarks.shape, mp_landmarks_view.shape, 'mp landmarks shape')
            mediapipe_landmarks_mask = torch.from_numpy(mediapipe_landmarks_scores_masks['mask']).long() if mediapipe_landmarks_scores_masks is not None else None

            if mp_landmarks_view is not None:
                if mp_landmarks_view[:,0].std() < 1e-3: 
                    mediapipe_landmarks_mask = torch.zeros_like(mediapipe_landmarks_mask).long()

            img_with_camera = self.augment_img_with_camera(
                subject,
                sequence,
                frame_img,
                normals_img,
                calib_img,
                to_meters=to_meters,
                perturbation=perturbation,
                landmarks=landmarks,
                depth_img=depth_img,
                extra_landmarks=mp_landmarks_view,
                dense_landmarks=dense_landmarks_view,
                dense_fan_landmarks=dense_fan_landmarks_view,
                dense_mediapipe_landmarks=dense_mediapipe_landmarks_view
            )
            perturbation = img_with_camera['perturbation'] 
            
            # print(img_with_camera['depth_map'].max(), img_with_camera['depth_map'].min(), 'depth img max min 2')
            # print(img_with_camera['depth_map_augmented'].max(), img_with_camera['depth_map_augmented'].min(), 'depth img max min aug')
# 
            # raise 

            # print(img_with_camera.keys())
            if img_with_camera is not None:
                    # print(img_with_camera['intrinsics'].shape, 'intri')
                    # if '_C' in get_filename(calib_fname):
                    color_images.append(img_with_camera['image'])
                    color_camera_intrinsics.append(img_with_camera['intrinsics'])
                    color_camera_extrinsics.append(img_with_camera['extrinsics'])
                    color_camera_distortions.append(img_with_camera['radial_distortion'])
                    color_camera_centers.append(img_with_camera['camera_center'])
                    color_images_augmented.append(img_with_camera['image_augmented'])
                    color_camera_intrinsics_augmented.append(img_with_camera['intrinsics_augmented'])
                    # color_camera_landmarks_masks.append(landmarks_mask)
                    # camera_names.append(get_filename(calib_fname))

                    if img_with_camera['landmarks'] is not None:
                        color_camera_landmarks.append(img_with_camera['landmarks'])
                        color_camera_landmarks_augmented.append(img_with_camera['landmarks_augmented'])

                    if img_with_camera.get('dense_landmarks', None) is not None:
                        color_camera_dense_landmarks.append(img_with_camera['dense_landmarks'])
                        color_camera_dense_landmarks_augmented.append(img_with_camera['dense_landmarks_augmented'])

                        color_camera_dense_landmarks_masks.append(dense_mask_all[i])
                        color_camera_dense_landmarks_masks_augmented.append(dense_mask_all[i])

                        color_camera_dense_fan_landmarks.append(img_with_camera['dense_fan_landmarks'])
                        color_camera_dense_fan_landmarks_augmented.append(img_with_camera['dense_fan_landmarks_augmented'])
                        color_camera_dense_mediapipe_landmarks.append(img_with_camera['dense_mediapipe_landmarks'])
                        color_camera_dense_mediapipe_landmarks_augmented.append(img_with_camera['dense_mediapipe_landmarks_augmented'])

                    # MediaPipe landmarks collected if present
                    if 'landmarks_mediapipe' in img_with_camera:
                        if 'color_camera_landmarks_mediapipe' not in locals():
                            color_camera_landmarks_mediapipe = []
                            color_camera_landmarks_mediapipe_augmented = []
                        color_camera_landmarks_mediapipe.append(img_with_camera['landmarks_mediapipe'])
                        color_camera_landmarks_mediapipe_augmented.append(img_with_camera['landmarks_mediapipe_augmented'])

                    if img_with_camera['normals_image'] is not None:
                        color_images_normals.append(img_with_camera['normals_image'])
                        color_images_normals_augmented.append(img_with_camera['normals_image_augmented'])
                        

                    # color_camera_landmarks_masks.append(torch.from_numpy(fan_landmarks_scores_masks['mask']))
                    if img_with_camera['depth_map'] is not None:
                        color_images_depth.append(img_with_camera['depth_map'])
                        color_images_depth_augmented.append(img_with_camera['depth_map_augmented'])

                    if img_with_camera['landmarks_hull_mask'] is not None:                    
                        color_camera_landmarks_hull_masks.append(img_with_camera['landmarks_hull_mask'])
                        color_camera_landmarks_hull_masks_augmented.append(img_with_camera['landmarks_hull_mask_augmented'])

                    if fan_landmarks is not None:
                        color_camera_landmarks_masks.append(fan_landmarks_mask)
                        color_camera_landmarks_masks_augmented.append(fan_landmarks_mask)

                    # MediaPipe simple masks (1 if non-degenerate else 0)
                    if mp_landmarks_view is not None:
                        if 'color_camera_landmarks_mediapipe_masks' not in locals():
                            color_camera_landmarks_mediapipe_masks = []
                            color_camera_landmarks_mediapipe_masks_augmented = []
                        color_camera_landmarks_mediapipe_masks.append(mediapipe_landmarks_mask)
                        color_camera_landmarks_mediapipe_masks_augmented.append(mediapipe_landmarks_mask)


                # else:
                #     stereo_images.append(img_with_camera['image'])
                #     stereo_camera_intrinsics.append(img_with_camera['intrinsics'])
                #     stereo_camera_extrinsics.append(img_with_camera['extrinsics'])
                #     stereo_camera_distortions.append(img_with_camera['radial_distortion'])   
                #     stereo_camera_centers.append(img_with_camera['camera_center'])         
                #     stereo_images_augmented.append(img_with_camera['image_augmented'])
                #     stereo_camera_intrinsics_augmented.append(img_with_camera['intrinsics_augmented'])                    
                #     # stereo_camera_landmarks.append(img_with_camera['landmarks'])
                #     # stereo_camera_landmarks_augmented.append(img_with_camera['landmarks_augmented'])
                #     stereo_camera_landmarks_masks.append(landmarks_mask)
                #     camera_names.append(get_filename(calib_fname))

                #     if img_with_camera['landmarks'] is not None:
                #         stereo_camera_landmarks.append(img_with_camera['landmarks'])
                #         stereo_camera_landmarks_augmented.append(img_with_camera['landmarks_augmented'])

                #     if img_with_camera['normals_image'] is not None:
                #         stereo_images_normals.append(img_with_camera['normals_image'])
                #         stereo_images_normals_augmented.append(img_with_camera['normals_image_augmented'])


        # --------------- View-dropout: randomly omit entire views ---------------- #
        # kept = []
        # for i in range(len(color_images)):
        #     if random.random() > self.view_drop_prob:
        #         kept.append(i)
        # if len(kept) == 0 and len(color_images) > 0:
        #     kept = [random.randrange(len(color_images))]
        # kept = random.sample(range(len(color_images)), len(color_images)//2)
        # kept = range(len(color_images)) # do not drop any views

        # color_images = [color_images[i] for i in kept]
        # color_camera_intrinsics = [color_camera_intrinsics[i] for i in kept]
        # color_camera_extrinsics = [color_camera_extrinsics[i] for i in kept]
        # color_camera_distortions = [color_camera_distortions[i] for i in kept]
        # color_camera_centers = [color_camera_centers[i] for i in kept]
        # color_images_augmented = [color_images_augmented[i] for i in kept]
        # color_camera_intrinsics_augmented = [color_camera_intrinsics_augmented[i] for i in kept]
        # color_images_normals = [color_images_normals[i] for i in kept] if len(color_images_normals) > 0 else []
        # color_images_normals_augmented = [color_images_normals_augmented[i] for i in kept] if len(color_images_normals_augmented) > 0 else []


        # # print(len(color_images), len(color_camera_intrinsics))
        # # raise
        # if len(stereo_images) > 0:
        #     stereo_images = torch.stack(stereo_images, dim=0)
        #     stereo_camera_intrinsics = torch.stack(stereo_camera_intrinsics, dim=0)
        #     stereo_camera_extrinsics = torch.stack(stereo_camera_extrinsics, dim=0)
        #     stereo_camera_distortions = torch.stack(stereo_camera_distortions, dim=0)
        #     stereo_camera_centers = torch.stack(stereo_camera_centers, dim=0)
        #     stereo_images_augmented = torch.stack(stereo_images_augmented, dim=0)
        #     stereo_camera_intrinsics_augmented = torch.stack(stereo_camera_intrinsics_augmented, dim=0)
        #     # stereo_camera_landmarks = torch.stack(stereo_camera_landmarks, dim=0)
        #     # stereo_camera_landmarks_augmented = torch.stack(stereo_camera_landmarks_augmented, dim=0)
        #     # stereo_camera_landmarks_masks = torch.stack(stereo_camera_landmarks_masks, dim=0)
        #     stereo_camera_landmarks = torch.stack(stereo_camera_landmarks, dim=0) if len(stereo_camera_landmarks) > 0 else None
        #     stereo_camera_landmarks_augmented = torch.stack(stereo_camera_landmarks_augmented, dim=0) if len(stereo_camera_landmarks_augmented) > 0 else None

        #     stereo_images_normals = torch.stack(stereo_images_normals, dim=0) if len(stereo_images_normals) > 0 else None
        #     stereo_images_normals_augmented = torch.stack(stereo_images_normals_augmented, dim=0) if len(stereo_images_normals_augmented) > 0 else None

        if len(color_images) > 0:
            color_images = torch.stack(color_images, dim=0)
            color_camera_intrinsics = torch.stack(color_camera_intrinsics, dim=0)
            color_camera_extrinsics = torch.stack(color_camera_extrinsics, dim=0)

            color_camera_extrinsics[:, :3, 3] /= self.to_meters_scale_factor

            color_camera_distortions = torch.stack(color_camera_distortions, dim=0)
            color_camera_centers = torch.stack(color_camera_centers, dim=0)
            color_images_augmented = torch.stack(color_images_augmented, dim=0)
            color_camera_intrinsics_augmented = torch.stack(color_camera_intrinsics_augmented, dim=0)

            if len(color_camera_landmarks) > 0:
                color_camera_landmarks = torch.stack(color_camera_landmarks, dim=0)
                color_camera_landmarks_augmented = torch.stack(color_camera_landmarks_augmented, dim=0)

            if len(color_camera_dense_landmarks) > 0:
                color_images_normals = torch.stack(color_images_normals, dim=0) if len(color_images_normals) > 0 else None
                color_images_normals_augmented = torch.stack(color_images_normals_augmented, dim=0) if len(color_images_normals_augmented) > 0 else None


            if len(color_images_depth) > 0:
                color_images_depth = torch.stack(color_images_depth, dim=0) if len(color_images_depth) > 0 else None
                color_images_depth_augmented = torch.stack(color_images_depth_augmented, dim=0) if len(color_images_depth_augmented) > 0 else None


            if len(color_camera_landmarks_masks) > 0:
                color_camera_landmarks_masks = torch.stack(color_camera_landmarks_masks, dim=0) if len(color_camera_landmarks_masks) > 0 else None
                color_camera_landmarks_hull_masks = torch.stack(color_camera_landmarks_hull_masks, dim=0) if len(color_camera_landmarks_hull_masks) > 0 else None
                color_camera_landmarks_hull_masks_augmented = torch.stack(color_camera_landmarks_hull_masks_augmented, dim=0) if len(color_camera_landmarks_hull_masks_augmented) > 0 else None
                color_camera_landmarks_masks_augmented = torch.stack(color_camera_landmarks_masks_augmented, dim=0) if len(color_camera_landmarks_masks_augmented) > 0 else None

            if len(color_camera_dense_landmarks) > 0:
                color_camera_dense_landmarks = torch.stack(color_camera_dense_landmarks, dim=0) if len(color_camera_dense_landmarks) > 0 else None
                color_camera_dense_landmarks_augmented = torch.stack(color_camera_dense_landmarks_augmented, dim=0) if len(color_camera_dense_landmarks_augmented) > 0 else None
                color_camera_dense_landmarks_masks = torch.stack(color_camera_dense_landmarks_masks, dim=0) if len(color_camera_dense_landmarks_masks) > 0 else None
                color_camera_dense_landmarks_masks_augmented = torch.stack(color_camera_dense_landmarks_masks_augmented, dim=0) if len(color_camera_dense_landmarks_masks_augmented) > 0 else None

                color_camera_dense_fan_landmarks = torch.stack(color_camera_dense_fan_landmarks, dim=0) if len(color_camera_dense_fan_landmarks) > 0 else None
                color_camera_dense_fan_landmarks_augmented = torch.stack(color_camera_dense_fan_landmarks_augmented, dim=0) if len(color_camera_dense_fan_landmarks_augmented) > 0 else None
                color_camera_dense_mediapipe_landmarks = torch.stack(color_camera_dense_mediapipe_landmarks, dim=0) if len(color_camera_dense_mediapipe_landmarks) > 0 else None
                color_camera_dense_mediapipe_landmarks_augmented = torch.stack(color_camera_dense_mediapipe_landmarks_augmented, dim=0) if len(color_camera_dense_mediapipe_landmarks_augmented) > 0 else None

            # Stack MediaPipe if present
            if 'color_camera_landmarks_mediapipe' in locals():
                color_camera_landmarks_mediapipe = torch.stack(color_camera_landmarks_mediapipe, dim=0)
                color_camera_landmarks_mediapipe_augmented = torch.stack(color_camera_landmarks_mediapipe_augmented, dim=0)
                color_camera_landmarks_mediapipe_masks = torch.stack(color_camera_landmarks_mediapipe_masks, dim=0)
                color_camera_landmarks_mediapipe_masks_augmented = torch.stack(color_camera_landmarks_mediapipe_masks_augmented, dim=0)

        data = {
            # img
            'color_images': color_images,
            'stereo_images': stereo_images,
            'color_images_augmented': color_images_augmented,
            'stereo_images_augmented': stereo_images_augmented,

            # normals
            'color_images_normals': color_images_normals,
            'stereo_images_normals': stereo_images_normals,
            'color_images_normals_augmented': color_images_normals_augmented,
            'stereo_images_normals_augmented': stereo_images_normals_augmented,

            # camera
            'color_camera_intrinsics': color_camera_intrinsics,
            'color_camera_extrinsics': color_camera_extrinsics,
            'color_camera_distortions': color_camera_distortions,
            'color_camera_centers': color_camera_centers,
            'stereo_camera_intrinsics': stereo_camera_intrinsics,
            'stereo_camera_extrinsics': stereo_camera_extrinsics,
            'stereo_camera_distortions': stereo_camera_distortions,
            'stereo_camera_centers': stereo_camera_centers,
            
            'color_camera_intrinsics_augmented': color_camera_intrinsics_augmented,
            'stereo_camera_intrinsics_augmented': stereo_camera_intrinsics_augmented,

            'color_camera_landmarks': color_camera_landmarks,
            # 'stereo_camera_landmarks': stereo_camera_landmarks,
            'color_camera_landmarks_augmented': color_camera_landmarks_augmented,
            # 'stereo_camera_landmarks_augmented': stereo_camera_landmarks_augmented,

            'color_camera_dense_landmarks': color_camera_dense_landmarks,
            'color_camera_dense_landmarks_augmented': color_camera_dense_landmarks_augmented,

            'color_camera_landmarks_masks': color_camera_landmarks_masks,
            # 'stereo_camera_landmarks_masks': stereo_camera_landmarks_masks,

            'color_camera_dense_landmarks_masks': color_camera_dense_landmarks_masks,

            'color_images_depth': color_images_depth,
            'color_images_depth_augmented': color_images_depth_augmented,

            'color_camera_landmarks_masks_augmented': color_camera_landmarks_masks_augmented,
            'color_camera_landmarks_hull_masks': color_camera_landmarks_hull_masks,
            'color_camera_landmarks_hull_masks_augmented': color_camera_landmarks_hull_masks_augmented,

            'color_camera_dense_landmarks_masks_augmented': color_camera_dense_landmarks_masks_augmented,


            'color_camera_dense_fan_landmarks': color_camera_dense_fan_landmarks,
            'color_camera_dense_fan_landmarks_augmented': color_camera_dense_fan_landmarks_augmented,
            'color_camera_dense_mediapipe_landmarks': color_camera_dense_mediapipe_landmarks,
            'color_camera_dense_mediapipe_landmarks_augmented': color_camera_dense_mediapipe_landmarks_augmented,


            # meta
            'index': index,
            'subject': subject,
            'sequence': sequence,
            'frame': frame,   
            # 'camera_names': camera_names
        }


        # Attach MediaPipe landmarks if available
        if 'color_camera_landmarks_mediapipe' in locals():
            data.update({
                'color_camera_landmarks_mediapipe': color_camera_landmarks_mediapipe,
                'color_camera_landmarks_mediapipe_augmented': color_camera_landmarks_mediapipe_augmented,
                'color_camera_landmarks_mediapipe_masks': color_camera_landmarks_mediapipe_masks,
                'color_camera_landmarks_mediapipe_masks_augmented': color_camera_landmarks_mediapipe_masks_augmented,
            })

        # remove all None values
        data = {k: v for k, v in data.items() if v is not None}

        # print(self.scan_fname)
        # raise

        if (self.scan_fname != '') and (self.scan_vertex_count > 0):
            scan_fname = self.scan_fname(subject, sequence, frame)
            if self.return_full_scan:
                try:
                    data['v_scan'], data['f_scan'] = self.load_scan(scan_fname)
                    data['v_scan'] = torch.from_numpy(data['v_scan'].astype(np.float32))
                    if self.to_meters:
                        data['v_scan'] /= 1000
                    data['f_scan'] = torch.from_numpy(data['f_scan'].astype(np.int64))
                except Exception as e:
                    print(f'Unable to load scan {scan_fname}: {e}')
                    # data['v_scan'] = None
                    # data['f_scan'] = None
            else:
                v_sampled = self.load_scan_vertices(scan_fname)
                data['v_scan'] = torch.from_numpy(np.array(v_sampled).astype(np.float32))

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

                if self.mesh_sampler is not None:
                    for level in range(1,self.mesh_sampler.get_number_levels()):
                        v_registration, f_registration = self.mesh_sampler.downsample(v_registration, return_faces=True)

                data['v_reg_sampled'] = torch.from_numpy(v_registration.astype(np.float32))
                data['f_reg_sampled'] = torch.from_numpy(f_registration.astype(np.int64))

        if self.global_mesh_fname != '':
            global_mesh_fname = self.global_mesh_fname(subject, sequence, frame)
            if not os.path.exists(global_mesh_fname):
                print(f'Global mesh not found {global_mesh_fname}')

            try:
                global_mesh = Mesh(filename=global_mesh_fname)
                if not self.to_meters:
                    global_mesh.v[:] *= 1000 # FLAME registrations are in meters, if to_meters is false, convert them to milimeters
            except:
                print(f'Unable to load global mesh {global_mesh_fname}')
            data['v_reg_global'] = torch.from_numpy(global_mesh.v.astype(np.float32))
            data['f_reg_global'] = torch.from_numpy(global_mesh.f.astype(np.int64)) 
        else:
            # If no data from the global stage are provided, use the downsampled registrations as global stage initialization. 
            # Otherwise, load the global meshes.      
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

    def load_scan_vertices(self, scan_fname):
        if not os.path.exists(scan_fname):
            raise RuntimeError(f'Scan not found {scan_fname}')

        file_extension = utils.get_extension(scan_fname)
        if file_extension.lower() in ['.obj', '.ply']:
            try:
                scan = Mesh(filename=scan_fname)
            except:
                raise RuntimeError(f'Unable to load scan {scan_fname}')

            import trimesh
            tr_mesh = trimesh.Trimesh(vertices=scan.v, faces=scan.f)
            v_sampled, _ = trimesh.sample.sample_surface(tr_mesh, self.scan_vertex_count)
            return v_sampled
        elif file_extension.lower() in ['.npy']:
            v_sampled = np.load(scan_fname)
            scan_v_ids = np.arange(v_sampled.shape[0])
            random.shuffle(scan_v_ids)
            scan_v_ids = scan_v_ids[:np.min((v_sampled.shape[0], self.scan_vertex_count))]
            return v_sampled[scan_v_ids]
        else:
            raise RuntimeError(f'Unknown scan file extension {file_extension}')

    def augment_img_with_camera(
        self,
        subject,
        sequence,
        image,
        normals_image,
        calib,
        to_meters=False,
        perturbation=None,
        landmarks=None,          # FAN landmarks
        depth_img=None,
        extra_landmarks=None,    # MediaPipe landmarks
        dense_landmarks=None,
        dense_fan_landmarks=None,
        dense_mediapipe_landmarks=None,
    ):
        # ---- inputs to float [0,1] for geometric ops ----
        image = image.astype(np.float32) / 255.0
        # normals_image = normals_image.astype(np.float32) / 255.0 if normals_image is not None else None
        normals_image = normals_image.astype(np.float32) if normals_image is not None else None
        h, w = image.shape[:2]

        camera = {
            'intrinsics': calib['intrinsics'],
            'extrinsics': calib['extrinsics'],
            'radial_distortion': calib['radial_distortion'],
            'camera_center': calib['centers'],
        }

        # decide which path we take
        use_direct_size = (
            getattr(self, 'output_image_height', None) is not None and
            getattr(self, 'output_image_width',  None) is not None
        )
        using_depth_path = (use_direct_size and (depth_img is not None))

        if using_depth_path:
            # ============================================================
            # PATH A: depth-guided fixed-size crop THEN random scale+crop
            # ============================================================
            target_h = int(self.output_image_height)
            target_w = int(self.output_image_width)

            # 1) Depth-guided square crop (geometry only) → K' = U_depth @ K
            crop_res = crop_square_by_depth_and_update_K(
                img=image,
                K=camera['intrinsics'],
                depth=depth_img.squeeze() if depth_img is not None else None,
                normals=normals_image,
                landmarks=landmarks,
                mediapipe_landmarks=extra_landmarks,  # make sure arg name matches helper
                dense_landmarks=dense_landmarks,
                target_size=(target_h, target_w),
                depth_nonzero_eps=0.0,
                bbox_margin=1.0,
                min_bbox_px=16,
            )
            image_c    = crop_res['image']
            normals_c  = crop_res['normals_image']
            depth_c    = crop_res['depth_img']
            K_c        = crop_res['K'] if crop_res['K'] is not None else camera['intrinsics']
            lms_c      = crop_res['landmarks'] if landmarks is not None else None
            lms_mp_c   = crop_res['landmarks_mediapipe'] if extra_landmarks is not None else None
            lms_dense_c = crop_res['landmarks_dense'] if dense_landmarks is not None else None

            # 2) Random scale+crop ON the cropped tensors (output stays target_h x target_w)
            if not DISABLE_AUGM:
                np.random.seed()
                crop_size = (target_h, target_w)
                scale_factor = self.scale_min + (self.scale_max - self.scale_min) * np.random.random()
                h_offset, w_offset = get_random_crop_offsets(crop_size, height=target_h, width=target_w)

                sc = scale_crop(
                    image_c, crop_size, h_offset, w_offset, scale_factor,
                    K=K_c, normals_image=normals_c, debug=False,
                    landmarks=lms_c, depth_map=depth_c
                )
                image_aug_np   = sc['image']
                K_aug_np       = sc['K'] if sc['K'] is not None else K_c
                lms_aug_np     = sc['landmarks']
                normals_aug_np = sc['normals_image']
                depth_aug_np   = sc['depth_map']

                # MediaPipe landmarks through same aug (no K branch)
                lms_mp_aug_np = None
                if lms_mp_c is not None:
                    sc_mp = scale_crop(
                        image_c, crop_size, h_offset, w_offset, scale_factor,
                        K=None, normals_image=None, debug=False,
                        landmarks=lms_mp_c, depth_map=None
                    )
                    lms_mp_aug_np = sc_mp['landmarks']
                lms_dense_aug_np = None
                if lms_dense_c is not None:
                    sc_dense = scale_crop(
                        image_c, crop_size, h_offset, w_offset, scale_factor,
                        K=None, normals_image=None, debug=False,
                        landmarks=lms_dense_c, depth_map=None
                    )
                    lms_dense_aug_np = sc_dense['landmarks']
            else:
                # no geometric aug
                image_aug_np   = image_c.copy()
                K_aug_np       = K_c.copy()
                lms_aug_np     = lms_c.copy() if lms_c is not None else None
                normals_aug_np = normals_c.copy() if normals_c is not None else None
                depth_aug_np   = depth_c.copy() if depth_c is not None else None
                lms_mp_aug_np  = lms_mp_c.copy() if lms_mp_c is not None else None
                lms_dense_aug_np = lms_dense_c.copy() if lms_dense_c is not None else None

            # 3) brightness jitter AFTER geometry
            if perturbation is None:
                perturb = 1.0 + self.brightness_sigma * np.random.randn(1, 1, 3)
            else:
                perturb = perturbation
            image_aug_np = np.clip(image_aug_np * perturb, 0.0, 1.0)

            # 4) convex hull masks (base & augmented)
            lm_mask      = np.zeros((target_h, target_w), dtype=np.uint8)
            lm_mask_aug  = np.zeros((target_h, target_w), dtype=np.uint8)
            if lms_c is not None and lms_c.shape[0] > 2:
                hull = cv2.convexHull(lms_c[:, :2].astype(np.int32))
                cv2.fillConvexPoly(lm_mask, hull, 1)
            if lms_aug_np is not None and lms_aug_np.shape[0] > 2:
                hull = cv2.convexHull(lms_aug_np[:, :2].astype(np.int32))
                cv2.fillConvexPoly(lm_mask_aug, hull, 1)
            lm_mask     = torch.from_numpy(lm_mask.astype(bool))
            lm_mask_aug = torch.from_numpy(lm_mask_aug.astype(bool))

            # 5) normalize RGB to your stats
            image_c_norm   = self.normalize_image(image_c)
            image_aug_norm = self.normalize_image(image_aug_np)

            # 6) pack tensors (NO image_resize_factor here — fixed output size has priority)
            image_t          = torch.from_numpy(image_c_norm.astype(np.float32)).permute(2, 0, 1).contiguous()
            image_aug_t      = torch.from_numpy(image_aug_norm.astype(np.float32)).permute(2, 0, 1).contiguous()
            intrinsics_t     = torch.from_numpy(K_c.astype(np.float32))
            intrinsics_aug_t = torch.from_numpy(K_aug_np.astype(np.float32))
            extrinsics_t     = torch.from_numpy(camera['extrinsics'].astype(np.float32))
            radial_t         = torch.from_numpy(camera['radial_distortion'].astype(np.float32))
            center_t         = torch.from_numpy(camera['camera_center'].astype(np.float32))

            normals_t        = torch.from_numpy(normals_c.astype(np.float32)).permute(2,0,1).contiguous() if normals_c is not None else None
            normals_aug_t    = torch.from_numpy(normals_aug_np.astype(np.float32)).permute(2,0,1).contiguous() if normals_aug_np is not None else normals_t

            depth_t          = torch.from_numpy(depth_c.astype(np.float32))[:, :, None] if depth_c is not None else None
            depth_aug_t      = torch.from_numpy(depth_aug_np.astype(np.float32))[:, :, None] if depth_aug_np is not None else depth_t

            lms_t            = torch.from_numpy(lms_c.astype(np.float32)) if lms_c is not None else None
            lms_aug_t        = torch.from_numpy(lms_aug_np.astype(np.float32)) if lms_aug_np is not None else lms_t
            lms_mp_t         = torch.from_numpy(lms_mp_c.astype(np.float32)) if lms_mp_c is not None else None
            lms_mp_aug_t     = torch.from_numpy(lms_mp_aug_np.astype(np.float32)) if lms_mp_aug_np is not None else lms_mp_t
            lms_dense_t      = torch.from_numpy(lms_dense_c.astype(np.float32)) if lms_dense_c is not None else None
            lms_dense_aug_t  = torch.from_numpy(lms_dense_aug_np.astype(np.float32)) if lms_dense_aug_np is not None else lms_dense_t

            out = {
                'image': image_t,
                'image_augmented': image_aug_t,
                'intrinsics': intrinsics_t,
                'intrinsics_augmented': intrinsics_aug_t,
                'extrinsics': extrinsics_t,
                'radial_distortion': radial_t,
                'camera_center': center_t,
                'normals_image': normals_t,
                'normals_image_augmented': normals_aug_t,
                'depth_map': depth_t,
                'depth_map_augmented': depth_aug_t,
                'landmarks': lms_t,
                'landmarks_augmented': lms_aug_t,
                'landmarks_mediapipe': lms_mp_t,
                'landmarks_mediapipe_augmented': lms_mp_aug_t,
                'dense_landmarks': lms_dense_t,
                'dense_landmarks_augmented': lms_dense_aug_t,
                'landmarks_hull_mask': lm_mask,
                'landmarks_hull_mask_augmented': lm_mask_aug,
                'perturbation': perturb,
            }
            return out

        # ============================================================
        # PATH B (fallback): original random scale+crop augmentation
        # ============================================================
        if not DISABLE_AUGM:
            np.random.seed()
            crop_size = (h, w)  # full-res crop window
            scale_factor = self.scale_min + (self.scale_max - self.scale_min) * np.random.random()
            h_offset, w_offset = get_random_crop_offsets(crop_size, height=h, width=w)

            normals_image = normals_image * 2.0 - 1.0 if normals_image is not None else None # bring to [-1,1] for warp ops
            # print("WAHG!")

            sc = scale_crop(
                image, crop_size, h_offset, w_offset, scale_factor,
                K=camera['intrinsics'], normals_image=normals_image, debug=False,
                debug_root='debug_/', landmarks=landmarks, depth_map=depth_img
            )
            image_augmented = sc['image']
            intrinsics_augmented = sc['K']
            landmarks_augmented = sc['landmarks']
            normals_augmented = sc['normals_image']
            depth_augmented = sc['depth_map']

            # MediaPipe landmarks: same aug params (no K)
            landmarks_mediapipe_augmented = None
            if extra_landmarks is not None:
                sc_mp = scale_crop(
                    image, crop_size, h_offset, w_offset, scale_factor,
                    K=None, normals_image=None, debug=False, debug_root=None,
                    landmarks=extra_landmarks, depth_map=None
                )
                landmarks_mediapipe_augmented = sc_mp['landmarks']

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

            dense_fan_landmarks_augmented = None
            if dense_fan_landmarks is not None:
                sc_dense_fan = scale_crop(
                    image, crop_size, h_offset, w_offset, scale_factor,
                    K=None, normals_image=None, debug=False, debug_root=None,
                    landmarks=dense_fan_landmarks, depth_map=None
                )
                dense_fan_landmarks_augmented = sc_dense_fan['landmarks']

        else:
            image_augmented = image.copy()
            intrinsics_augmented = camera['intrinsics'].copy()
            normals_augmented = normals_image.copy() if normals_image is not None else None
            landmarks_augmented = landmarks.copy() if landmarks is not None else None
            depth_augmented = depth_img.copy() if depth_img is not None else None
            landmarks_mediapipe_augmented = extra_landmarks.copy() if extra_landmarks is not None else None
            dense_landmarks_augmented = dense_landmarks.copy() if dense_landmarks is not None else None
            dense_mediapipe_landmarks_augmented = dense_mediapipe_landmarks.copy() if dense_mediapipe_landmarks is not None else None
            dense_fan_landmarks_augmented = dense_fan_landmarks.copy() if dense_fan_landmarks is not None else None

        # brightness jitter AFTER geometry
        if perturbation is None:
            perturb = 1.0 + self.brightness_sigma * np.random.randn(1, 1, 3)
        else:
            perturb = perturbation
        image_augmented = np.clip(image_augmented * perturb, 0.0, 1.0)

        # convex hull masks (base & augmented)
        h_aug, w_aug = image_augmented.shape[:2]
        landmark_mask_augmented = np.zeros((h_aug, w_aug), dtype=np.uint8)
        if landmarks_augmented is not None and landmarks_augmented.shape[0] > 2:
            pts = landmarks_augmented[:, :2].astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(landmark_mask_augmented, hull, 1)
        landmark_mask_augmented = torch.from_numpy(landmark_mask_augmented.astype(bool))

        landmark_mask = np.zeros((h, w), dtype=np.uint8)
        if landmarks is not None and landmarks.shape[0] > 2:
            pts = landmarks[:, :2].astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(landmark_mask, hull, 1)
        landmark_mask = torch.from_numpy(landmark_mask.astype(bool))

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

            # adjust landmarks
            if landmarks is not None:
                landmarks[:, 0] /= self.image_resize_factor
                landmarks[:, 1] /= self.image_resize_factor
            if landmarks_augmented is not None:
                landmarks_augmented[:, 0] /= self.image_resize_factor
                landmarks_augmented[:, 1] /= self.image_resize_factor
            if extra_landmarks is not None:
                extra_landmarks[:, 0] /= self.image_resize_factor
                extra_landmarks[:, 1] /= self.image_resize_factor
            if landmarks_mediapipe_augmented is not None:
                landmarks_mediapipe_augmented[:, 0] /= self.image_resize_factor
                landmarks_mediapipe_augmented[:, 1] /= self.image_resize_factor
            if dense_landmarks is not None:
                dense_landmarks[:, 0] /= self.image_resize_factor
                dense_landmarks[:, 1] /= self.image_resize_factor
            if dense_landmarks_augmented is not None:
                dense_landmarks_augmented[:, 0] /= self.image_resize_factor
                dense_landmarks_augmented[:, 1] /= self.image_resize_factor

            if dense_fan_landmarks is not None:
                dense_fan_landmarks[:, 0] /= self.image_resize_factor
                dense_fan_landmarks[:, 1] /= self.image_resize_factor
            if dense_fan_landmarks_augmented is not None:
                dense_fan_landmarks_augmented[:, 0] /= self.image_resize_factor
                dense_fan_landmarks_augmented[:, 1] /= self.image_resize_factor
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

        landmarks_t = torch.from_numpy(landmarks.astype(np.float32)) if landmarks is not None else None
        landmarks_aug_t = torch.from_numpy(landmarks_augmented.astype(np.float32)) if landmarks_augmented is not None else None
        landmarks_mediapipe_t = torch.from_numpy(extra_landmarks.astype(np.float32)) if extra_landmarks is not None else None
        landmarks_mediapipe_aug_t = torch.from_numpy(landmarks_mediapipe_augmented.astype(np.float32)) if landmarks_mediapipe_augmented is not None else None
        dense_landmarks_t = torch.from_numpy(dense_landmarks.astype(np.float32)) if dense_landmarks is not None else None
        dense_landmarks_aug_t = torch.from_numpy(dense_landmarks_augmented.astype(np.float32)) if dense_landmarks_augmented is not None else dense_landmarks_t
        dense_fan_landmarks_t = torch.from_numpy(dense_fan_landmarks.astype(np.float32)) if dense_fan_landmarks is not None else None
        dense_fan_landmarks_aug_t = torch.from_numpy(dense_fan_landmarks_augmented.astype(np.float32)) if dense_fan_landmarks_augmented is not None else dense_fan_landmarks_t
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
            'landmarks': landmarks_t,
            'landmarks_augmented': landmarks_aug_t,

            'depth_map_augmented': depth_aug_t,
            'depth_map': depth_img_t,

            'landmarks_hull_mask': landmark_mask,
            'landmarks_hull_mask_augmented': landmark_mask_augmented,
        }

        if landmarks_mediapipe_t is not None:
            out.update({
                'landmarks_mediapipe': landmarks_mediapipe_t,
                'landmarks_mediapipe_augmented': landmarks_mediapipe_aug_t,
            })

        if dense_landmarks_t is not None:
            out.update({
                'dense_landmarks': dense_landmarks_t,
                'dense_landmarks_augmented': dense_landmarks_aug_t,
            })

        if dense_fan_landmarks_t is not None:
            out.update({
                'dense_fan_landmarks': dense_fan_landmarks_t,
                'dense_fan_landmarks_augmented': dense_fan_landmarks_aug_t,
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


import numpy as np
import cv2

def crop_square_by_depth_and_update_K(
    img,
    K=None,
    depth=None,
    normals=None,
    landmarks=None,
    mediapipe_landmarks=None,
    dense_landmarks=None,
    target_size=(308, 308),      # (H, W) — can be rectangular; square recommended
    depth_nonzero_eps=0.0,       # treat values > eps as valid depth
    bbox_margin=1.15,            # enlarge bbox by this factor
    min_bbox_px=16,              # fallback bbox size if too small
    border_mode=cv2.BORDER_CONSTANT,
    border_value_img=0,
    border_value_normals=0,
    border_value_depth=0
):
    """
    Returns:
        {
          'image': np.ndarray(Ht, Wt, 3), float32 in [0,1] if input was,
          'normals_image': np.ndarray(Ht, Wt, C) or None,
          'depth_img': np.ndarray(Ht, Wt) or None,
          'K': np.ndarray(3,3) or None,
          'landmarks': np.ndarray(N,2) or None,
          'transform': np.ndarray(3,3)  # pixel homography U applied to go img->cropped
        }
    """
    assert img.ndim == 3 and img.shape[2] == 3, f"img expected HxWx3, got {img.shape}"
    H, W = img.shape[:2]
    Ht, Wt = target_size

    # ---------- 1) make a bbox from depth ----------
    # If no depth, fallback to entire image.
    if depth is not None:
        assert depth.ndim in (2,3), "depth should be HxW or HxWx1"
        d = depth.squeeze()
        valid = d > depth_nonzero_eps
        ys, xs = np.where(valid)
        if ys.size > 0:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
        else:
            # fallback: entire image
            y0, y1, x0, x1 = 0, H-1, 0, W-1
    else:
        y0, y1, x0, x1 = 0, H-1, 0, W-1

    # ---------- 2) expand to a square around center ----------
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    # expand bbox a bit
    bw = max(bw, min_bbox_px)
    bh = max(bh, min_bbox_px)
    size = max(bw, bh) * float(bbox_margin)

    # center of bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5

    # square window edges in original pixels (float)
    half = size * 0.5
    left   = cx - half
    top    = cy - half

    # ---------- 3) build the similarity (scale+translate) transform ----------
    # Map the square window [left, top, size] to [0..Wt-1, 0..Ht-1].
    # No rotation, just scale+translate:
    sx = (Wt - 1) / size
    sy = (Ht - 1) / size
    tx = -left
    ty = -top

    # 3x3 homography (pixel -> pixel)
    U = np.array([
        [sx,  0,  sx * tx],
        [ 0, sy,  sy * ty],
        [ 0,  0,        1]
    ], dtype=np.float64)

    # 2x3 for warpAffine
    A = U[:2, :]

    # ---------- 4) warp image / normals / depth ----------
    # NOTE: If your img is in [0,1] float, cv2.warpAffine preserves dtype; pick interpolation accordingly.
    img_out = cv2.warpAffine(img, A, (Wt, Ht),
                             flags=cv2.INTER_LINEAR,
                             borderMode=border_mode,
                             borderValue=border_value_img)

    normals_out = None
    if normals is not None:
        normals_out = cv2.warpAffine(normals, A, (Wt, Ht),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=border_mode,
                                     borderValue=border_value_normals)

    depth_out = None
    if depth is not None:
        # Keep depth discrete/unaltered where possible (nearest)
        depth_squeeze = depth.squeeze()
        depth_out = cv2.warpAffine(depth_squeeze, A, (Wt, Ht),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=border_value_depth)

    # ---------- 5) transform landmarks ----------
    landmarks_out = None
    if landmarks is not None:
        if landmarks.shape[1] == 2:
            ones = np.ones((landmarks.shape[0], 1), dtype=landmarks.dtype)
            P = np.hstack([landmarks, ones])        # Nx3
            Pp = (U @ P.T).T                         # Nx3
            landmarks_out = Pp[:, :2]
        else:
            raise ValueError("landmarks must be (N,2)")

    landmarks_mediapipe_out = None
    if mediapipe_landmarks is not None:
        if mediapipe_landmarks.shape[1] == 2:
            ones = np.ones((mediapipe_landmarks.shape[0], 1), dtype=mediapipe_landmarks.dtype)
            P = np.hstack([mediapipe_landmarks, ones])        # Nx3
            Pp = (U @ P.T).T                         # Nx3
            landmarks_mediapipe_out = Pp[:, :2]
        else:
            raise ValueError("mediapipe_landmarks must be (N,2)")

    landmarks_dense_out = None
    if dense_landmarks is not None:
        if dense_landmarks.shape[1] == 2:
            ones = np.ones((dense_landmarks.shape[0], 1), dtype=dense_landmarks.dtype)
            P = np.hstack([dense_landmarks, ones])        # Nx3
            Pp = (U @ P.T).T                              # Nx3
            landmarks_dense_out = Pp[:, :2]
        else:
            raise ValueError("dense_landmarks must be (N,2)")

    # ---------- 6) update intrinsics ----------
    K_out = None
    if K is not None:
        # Left-multiply pixel transform: x' = U * (K * X_cam)
        K_out = (U @ K).astype(np.float64)

    return {
        'image': img_out,
        'normals_image': normals_out,
        'depth_img': depth_out,
        'K': K_out if K is not None else None,
        'landmarks': landmarks_out,
        'transform': U,
        'landmarks_mediapipe': landmarks_mediapipe_out,
        'landmarks_dense': landmarks_dense_out
    }
