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
from utils.data_augment import get_random_crop_offsets, scale_crop, crop_img, pad_width_or_height
from utils.utils import get_filename
import cv2

def undistort_image(K, distortion, rgb, landmarks=None):
    # print('Undistorting image with shape:', rgb.shape)
    width, height = rgb.shape[1], rgb.shape[0]
    # print(K.shape, distortion.shape, rgb.shape)
    distortion = np.concatenate([distortion, np.zeros(2)], axis=0)  # Ensure distortion has 4 coefs -> p1 p2 are 0 for our cameras
    # import time
    # start = time.time()
    new_K, validPixROI = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), centerPrincipalPoint=True)    
    map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    # set to black
    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # reshape into Nx1×2 for OpenCV:
    pts = landmarks.astype(np.float32).reshape(-1,1,2)

    # call undistortPoints—R=None means no rectification rotation,
    # and P=new_K means reproject back into pixel coordinates
    undistorted_pts = cv2.undistortPoints(pts,
                                        cameraMatrix=K,
                                        distCoeffs=distortion,
                                        R=None,
                                        P=new_K)

    # reshape back to (68,2)
    fan_landmarks_undist = undistorted_pts.reshape(-1,2)

    return new_K, undistorted_image, fan_landmarks_undist

class FaceAlignDatasetMPI(data.Dataset):
    def __init__(self, 
                data_list_fname,
                dataset_root_dir='',
                image_dir='',
                image_resize_factor = 1,
                calibration_dir='',
                scan_dir='',
                normals_dir='',
                registration_root_dir='',   
                global_registration_root_dir='',
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
                return_full_scan=False,
                undistort_images=False,
                undistorted_K_dir='',
                segmentation_dir='',
                mediapipe_landmarks_dir='',
                dense_landmarks_dir=''
                ):
        super().__init__()
       
        if os.path.exists(data_list_fname):
            self.split_list = utils.load_json(data_list_fname)
        else:
            raise RuntimeError('Invalid data path - %s' % data_list_fname)
        print()
        # self.split_list = self.split_list[100:102]
        self.load_stereo_images = load_stereo_images
        self.load_color_images = load_color_images
        self.calibration_blacklist = calibration_blacklist
        self.return_full_scan = return_full_scan

        self.undistorted_K_dir = undistorted_K_dir

        # augmentation
        self.scale_min = scale_min # random scaling
        self.scale_max = scale_max
        self.brightness_sigma = brightness_sigma # random brightness perturbation

        self.mesh_sampler = mesh_sampler
        self.scan_vertex_count = scan_vertex_count
        self.registration_root_dir = registration_root_dir
        self.undistort_images = undistort_images

        if os.path.exists(image_dir):
            self.img_dir = lambda subject, sequence, frame : os.path.join(image_dir, subject, sequence, frame)
            self.img_fname = lambda subject, sequence, frame, view : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s.%s' % (sequence, frame, view, image_file_ext))
            self.img_fname_grid = lambda subject, sequence, frame : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s' % (sequence, frame, image_file_ext))

        elif os.path.exists(dataset_root_dir):
            self.img_dir = lambda subject, sequence, frame : os.path.join(dataset_root_dir, subject, sequence, 'images', frame)
            self.img_fname = lambda subject, sequence, frame, view : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s.%s' % (sequence, frame, view, image_file_ext))
        else:
            raise RuntimeError('Invalid image directory')


        if os.path.exists(normals_dir):
            self.normals_img_dir = lambda subject, sequence, frame : os.path.join(normals_dir, subject, sequence, frame)
            self.normals_img_fname = lambda subject, sequence, frame, view : os.path.join(self.normals_img_dir(subject, sequence, frame), '%s.%s.%s.%s' % (sequence, frame, view, image_file_ext))
            self.normals_img_fname_grid = lambda subject, sequence, frame : os.path.join(self.normals_img_dir(subject, sequence, frame), '%s.%s.%s' % (sequence, frame, image_file_ext))
        else:
            self.normals_img_dir = None
            self.normals_img_fname = None
            # raise RuntimeError('Invalid normals directory')

        if os.path.exists(calibration_dir):
            self.calibration_dir = lambda subject, sequence : os.path.join(calibration_dir, subject, sequence)
        elif os.path.exists(dataset_root_dir):
            self.calibration_dir = lambda subject, sequence : os.path.join(dataset_root_dir, subject, sequence,  'meshes', '*')
        else:
            raise RuntimeError('Invalid calibration directory')

        if os.path.exists(fan_landmarks_dir):
            self.fan_landmarks_dir = lambda subject, sequence, frame : os.path.join(fan_landmarks_dir, subject, sequence, frame)
            self.fan_landmarks_fname = lambda subject, sequence, frame, view : os.path.join(self.fan_landmarks_dir(subject, sequence, frame), '%s.%s.%s.npz' % (sequence, frame, view))
        else:
            self.fan_landmarks_dir = None
            self.fan_landmarks_fname = None
            # raise RuntimeError('Invalid fan landmarks directory')

        # optional mediapipe landmarks dir (npz files with keys 'landmarks' (1,478,3) and 'mask')
        if os.path.exists(mediapipe_landmarks_dir):
            self.mediapipe_landmarks_dir = lambda subject, sequence, frame : os.path.join(mediapipe_landmarks_dir, subject, sequence, frame)
            self.mediapipe_landmarks_fname = lambda subject, sequence, frame, view : os.path.join(self.mediapipe_landmarks_dir(subject, sequence, frame), '%s.%s.%s.npz' % (sequence, frame, view))
        else:
            self.mediapipe_landmarks_dir = None
            self.mediapipe_landmarks_fname = None

        if os.path.exists(dense_landmarks_dir):
            self.dense_landmarks_dir = lambda subject, sequence, frame : os.path.join(dense_landmarks_dir, subject, sequence, frame)
            self.dense_landmarks_fname = lambda subject, sequence, frame, view : os.path.join(self.dense_landmarks_dir(subject, sequence, frame), '%s.%s.%s.npy' % (sequence, frame, view))
        else:
            self.dense_landmarks_dir = None
            self.dense_landmarks_fname = None
            # raise RuntimeError('Invalid dense landmarks directory')

        # optional segmentation dir (npz files with key 'seg_mask')
        if os.path.exists(segmentation_dir):
            self.segmentation_dir = lambda subject, sequence, frame : os.path.join(segmentation_dir, subject, sequence, frame)
            self.segmentation_fname = lambda subject, sequence, frame, view : os.path.join(self.segmentation_dir(subject, sequence, frame), '%s.%s.%s.npz' % (sequence, frame, view))
        else:
            self.segmentation_dir = None
            self.segmentation_fname = None

        self.scan_fname = ''
        if os.path.exists(scan_dir):
            if return_full_scan:
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


        if undistorted_K_dir != '':
            self.undistorted_K_fname = lambda subject, sequence, frame : os.path.join(undistorted_K_dir, subject, sequence, frame, '%s.%s_intrinsics.npy' % (sequence, frame))

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
        to_meters = False
        subject, sequence, frame = self.split_list[index]

        # Read calibration files
        # print(self.calibration_dir(subject, sequence))
        calib_fnames = sorted(glob.glob(os.path.join(self.calibration_dir(subject, sequence), '*.tka')))
        # print('Num calib fnames: %d' % len(calib_fnames))
        # print(calib_fnames)

        # Read stereo and color images for each calibration file
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

        color_camera_landmarks_masks = []
        stereo_camera_landmarks_masks = []

        color_camera_dense_landmarks = []
        stereo_camera_dense_landmarks = []
        color_camera_dense_landmarks_augmented = []
        stereo_camera_dense_landmarks_augmented = []

        # mediapipe landmarks (mirroring FAN landmarks API)
        color_camera_mediapipe_landmarks = []
        stereo_camera_mediapipe_landmarks = []
        color_camera_mediapipe_landmarks_augmented = []
        stereo_camera_mediapipe_landmarks_augmented = []

        color_camera_mediapipe_landmarks_masks = []
        stereo_camera_mediapipe_landmarks_masks = []

        # segmentation maps per view (optional)
        color_segmentation_maps = []
        color_segmentation_maps_augmented = []
        stereo_segmentation_maps = []
        stereo_segmentation_maps_augmented = []

        camera_names =[]

        frame_grid = None
        intrinsics_grid = None

        if not self.load_stereo_images and not self.load_color_images:
            raise RuntimeError('No images to load - set load_stereo_images or load_color_images to True')
        if not self.load_stereo_images and self.load_color_images:
            calib_fnames = [f for f in calib_fnames if '_C' in get_filename(f)]
        if not self.load_color_images and self.load_stereo_images:
            calib_fnames = [f for f in calib_fnames if '_A' in get_filename(f) or '_B' in get_filename(f)]
        
        if len(calib_fnames) == 0:
            raise RuntimeError('No calibration files found for subject %s, sequence %s, frame %s' % (subject, sequence, frame))


        for i,calib_fname in enumerate(calib_fnames):
            # print('Process calib fname %s' % calib_fname)
            if (not self.load_stereo_images) and ('_A' in get_filename(calib_fname) or '_B' in get_filename(calib_fname)):
                continue
            if (not self.load_color_images) and ('_C' in get_filename(calib_fname)):
                continue
            if get_filename(calib_fname) in self.calibration_blacklist:
                continue
            

            if self.fan_landmarks_dir is not None:
                landmarks, landmarks_mask = self.read_fan_landmarks(subject, sequence, frame, calib_fname)
                if landmarks is None or landmarks_mask is None:
                    landmarks = np.zeros((68,2), dtype=np.float32)
                    landmarks_mask = np.zeros((), dtype=np.float32)
                else:
                    landmarks = landmarks[0]
                    landmarks_mask = landmarks_mask[0]
                # print('Landmarks shape:', landmarks.shape, 'Mask shape:', landmarks_mask.shape)
            else:
                landmarks = np.zeros((68,2), dtype=np.float32)
                landmarks_mask = np.zeros((), dtype=np.float32)

            # Mediapipe landmarks (optional)
            if self.mediapipe_landmarks_dir is not None:
                mp_landmarks, mp_landmarks_mask = self.read_mediapipe_landmarks(subject, sequence, frame, calib_fname)
                if mp_landmarks is None or mp_landmarks_mask is None:
                    mp_landmarks = np.zeros((478,2), dtype=np.float32)
                    mp_landmarks_mask = np.zeros((), dtype=np.float32)
                else:
                    mp_landmarks = mp_landmarks[0]
                    mp_landmarks_mask = mp_landmarks_mask[0]
            else:
                mp_landmarks = np.zeros((478,2), dtype=np.float32)
                mp_landmarks_mask = np.zeros((), dtype=np.float32)

            
            if self.dense_landmarks_dir is not None:
                dense_landmarks_fname = self.dense_landmarks_fname(subject, sequence, frame, get_filename(calib_fname))
                # print(dense_landmarks_fname)
                if not os.path.exists(dense_landmarks_fname):
                    print('No dense landmarks found at:', dense_landmarks_fname)
                    raise
                    dense_landmarks = None
                else:
                    dense_landmarks = np.load(dense_landmarks_fname)
            else:
                dense_landmarks = None
                    
                    # print('dense landmarks shape:', dense_landmarks.shape)
                    # raise
            # print(dense_landmarks.shape)
            # check if the frames are saved as one long hconcat image
            # print('Looking for image grid at:', self.img_fname_grid(subject, sequence, frame))
            if os.path.exists(self.img_fname_grid(subject, sequence, frame)):
                if frame_grid is None:
                    frame_grid = imageio.imread(self.img_fname_grid(subject, sequence, frame), pilmode='RGB')
                    intrinsics_grid = np.load(self.undistorted_K_fname(subject, sequence, frame))
                    normals_grid = imageio.imread(self.normals_img_fname_grid(subject, sequence, frame), pilmode='RGB').astype(np.float32) / 255.

                h_grid, w_grid = frame_grid.shape[:2]
                h_one, w_one = h_grid, w_grid // len(calib_fnames)
                # print(h_grid, frame_grid.shape, w_one, w_grid)
                frame_img = frame_grid[:, i * w_one:(i + 1) * w_one, :]
                intrinsics_img = intrinsics_grid[i]
                normals_img = normals_grid[:, i * w_one:(i + 1) * w_one, :]
                # print(normals_grid.shape, frame_grid.shape, frame_img.shape, normals_img.shape)


                # optional segmentation map per view
                segmentation_img = None
                if self.segmentation_dir is not None:
                    seg_fname = self.segmentation_fname(subject, sequence, frame, get_filename(calib_fname))
                    if os.path.exists(seg_fname):
                        try:
                            seg_npz = np.load(seg_fname)
                            segmentation_img = seg_npz['seg_mask']
                        except Exception as e:
                            print(f'Unable to load segmentation npz {seg_fname}: {e}')

                img_with_camera = self.read_img_with_camera(subject, sequence, frame_img, calib_fname, to_meters=to_meters, update_intrinsics=intrinsics_img,
                    landmarks=landmarks,
                    normals_image=normals_img,
                    segmentation_map=segmentation_img,
                    mediapipe_landmarks=mp_landmarks)
            else:
                # optional segmentation map per view
                segmentation_img = None
                if self.segmentation_dir is not None:
                    seg_fname = self.segmentation_fname(subject, sequence, frame, get_filename(calib_fname))
                    if os.path.exists(seg_fname):
                        try:
                            seg_npz = np.load(seg_fname)
                            segmentation_img = seg_npz['seg_mask']
                            # print(np.unique(segmentation_img, return_counts=True))
                        except Exception as e:
                            print(f'Unable to load segmentation npz {seg_fname}: {e}')

                img_with_camera = self.read_img_with_camera(subject, sequence, frame, calib_fname, to_meters=to_meters, landmarks=landmarks,
                                                            segmentation_map=segmentation_img,
                                                            mediapipe_landmarks=mp_landmarks, dense_landmarks=dense_landmarks)

            # print(img_with_camera.keys())
            if img_with_camera is not None:
                if '_C' in get_filename(calib_fname):
                    color_images.append(img_with_camera['image'])
                    color_camera_intrinsics.append(img_with_camera['intrinsics'])
                    color_camera_extrinsics.append(img_with_camera['extrinsics'])
                    color_camera_distortions.append(img_with_camera['radial_distortion'])
                    color_camera_centers.append(img_with_camera['camera_center'])
                    color_images_augmented.append(img_with_camera['image_augmented'])
                    color_camera_intrinsics_augmented.append(img_with_camera['intrinsics_augmented'])
                    color_camera_landmarks_masks.append(landmarks_mask)
                    color_camera_mediapipe_landmarks_masks.append(mp_landmarks_mask)
                    camera_names.append(get_filename(calib_fname))

                    if img_with_camera['landmarks'] is not None:
                        color_camera_landmarks.append(img_with_camera['landmarks'])
                        color_camera_landmarks_augmented.append(img_with_camera['landmarks_augmented'])

                    if img_with_camera.get('mediapipe_landmarks', None) is not None:
                        color_camera_mediapipe_landmarks.append(img_with_camera['mediapipe_landmarks'])
                        color_camera_mediapipe_landmarks_augmented.append(img_with_camera['mediapipe_landmarks_augmented'])
                        
                    if img_with_camera.get('dense_landmarks', None) is not None:
                        # print('dense landmarks shape:', img_with_camera['dense_landmarks'].shape)
                        color_camera_dense_landmarks.append(img_with_camera['dense_landmarks'])
                        color_camera_dense_landmarks_augmented.append(img_with_camera['dense_landmarks_augmented'])

                    if img_with_camera['normals_image'] is not None:
                        color_images_normals.append(img_with_camera['normals_image'])
                        color_images_normals_augmented.append(img_with_camera['normals_image_augmented'])
                    # segmentation
                    if img_with_camera.get('segmentation_map', None) is not None:
                        color_segmentation_maps.append(img_with_camera['segmentation_map'].numpy()) if isinstance(img_with_camera['segmentation_map'], torch.Tensor) else color_segmentation_maps.append(img_with_camera['segmentation_map'])
                        if img_with_camera.get('segmentation_map_augmented', None) is not None:
                            color_segmentation_maps_augmented.append(img_with_camera['segmentation_map_augmented'].numpy()) if isinstance(img_with_camera['segmentation_map_augmented'], torch.Tensor) else color_segmentation_maps_augmented.append(img_with_camera['segmentation_map_augmented'])
                else:
                    stereo_images.append(img_with_camera['image'])
                    stereo_camera_intrinsics.append(img_with_camera['intrinsics'])
                    stereo_camera_extrinsics.append(img_with_camera['extrinsics'])
                    stereo_camera_distortions.append(img_with_camera['radial_distortion'])   
                    stereo_camera_centers.append(img_with_camera['camera_center'])         
                    stereo_images_augmented.append(img_with_camera['image_augmented'])
                    stereo_camera_intrinsics_augmented.append(img_with_camera['intrinsics_augmented'])                    
                    # stereo_camera_landmarks.append(img_with_camera['landmarks'])
                    # stereo_camera_landmarks_augmented.append(img_with_camera['landmarks_augmented'])
                    stereo_camera_landmarks_masks.append(landmarks_mask)
                    stereo_camera_mediapipe_landmarks_masks.append(mp_landmarks_mask)
                    camera_names.append(get_filename(calib_fname))

                    if img_with_camera['landmarks'] is not None:
                        stereo_camera_landmarks.append(img_with_camera['landmarks'])
                        stereo_camera_landmarks_augmented.append(img_with_camera['landmarks_augmented'])

                    if img_with_camera.get('mediapipe_landmarks', None) is not None:
                        stereo_camera_mediapipe_landmarks.append(img_with_camera['mediapipe_landmarks'])
                        stereo_camera_mediapipe_landmarks_augmented.append(img_with_camera['mediapipe_landmarks_augmented'])

                    if img_with_camera.get('dense_landmarks', None) is not None:
                        # print('dense landmarks shape:', img_with_camera['dense_landmarks'].shape)
                        stereo_camera_dense_landmarks.append(img_with_camera['dense_landmarks'])
                        stereo_camera_dense_landmarks_augmented.append(img_with_camera['dense_landmarks_augmented'])

                    if img_with_camera['normals_image'] is not None:
                        stereo_images_normals.append(img_with_camera['normals_image'])
                        stereo_images_normals_augmented.append(img_with_camera['normals_image_augmented'])

                    if img_with_camera.get('segmentation_map', None) is not None:
                        stereo_segmentation_maps.append(img_with_camera['segmentation_map'].numpy()) if isinstance(img_with_camera['segmentation_map'], torch.Tensor) else stereo_segmentation_maps.append(img_with_camera['segmentation_map'])
                        if img_with_camera.get('segmentation_map_augmented', None) is not None:
                            stereo_segmentation_maps_augmented.append(img_with_camera['segmentation_map_augmented'].numpy()) if isinstance(img_with_camera['segmentation_map_augmented'], torch.Tensor) else stereo_segmentation_maps_augmented.append(img_with_camera['segmentation_map_augmented'])

    
        color_camera_landmarks_masks = torch.from_numpy(np.array(color_camera_landmarks_masks)).int()
        stereo_camera_landmarks_masks = torch.from_numpy(np.array(stereo_camera_landmarks_masks)).int()
        # print(stereo_camera_landmarks_masks)

        if len(stereo_images) > 0:
            stereo_images = torch.stack(stereo_images, dim=0)
            stereo_camera_intrinsics = torch.stack(stereo_camera_intrinsics, dim=0)
            stereo_camera_extrinsics = torch.stack(stereo_camera_extrinsics, dim=0)
            stereo_camera_distortions = torch.stack(stereo_camera_distortions, dim=0)
            stereo_camera_centers = torch.stack(stereo_camera_centers, dim=0)
            stereo_images_augmented = torch.stack(stereo_images_augmented, dim=0)
            stereo_camera_intrinsics_augmented = torch.stack(stereo_camera_intrinsics_augmented, dim=0)
            # stereo_camera_landmarks = torch.stack(stereo_camera_landmarks, dim=0)
            # stereo_camera_landmarks_augmented = torch.stack(stereo_camera_landmarks_augmented, dim=0)
            # stereo_camera_landmarks_masks = torch.stack(stereo_camera_landmarks_masks, dim=0)
            stereo_camera_landmarks = torch.stack(stereo_camera_landmarks, dim=0) if len(stereo_camera_landmarks) > 0 else None
            stereo_camera_landmarks_augmented = torch.stack(stereo_camera_landmarks_augmented, dim=0) if len(stereo_camera_landmarks_augmented) > 0 else None

            stereo_camera_mediapipe_landmarks = torch.stack(stereo_camera_mediapipe_landmarks, dim=0) if len(stereo_camera_mediapipe_landmarks) > 0 else None
            stereo_camera_mediapipe_landmarks_augmented = torch.stack(stereo_camera_mediapipe_landmarks_augmented, dim=0) if len(stereo_camera_mediapipe_landmarks_augmented) > 0 else None

            stereo_camera_dense_landmarks = torch.stack(stereo_camera_dense_landmarks, dim=0) if len(stereo_camera_dense_landmarks) > 0 else None
            stereo_camera_dense_landmarks_augmented = torch.stack(stereo_camera_dense_landmarks_augmented, dim=0) if len(stereo_camera_dense_landmarks_augmented) > 0 else None

            stereo_images_normals = torch.stack(stereo_images_normals, dim=0) if len(stereo_images_normals) > 0 else None
            stereo_images_normals_augmented = torch.stack(stereo_images_normals_augmented, dim=0) if len(stereo_images_normals_augmented) > 0 else None

            # stack segmentation maps if present
            if len(stereo_segmentation_maps) > 0:
                try:
                    stereo_segmentation_maps = torch.stack([torch.from_numpy(m.astype(np.int64)) for m in stereo_segmentation_maps], dim=0)
                except Exception as e:
                    print('Error stacking stereo segmentation maps:', e)
                    stereo_segmentation_maps = None
                if len(stereo_segmentation_maps_augmented) > 0 and stereo_segmentation_maps_augmented[0] is not None:
                    try:
                        stereo_segmentation_maps_augmented = torch.stack([torch.from_numpy(m.astype(np.int64)) for m in stereo_segmentation_maps_augmented], dim=0)
                    except Exception as e:
                        print('Error stacking stereo segmentation maps augmented:', e)
                        stereo_segmentation_maps_augmented = None
            else:
                stereo_segmentation_maps = None
                stereo_segmentation_maps_augmented = None

        if len(color_images) > 0:
            color_images = torch.stack(color_images, dim=0)
            color_camera_intrinsics = torch.stack(color_camera_intrinsics, dim=0)
            color_camera_extrinsics = torch.stack(color_camera_extrinsics, dim=0)
            color_camera_distortions = torch.stack(color_camera_distortions, dim=0)
            color_camera_centers = torch.stack(color_camera_centers, dim=0)
            color_images_augmented = torch.stack(color_images_augmented, dim=0)
            color_camera_intrinsics_augmented = torch.stack(color_camera_intrinsics_augmented, dim=0)
            # color_camera_landmarks = torch.stack(color_camera_landmarks, dim=0)
            color_camera_mediapipe_landmarks = torch.stack(color_camera_mediapipe_landmarks, dim=0) if len(color_camera_mediapipe_landmarks) > 0 else None
            color_camera_mediapipe_landmarks_augmented = torch.stack(color_camera_mediapipe_landmarks_augmented, dim=0) if len(color_camera_mediapipe_landmarks_augmented) > 0 else None
            # color_camera_landmarks_augmented = torch.stack(color_camera_landmarks_augmented, dim=0)
            # color_camera_landmarks_masks = torch.stack(color_camera_landmarks_masks, dim=0)

            color_camera_dense_landmarks = torch.stack(color_camera_dense_landmarks, dim=0) if len(color_camera_dense_landmarks) > 0 else None
            color_camera_dense_landmarks_augmented = torch.stack(color_camera_dense_landmarks_augmented, dim=0) if len(color_camera_dense_landmarks_augmented) > 0 else None

            color_camera_landmarks = torch.stack(color_camera_landmarks, dim=0) if len(color_camera_landmarks) > 0 else None
            color_camera_landmarks_augmented = torch.stack(color_camera_landmarks_augmented, dim=0) if len(color_camera_landmarks_augmented) > 0 else None

            color_images_normals = torch.stack(color_images_normals, dim=0) if len(color_images_normals) > 0 else None
            color_images_normals_augmented = torch.stack(color_images_normals_augmented, dim=0) if len(color_images_normals_augmented) > 0 else None

            # stack segmentation maps if present
            if len(color_segmentation_maps) > 0:
                try:
                    color_segmentation_maps = torch.stack([torch.from_numpy(m.astype(np.int64)) for m in color_segmentation_maps], dim=0)
                except Exception as e:
                    print('Error stacking color segmentation maps:', e)
                    color_segmentation_maps = None
                if len(color_segmentation_maps_augmented) > 0 and color_segmentation_maps_augmented[0] is not None:
                    try:
                        color_segmentation_maps_augmented = torch.stack([torch.from_numpy(m.astype(np.int64)) for m in color_segmentation_maps_augmented], dim=0)
                    except Exception as e:
                        print('Error stacking color segmentation maps augmented:', e)
                        color_segmentation_maps_augmented = None
            else:
                color_segmentation_maps = None
                color_segmentation_maps_augmented = None
            

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

            # segmentation maps (labels)
            'color_segmentation_maps': color_segmentation_maps,
            'stereo_segmentation_maps': stereo_segmentation_maps,
            'color_segmentation_maps_augmented': color_segmentation_maps_augmented,
            'stereo_segmentation_maps_augmented': stereo_segmentation_maps_augmented,

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
            'stereo_camera_landmarks': stereo_camera_landmarks,
            'color_camera_landmarks_augmented': color_camera_landmarks_augmented,
            'stereo_camera_landmarks_augmented': stereo_camera_landmarks_augmented,

            'color_camera_dense_landmarks': color_camera_dense_landmarks,
            'stereo_camera_dense_landmarks': stereo_camera_dense_landmarks,

            'color_camera_landmarks_masks': color_camera_landmarks_masks,
            'stereo_camera_landmarks_masks': stereo_camera_landmarks_masks,

            # mediapipe landmarks
            'color_camera_mediapipe_landmarks': color_camera_mediapipe_landmarks,
            'stereo_camera_mediapipe_landmarks': stereo_camera_mediapipe_landmarks,
            'color_camera_mediapipe_landmarks_augmented': color_camera_mediapipe_landmarks_augmented,
            'stereo_camera_mediapipe_landmarks_augmented': stereo_camera_mediapipe_landmarks_augmented,
            'color_camera_mediapipe_landmarks_masks': torch.from_numpy(np.array(color_camera_mediapipe_landmarks_masks)).int(),
            'stereo_camera_mediapipe_landmarks_masks': torch.from_numpy(np.array(stereo_camera_mediapipe_landmarks_masks)).int(),

            # meta
            'index': index,
            'subject': subject,
            'sequence': sequence,
            'frame': frame,   
            'camera_names': camera_names
        }

        # remove all None values
        data = {k: v for k, v in data.items() if v is not None}

        if (self.scan_fname != '') and (self.scan_vertex_count > 0):
            scan_fname = self.scan_fname(subject, sequence, frame)
            if self.return_full_scan:
                try:
                    data['v_scan'], data['f_scan'] = self.load_scan(scan_fname)
                    data['v_scan'] = torch.from_numpy(data['v_scan'].astype(np.float32))
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
                    if not to_meters:
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
                if not to_meters:
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

    def read_fan_landmarks(self, subject, sequence, frame, calib_fname):
        fan_landmarks_fname = self.fan_landmarks_fname(subject, sequence, frame, get_filename(calib_fname))
        # print('Process fan landmarks %s' % fan_landmarks_fname)
        if not os.path.exists(fan_landmarks_fname):
            # raise RuntimeError('Invalid fan landmarks file - %s' % fan_landmarks_fname)
            return None, None
        try:
            landmarks = np.load(fan_landmarks_fname)
        except:
            print(f'Unable to load fan landmarks {fan_landmarks_fname}')
            return None, None
        landmarks_fan = landmarks['landmarks'][...,:2]
        landmarks_mask = landmarks['mask']
        # print('Landmarks shape:', landmarks_fan.shape, 'Mask shape:', landmarks_mask.shape, 'scores shape:', landmarks.get('scores', None).shape)
        # raise

        return landmarks_fan, landmarks_mask

    def read_mediapipe_landmarks(self, subject, sequence, frame, calib_fname):
        mediapipe_landmarks_fname = self.mediapipe_landmarks_fname(subject, sequence, frame, get_filename(calib_fname))
        if not os.path.exists(mediapipe_landmarks_fname):
            return None, None
        try:
            landmarks = np.load(mediapipe_landmarks_fname)
        except Exception as e:
            print(f'Unable to load mediapipe landmarks {mediapipe_landmarks_fname}: {e}')
            return None, None
        landmarks_mp = landmarks['landmarks'][...,:2]
        landmarks_mask = landmarks['mask']
        return landmarks_mp, landmarks_mask

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

    def read_img_with_camera(self, subject, sequence, frame, calib_fname, to_meters=True, landmarks=None, landmarks_mask=None,
                    update_intrinsics=None, normals_image=None, segmentation_map=None, mediapipe_landmarks=None, dense_landmarks=None):
        # print(type(frame))
        if isinstance(frame, str):
            image_fname = self.img_fname(subject, sequence, frame, get_filename(calib_fname))
            if not os.path.exists(image_fname):
                print('Image file not found - %s' % image_fname)
                return None

            try:
                image = imageio.imread(image_fname, pilmode='RGB')
            except:
                raise RuntimeError('Error loading image - %s' % image_fname)

            if self.normals_img_dir is not None:
                normals_image = imageio.imread(self.normals_img_fname(subject, sequence, frame, get_filename(calib_fname)), pilmode='RGB').astype(np.float32) / 255.
            else:
                normals_image = None

        elif isinstance(frame, np.ndarray) or isinstance(frame, imageio.core.util.Array):
            image = frame
            normals_image = normals_image
            image_fname = ''
        else:
            raise RuntimeError('Invalid frame type - %s' % type(frame))


        import cv2

        camera = load_mpi_camera(calib_fname, self.image_resize_factor, to_meters=to_meters)
        if camera is None:
            return None
        
        if update_intrinsics is not None:
            camera['intrinsics'] = update_intrinsics
            camera['radial_distortion'] = np.zeros(2)  # Set radial distortion to zero - these are already undistorted images
        # print(camera)
        # print(camera['name'], camera['intrinsics'][0][0])

        if self.undistort_images:
            D = camera['radial_distortion']
            K = camera['intrinsics']
            new_K, image, landmarks = undistort_image(K, D, image, landmarks)
            # undistort mediapipe landmarks with the same new_K
            if mediapipe_landmarks is not None:
                distortion = np.concatenate([D, np.zeros(2)], axis=0)
                pts = mediapipe_landmarks.astype(np.float32).reshape(-1,1,2)
                und = cv2.undistortPoints(pts, cameraMatrix=K, distCoeffs=distortion, R=None, P=new_K)
                mediapipe_landmarks = und.reshape(-1,2)

            if dense_landmarks is not None:
                # we have an issue here -> dense landmarks were computed on 1600x1200 images (wrong scale by me). so let's bring them to the original image size first
                dense_landmarks[:,0] = dense_landmarks[:,0] / self.image_resize_factor
                dense_landmarks[:,1] = dense_landmarks[:,1] / self.image_resize_factor

                distortion = np.concatenate([D, np.zeros(2)], axis=0)
                pts = dense_landmarks.astype(np.float32).reshape(-1,1,2)
                und = cv2.undistortPoints(pts, cameraMatrix=K, distCoeffs=distortion, R=None, P=new_K)
                dense_landmarks = und.reshape(-1,2)

            camera['intrinsics'] = new_K
            camera['radial_distortion'] = np.zeros(2)  # Set radial distortion to zero after undistortion
            # undistort segmentation map as labels (nearest)
            if segmentation_map is not None:
                seg = segmentation_map
                # Ensure 2D HxW labels before remap
                if isinstance(seg, np.ndarray) and seg.ndim == 3:
                    if seg.shape[0] == 1:         # (1,H,W) -> (H,W)
                        seg = seg[0]
                    elif seg.shape[-1] == 1:      # (H,W,1) -> (H,W)
                        seg = seg[..., 0]
                    else:
                        # If one-hot / multi-channel, take argmax along channel
                        seg = np.argmax(seg, axis=0) if seg.shape[0] <= 32 else np.argmax(seg, axis=-1)

                H, W = seg.shape[:2]
                distortion = np.concatenate([D, np.zeros(2)], axis=0)
                new_K_tmp, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (W, H), 1, (W, H), centerPrincipalPoint=True)
                map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K_tmp, (W, H), cv2.CV_32FC1)
                seg = cv2.remap(seg, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                segmentation_map = seg

            # print('Old intrinsics:', K)
            # print('New intrinsics:', camera['intrinsics'])

        image = image.astype(np.float32) / 255.

        # vis_img = image.copy()

        # import cv2
        # for i in range(landmarks.shape[0]):
        #     cv2.circle(vis_img, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (1.0, 0.0, 0.0), -1)
        # imageio.imsave('0_inp.png', (255.*vis_img).astype(np.uint8))

        # if self.image_resize_factor != 1:
        # print(image.shape, camera['image_size'])
        if (image.shape[0] != camera['image_size'][0]) or (image.shape[1] != camera['image_size'][1]):
            # print(camera['image_size'], image.shape)
            orig_shape = image.shape

            image = resize(image, (camera['image_size'][0], camera['image_size'][1]), anti_aliasing=True)
            if landmarks is not None:
                # print(landmarks.shape)
                # landmarks = landmarks/2
                landmarks[:,0] = landmarks[:,0] * camera['image_size'][1] / orig_shape[1]
                landmarks[:,1] = landmarks[:,1] * camera['image_size'][0] / orig_shape[0]
            if mediapipe_landmarks is not None:
                mediapipe_landmarks[:,0] = mediapipe_landmarks[:,0] * camera['image_size'][1] / orig_shape[1]
                mediapipe_landmarks[:,1] = mediapipe_landmarks[:,1] * camera['image_size'][0] / orig_shape[0]

            if dense_landmarks is not None:
                # print(orig_shape)
                raise
                dense_landmarks[:,0] = dense_landmarks[:,0] * camera['image_size'][1] / orig_shape[1]
                dense_landmarks[:,1] = dense_landmarks[:,1] * camera['image_size'][0] / orig_shape[0]

                # vis_img = image.copy()

                # import cv2
                # for i in range(landmarks.shape[0]):
                #     cv2.circle(vis_img, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (1.0, 0.0, 0.0), -1)
                # imageio.imsave('0_resz.png', (255.*vis_img).astype(np.uint8))
                # raise
        if (normals_image is not None) and self.image_resize_factor == 8:
            # downsample /2 
            normals_image = resize(normals_image, (normals_image.shape[0]//2, normals_image.shape[1]//2), anti_aliasing=True)

        if camera['image_size'][0] > camera['image_size'][1]:
            # The dataset contains images of landscape and portrait images of resolutions (A x B) and (B x A). 
            # To unify the images for batch handling, rotate all portrait images to landscape.
            # print(normals_image.shape, image.shape,'aa')
            # CARE -> NORMALS ARE ALREADY ROTATED AS SAVED 
            # print('Rotating !?')
            prev_camera = dict(camera)
            rot_results = rotate_image(image, camera, landmarks)#, normals_image=normals_image)
            image = rot_results['image']
            camera = rot_results['camera']
            landmarks = rot_results['landmarks']
            # normals_image = rot_results['normals_image']
            # print(normals_image.shape, image.shape,'bb')
            if segmentation_map is not None:
                segmentation_map = np.rot90(segmentation_map)
            # rotate mediapipe landmarks the same way
            if mediapipe_landmarks is not None:
                Rt = np.array([
                    [ 0,  1,                      0            ],
                    [-1,  0,  prev_camera['image_size'][1] ],
                    [ 0,  0,                      1            ]
                ])
                pts_h = np.concatenate((mediapipe_landmarks, np.ones((mediapipe_landmarks.shape[0],1))), axis=1)
                pts_rot_h = Rt.dot(pts_h.T).T
                mediapipe_landmarks = pts_rot_h[:, :2]

            if dense_landmarks is not None:
                Rt = np.array([
                    [ 0,  1,                      0            ],
                    [-1,  0,  prev_camera['image_size'][1] ],
                    [ 0,  0,                      1            ]
                ])
                pts_h = np.concatenate((dense_landmarks, np.ones((dense_landmarks.shape[0],1))), axis=1)
                pts_rot_h = Rt.dot(pts_h.T).T
                dense_landmarks = pts_rot_h[:, :2]
                
        # print(normals_image.shape, image.shape, 'cc')
        # geometric augmentation by random scaling and cropping
        np.random.seed()
        crop_size = (camera['image_size'][0], camera['image_size'][1])
        scale_factor = self.scale_min + (self.scale_max - self.scale_min) * np.random.random()
        h_offset, w_offset = get_random_crop_offsets(crop_size, height=camera['image_size'][0], width=camera['image_size'][1])

        # image_augmented, normals_augmented, intrinsics_augmented, landmarks_augmented = scale_crop(image, crop_size, h_offset, w_offset, scale_factor, K=camera['intrinsics'], 
        #     normals_image=normals_image, landmarks=landmarks.copy(), debug=False, debug_root="./debug/{}_{}_{}_{}".format(subject, sequence, frame, get_filename(calib_fname)))

        # THIS SHOULD BE LANDMARKS.COPY() I THINK
        scale_crop_results = scale_crop(image, crop_size, h_offset, w_offset, scale_factor, K=camera['intrinsics'],
            normals_image=normals_image, landmarks=landmarks, debug=False, debug_root=None)
        image_augmented = scale_crop_results['image']
        intrinsics_augmented = scale_crop_results['K']
        landmarks_augmented = scale_crop_results['landmarks']
        normals_augmented = scale_crop_results['normals_image']
        # replicate the same transform for mediapipe landmarks
        mediapipe_landmarks_augmented = None
        if mediapipe_landmarks is not None:
            scale_crop_results_mp = scale_crop(image, crop_size, h_offset, w_offset, scale_factor, K=camera['intrinsics'],
                normals_image=None, landmarks=mediapipe_landmarks, debug=False, debug_root=None)
            mediapipe_landmarks_augmented = scale_crop_results_mp['landmarks']

        if dense_landmarks is not None:
            scale_crop_results_dl = scale_crop(image, crop_size, h_offset, w_offset, scale_factor, K=camera['intrinsics'],
                normals_image=None, landmarks=dense_landmarks, debug=False, debug_root=None)
            dense_landmarks_augmented = scale_crop_results_dl['landmarks']

        # replicate transform for segmentation map using nearest-neighbor
        segmentation_map_augmented = None
        if segmentation_map is not None:
            if isinstance(crop_size, tuple):
                crop_h, crop_w = crop_size
            else:
                crop_h = crop_w = crop_size
            new_h = max(1, int(round(segmentation_map.shape[0] * scale_factor)))
            new_w = max(1, int(round(segmentation_map.shape[1] * scale_factor)))
            seg_scaled = cv2.resize(segmentation_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            seg_cropped = crop_img(seg_scaled, (crop_h, crop_w), h_offset, w_offset)
            if seg_cropped.shape[0] < crop_h or seg_cropped.shape[1] < crop_w:
                seg_cropped = pad_width_or_height(seg_cropped, output_width=crop_w, output_height=crop_h, pad_value=0)
            segmentation_map_augmented = seg_cropped
            

        # random brightness perturbation
        perturb = 1.0 + self.brightness_sigma * np.random.randn(1,1,3)
        image_augmented = image_augmented * perturb
        image_augmented = np.clip(image_augmented, 0., 1.)

        # normalize rgb
        image = self.normalize_image(image)
        image_augmented = self.normalize_image(image_augmented)

        image = torch.FloatTensor(torch.from_numpy(image.astype(np.float32))).permute(2,0,1).contiguous() # (3,H,W) range (0,1) only rgb
        intrinsics = torch.FloatTensor(torch.from_numpy(camera['intrinsics'].astype(np.float32)))
        extrinsics = torch.FloatTensor(torch.from_numpy(camera['extrinsics'].astype(np.float32)))
        radial_distortion = torch.FloatTensor(torch.from_numpy(camera['radial_distortion'].astype(np.float32)))
        camera_center = torch.FloatTensor(torch.from_numpy(camera['camera_center'].astype(np.float32)))

        normals_image = torch.FloatTensor(torch.from_numpy(normals_image.astype(np.float32))).permute(2,0,1).contiguous() if normals_image is not None else None

        image_augmented = torch.FloatTensor(torch.from_numpy(image_augmented.astype(np.float32))).permute(2,0,1).contiguous() # (3,H,W) range (0,1) only rgb
        intrinsics_augmented = torch.FloatTensor(torch.from_numpy(intrinsics_augmented.astype(np.float32)))

        normals_image_augmented = torch.FloatTensor(torch.from_numpy(normals_augmented.astype(np.float32))).permute(2,0,1).contiguous() if normals_augmented is not None else None

        # segmentation tensors
        seg_tensor = torch.from_numpy(segmentation_map.astype(np.int64)) if segmentation_map is not None else None
        seg_aug_tensor = torch.from_numpy(segmentation_map_augmented.astype(np.int64)) if segmentation_map_augmented is not None else None

        landmarks = torch.FloatTensor(torch.from_numpy(landmarks.astype(np.float32)))
        landmarks_augmented = torch.FloatTensor(torch.from_numpy(landmarks_augmented.astype(np.float32)))
        mediapipe_landmarks = torch.FloatTensor(torch.from_numpy(mediapipe_landmarks.astype(np.float32))) if mediapipe_landmarks is not None else None
        mediapipe_landmarks_augmented = torch.FloatTensor(torch.from_numpy(mediapipe_landmarks_augmented.astype(np.float32))) if mediapipe_landmarks_augmented is not None else None

        if dense_landmarks is not None:
            dense_landmarks = torch.FloatTensor(torch.from_numpy(dense_landmarks.astype(np.float32)))
            dense_landmarks_augmented = torch.FloatTensor(torch.from_numpy(dense_landmarks_augmented.astype(np.float32)))
        else:
            dense_landmarks = None
            dense_landmarks_augmented = None

        # scale landmarks from -1 to 1
        # landmarks[:,0] = landmarks[:,0] / image.shape[1]*2 - 1
        # landmarks[:,1] = landmarks[:,1] / image.shape[0]*2 - 1

        # landmarks_augmented[:,0] = landmarks_augmented[:,0] / image_augmented.shape[2]*2 - 1
        # landmarks_augmented[:,1] = landmarks_augmented[:,1] / image_augmented.shape[1]*2 - 1

        return {
                    'image': image, 
                    'image_fname': image_fname, 
                    'intrinsics': intrinsics, 
                    'extrinsics': extrinsics, 
                    'radial_distortion': radial_distortion,
                    'camera_center': camera_center,
                    #augmented images
                    'image_augmented': image_augmented,
                    'intrinsics_augmented': intrinsics_augmented,
                    # landmarks
                    'landmarks': landmarks,
                    'landmarks_augmented': landmarks_augmented,
                    # mediapipe landmarks
                    'mediapipe_landmarks': mediapipe_landmarks,
                    'mediapipe_landmarks_augmented': mediapipe_landmarks_augmented,
                    # normals
                    'normals_image': normals_image,
                    'normals_image_augmented': normals_image_augmented,
                    # segmentation
                    'segmentation_map': seg_tensor,
                    'segmentation_map_augmented': seg_aug_tensor,

                    # dense landmarks
                    'dense_landmarks': dense_landmarks,
                    'dense_landmarks_augmented': dense_landmarks_augmented,
                }

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
