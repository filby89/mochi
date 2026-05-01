import os
import time
import random
from typing import Optional
import wandb
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.autograd import Variable
from utils.utils import print_memory, to_numpy, get_time_string, add_labels_to_images
from utils.mesh_renderer import render_mesh, dist_to_rgb
from utils.data_augment import get_subset_views
from utils.point_to_point_loss import PointToPointLoss
from utils.point_to_surface_loss import PointToSurfaceLoss, compute_s2m_distance, GMO, batch_chamfer_distance
from utils.edge_loss import EdgeLoss
from utils.mesh_helper import MeshHelper, pointmap_to_rgb, depth_to_pointmap_robust
from kaolin.metrics.trianglemesh import uniform_laplacian_smoothing
from trainer.base_trainer import BaseTrainer
from option_handler.train_options_global import TrainOptions
from psbody.mesh import Mesh
from models.FLAME.FLAME import FLAME
import math
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pickle
import pprint
import kaolin
from pytorch3d.ops import corresponding_points_alignment
import trimesh
from tqdm import tqdm
import psutil
from trainer.utils import get_dense_vertex_colors, random_pixel_mask, draw_dense_points
import kornia
from utils.losses import geodesic_normal_loss_p3d, calculate_map_loss, calculate_gradient_map_loss, _points2surface_metric
import torch.nn.functional as F
import kornia
import pandas as pd



class Trainer(BaseTrainer):

    def __init__(self, args, device):
        super().__init__(args)
        self.args = args
        self.device = device

        self.no_jaw = True

        from models.FLAME.pliks_flame_2 import PliksFlameSolver  # <-- NEW

        self.flame = FLAME(flame_model_path='assets/FLAME2023/flame2023_no_jaw.pkl', no_jaw=True).to(device)

        self.pliks_solver = PliksFlameSolver(self.flame, locked_joint_ids=torch.Tensor([2]).long()).to(device)

        self.faces = self.flame.faces_tensor.cpu()

        self.mesh_helper = MeshHelper(num_vertices=5023, faces=self.faces)

        self.load_flame_masks()

        self.unit_factor = 1000.0

        if self.args.to_meters:
            self.args.global_voxel_inc = self.args.global_voxel_inc / self.unit_factor
            self.args.global_origin = [x/self.unit_factor for x in self.args.global_origin]


        if self.args.input_image_type == 'color_images':
            self.rotated_views_global = [6, 7]  # hardcoded for 8-view setup
            self.non_rotated_views_global = [i for i in range(8) if i not in self.rotated_views_global]
        elif self.args.input_image_type == 'stereo_images':
            self.rotated_views_global = [12, 13, 14, 15]  # no rotated views in stereo setup
            self.non_rotated_views_global = [i for i in range(16) if i not in self.rotated_views_global]
            
            
        self.dense_landmarks_weights_mask = torch.zeros((5023,), dtype=torch.float32).to(self.device)
        for key in self.vertex_masks_tempeh.keys():
            if 'vertex_count' in key:
                continue
            for idx in self.vertex_masks_tempeh[key].tolist():
                self.dense_landmarks_weights_mask[idx] = self.args.dense_mask_weights.get(key)
        self.dense_landmarks_weights_mask = self.dense_landmarks_weights_mask.unsqueeze(0).unsqueeze(0)  # (1,1,V)
        print(sum(self.dense_landmarks_weights_mask > 0.0), 'vertices with dense landmarks supervision')
        

        mp_indices_for_loss = np.arange(0,105).tolist()
        indices_to_ignore = [17,10,14,12,18,11,16,15,19,13,3,9,6,5,8,1,2,4,7,
                    52,53,54,55,56,57,58,59,60,61,62,63,64]

        self.mp_indices_for_loss = [x for x in mp_indices_for_loss if x not in indices_to_ignore]



    def register_model(self):
        import models.model_aligner.prototypes.model_global_stage as models_global
        # import models.model_aligner.prototypes.model_global_stage_dino as models
        model = models_global.Model(args=self.args)
        # model.initialize(init_method='normal')
        self.model = model.to(self.device)

        self.base_model = models_global.Model(self.args).to(self.device)
        self.base_model = self.base_model.to(self.device)
            
        if self.args.pretrained_path:
            cp = torch.load(self.args.pretrained_path, map_location=self.device)
            self.model.load_state_dict(cp['model'], strict=True)
            self.base_model.load_state_dict(cp['model'], strict=True)
        
        self.model = torch.nn.DataParallel(self.model)

        base_params, special_params = [], []
        for name, param in self.model.named_parameters():
            if 'grid_refiner' in name:
                special_params.append(param)
            else:
                base_params.append(param)

        self.optimizer_model = torch.optim.AdamW([
                                    {'params': base_params, 'lr': self.args.learning_rate, 'group_id': 'base'}, 
                                    {'params': special_params, 'lr': 5e-5, 'group_id': 'special'}])
        # count. parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total model parameters: {total_params/1e6:.2f} Mio')
        # trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable model parameters: {trainable_params/1e6:.2f} Mio')

        if self.args.enable_smirk_loss:
            from models.smirk.smirk_generator import SmirkGenerator
            if self.args.image_resize_factor == 1:
                self.fuse_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(self.device)
            else:
                self.fuse_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5, input_size=(150,200)).to(self.device)

            self.optimizer_fuse = torch.optim.AdamW(self.fuse_generator.parameters(), lr=1e-3)
            
            if self.args.pretrained_fuse:
                cp = torch.load(self.args.pretrained_fuse, map_location=self.device)
                self.fuse_generator.load_state_dict(cp['model'], strict=True)

        if self.args.enable_local:
            import models.model_aligner.prototypes.model_local_stage as models
            from models.model_aligner import FeatureNet2D

            feature_net = FeatureNet2D(input_ch=3, output_ch=self.args.descriptor_dim)
            feature_net = feature_net.to(self.device)

            feature_net.load_state_dict(self.model.module.feature_net.state_dict(), strict=True)

            model = models.Model(args=self.args, mesh_sampler=self.mesh_sampler, feature_net=feature_net)
            model.initialize(init_method='normal')
            model = model.to(self.device)

            self.base_local_model = models.Model(args=self.args, mesh_sampler=self.mesh_sampler, feature_net=feature_net)
            self.base_local_model = self.base_local_model.to(self.device)

            self.local_model = model


            base_params, special_params = [], []
            for name, param in self.local_model.named_parameters(): 
                if 'grid_refiner' in name:
                    special_params.append(param)
                else:
                    base_params.append(param)
         
            self.optimizer_model_local = torch.optim.AdamW([
                                        {'params': base_params, 'lr': self.args.learning_rate, 'group_id': 'base'}, 
                                        {'params': special_params, 'lr': 1e-4, 'group_id': 'special'}])

            if self.args.pretrained_local_path:
                cp = torch.load(self.args.pretrained_local_path, map_location=self.device)
                # remove module.
                cp_filtered = {}
                for k in cp['model']:
                    if k.startswith('module.'):
                        cp_filtered[k[7:]] = cp['model'][k]
                    else:
                        cp_filtered[k] = cp['model'][k]
                self.local_model.load_state_dict(cp_filtered, strict=True)
                self.base_local_model.load_state_dict(cp_filtered, strict=True)
                print('Loaded pretrained LOCAL model from %s' % self.args.pretrained_local_path)

            self.base_local_model = torch.nn.DataParallel(self.base_local_model)
            self.local_model = torch.nn.DataParallel(self.local_model)



            tempeh_model = models_global.Model(args=self.args)
            # model.initialize(init_method='normal')
            self.tempeh_model = tempeh_model.to(self.device)

            tempeh_checkpoint = "../TEMPEH/runs/coarse/coarse__TEMPEH_coarse_part_1_color__October11__14-24-16/checkpoints/model_00800000.pth"
            cp = torch.load(tempeh_checkpoint, map_location=self.device)
            self.tempeh_model.load_state_dict(cp['model'], strict=True)
            self.tempeh_model = self.tempeh_model.to(self.device)

            tempeh_local_checkpoint = "../TEMPEH/runs/refinement/refinement__TEMPEH__November01__15-51-22/checkpoints/model_00150000.pth"
            self.tempeh_local_model = models.Model(args=self.args, mesh_sampler=self.mesh_sampler, feature_net=None)
            cp = torch.load(tempeh_local_checkpoint, map_location=self.device)
            self.tempeh_local_model.load_state_dict(cp['model'], strict=True)
            self.tempeh_local_model = self.tempeh_local_model.to(self.device)


    def forward_tempeh(self):
        random_grid = False

        with torch.inference_mode():
            self.tempeh_model.eval()
            self.tempeh_local_model.eval()
            self.coarse_results_tempeh = self.tempeh_model(**self.inputs_coarse, random_grid=False)
            self.coarse_points_tempeh = self.coarse_results_tempeh['vertices']

            print(self.inputs['images'].device, self.inputs['camera_intrinsics'].device,
                  self.inputs['images'].device, self.inputs['camera_intrinsics'].device,
                  self.coarse_results_tempeh['vertices'].device, self.inputs['camera_distortions'].device, self.camera_centers.device  )

            results = self.tempeh_local_model(  images=self.inputs['images'],
                                        camera_intrinsics=self.inputs['camera_intrinsics'],
                                        camera_extrinsics=self.inputs['camera_extrinsics'],
                                        camera_distortions=self.inputs['camera_distortions'],
                                        camera_centers=self.camera_centers,
                                        global_points=self.coarse_results_tempeh['vertices'],
                                        random_grid=random_grid)

            self.global_points_tempeh = results[-1]


    def register_dataset(self, index=0):
        from utils import mesh_sampling, utils

        self.split_list = utils.load_json(self.args.train_data_list_fname)

        # # save tmp
        import json
        # os.makedirs(os.path.join(self.directory_output, mode + '_images'), exist_ok=True)

        # json_out = f"assets/tmp.json"
        # create a unique uuid for this complete run
        import uuid
        if hasattr(self, 'run_uuid'):
            run_uuid = getattr(self, 'run_uuid')
        else:
            self.run_uuid = uuid.uuid4().hex

        json_out = os.path.join(self.directory_output, f"temp_data_{self.run_uuid}.json")
        with open(json_out, 'w') as f:
            json.dump(self.split_list[index:index+1], f)


        from datasets.face_align_dataset_mpi_grid import FaceAlignDatasetMPI as DatasetCls

        common_kwargs = dict(
            dataset_root_dir=self.args.dataset_directory,
            image_dir=self.args.image_directory,
            calibration_dir=self.args.calibration_directory,
            scan_dir=self.args.scan_directory,
            return_full_scan="meshes" in self.args.scan_directory,
            registration_root_dir=self.args.processed_directory,
            image_resize_factor=self.args.image_resize_factor,
            mesh_sampler=None,
            scan_vertex_count=self.args.scan_vertex_count,
            brightness_sigma=self.args.brightness_sigma,
            load_stereo_images=self.args.input_image_type == 'stereo_images',
            load_color_images=self.args.input_image_type == 'color_images',
            image_file_ext=self.args.image_file_ext,
            fan_landmarks_dir='',
            mediapipe_landmarks_dir='',
            dense_landmarks_dir=self.args.dense_landmarks_dir,
            dense_semantic_landmarks_dir=self.args.dense_semantic_landmarks_dir,
            normals_dir=self.args.normals_image_directory,
            to_meters=self.args.to_meters,
            depths_dir=self.args.depths_image_directory,
            output_image_height=None,
            output_image_width=None,
        )

        self.dataset_train = DatasetCls(
            data_list_fname=json_out,
            **common_kwargs
        )

        self.dataset_val = DatasetCls(
            data_list_fname=json_out,
            **common_kwargs
        )

        self.dataloader_train = self.make_data_loader(self.dataset_train, cuda=True, shuffle=True)
        self.dataloader_val = self.make_data_loader(self.dataset_val, cuda=True, shuffle=False)

        print(len(self.dataloader_train), 'train samples')
        print(len(self.dataloader_val), 'validation samples')


    def register_losses(self):
        vertex_masks = np.load(self.args.vertex_mask_fname)
        sample_mesh = self.mesh_sampler.get_mesh(-1)

        self.points_loss_function = PointToPointLoss(num_vertices=sample_mesh.v.shape[0],
                            vertex_masks=vertex_masks, mask_weights=self.args.point_mask_weights,
                            mesh_sampler=self.mesh_sampler,
                            loss_function=torch.nn.MSELoss())

        self.points2surface_loss_function = PointToSurfaceLoss(gmo_sigma=self.args.gmo_sigma)
        self.edge_loss_function = EdgeLoss( num_vertices=sample_mesh.v.shape[0], faces=sample_mesh.f, 
                                            vertex_masks=vertex_masks, mask_weights=self.args.edge_mask_weights,
                                            mesh_sampler=self.mesh_sampler)

        from models.losses.VGGPerceptualLoss import VGGPerceptualLoss
        self.vgg_loss = VGGPerceptualLoss().to(self.device)
        self.vgg_loss.eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad_(False)


    def feed_data(self, data, mode='train'):
        if mode == 'train':
            suffix = '_augmented'
        else:
            suffix = ''

        print('suffx', suffix)

        number_views = data['color_images'].shape[1]
        images = data['color_images' + suffix]
        camera_intrinsics = data['color_camera_intrinsics' + suffix]
        # print(camera_intrinsics[0])
        camera_extrinsics = data['color_camera_extrinsics']

        # print(camera_extrinsics[0])
        camera_distortions = data['color_camera_distortions'] 
        camera_centers = data['color_camera_centers']
        normals_images = data['color_images_normals' + suffix]
        depth_maps = data['color_images_depth' + suffix]

        self.current_subjects = data['subject']
        self.current_sequences = data['sequence']
        self.current_frames = data['frame']

        if self.args.input_image_type == 'stereo_images':
            if self.args.sample_views and self.model.training:
                views = get_subset_views(number_views, minimum_views=self.args.minimum_sample_views)
            else:
                views = np.arange(number_views)
        else:
            views = np.arange(number_views)

        # positions in the subsampled 'views' array
        self.rotated_views = [i for i, v in enumerate(views) if v in self.rotated_views_global]
        self.non_rotated_views = [i for i, v in enumerate(views) if v not in self.rotated_views_global]


        self.visualization_view_ids = views
        self.current_batch_view_ids = views
        self.data = data
        self.inputs = {
            'images': Variable(images[:,views,...]).to(self.device),
            'camera_intrinsics': Variable(camera_intrinsics[:,views,...]).to(self.device),
            'camera_extrinsics': Variable(camera_extrinsics[:,views,...]).to(self.device),
            'camera_distortions': Variable(camera_distortions[:,views,...]).to(self.device),
        }
        self.camera_centers = Variable(camera_centers[:,views,...]).to(self.device)

        self.target_vertices = data['v_registration'].to(self.device)

        if depth_maps is not None:

            self.depth_maps_gt = Variable(depth_maps[:,views,...]).to(self.device) #/self.unit_factor
            if self.args.to_meters:
                self.depth_maps_gt = self.depth_maps_gt / self.unit_factor
            self.normal_maps_gt = Variable(normals_images[:,views,...]).to(self.device)

            self.normal_maps_gt_norm = self.normal_maps_gt / self.normal_maps_gt.norm(dim=2, keepdim=True).clamp(min=1e-12) * (self.depth_maps_gt.squeeze(-1) > 0.0).unsqueeze(2)

            self.normal_maps_gt_01 = (self.normal_maps_gt + 1.0) / 2.0 * (self.depth_maps_gt.squeeze(-1) > 0.0).unsqueeze(2)


            self.pointmaps_gt = depth_to_pointmap_robust(
                depth = self.depth_maps_gt.squeeze(-1),              # (B,V,H,W)
                K     = self.inputs['camera_intrinsics'],
                extr  = self.inputs['camera_extrinsics']).permute(0,1,4,2,3)  # (B,V,H,W,3)

        dense_key = 'color_camera_dense_landmarks' + suffix
        dense_mask_key = 'color_camera_dense_landmarks_masks' + suffix

        self.landmarks_dense = data[dense_key][:, views].to(self.device)
        if dense_mask_key in data and data[dense_mask_key] is not None:
            mask = data[dense_mask_key][:, views].to(self.device).squeeze(-1)
        else:
            mask = torch.ones(self.landmarks_dense.shape[:2], device=self.device, dtype=torch.float32)
        self.landmarks_dense_mask = mask


        self.landmarks_dense_uv = self.landmarks_dense.clone()
        self.landmarks_dense_uv[..., 0] = self.landmarks_dense_uv[..., 0]/(self.inputs['images'].shape[-1]-1) * 2.0 - 1.0
        self.landmarks_dense_uv[..., 1] = self.landmarks_dense_uv[..., 1]/(self.inputs['images'].shape[-2]-1) * 2.0 - 1.0


        dense_fan_landmarks_key = 'color_camera_dense_fan_landmarks' + suffix
        self.landmarks_dense_fan = data[dense_fan_landmarks_key][:, views].to(self.device)
        dense_mediapipe_landmarks_key = 'color_camera_dense_mediapipe_landmarks' + suffix
        self.landmarks_dense_mediapipe = data[dense_mediapipe_landmarks_key][:, views].to(self.device)
        print(self.landmarks_dense_fan.shape, 'dense fan landmarks shape', self.landmarks_dense_mediapipe.shape, 'dense mediapipe landmarks shape')

        self.landmarks_dense_fan_uv = self.landmarks_dense_fan.clone()
        self.landmarks_dense_fan_uv[..., 0] = self.landmarks_dense_fan_uv[..., 0]/(self.inputs['images'].shape[-1]-1) * 2.0 - 1.0
        self.landmarks_dense_fan_uv[..., 1] = self.landmarks_dense_fan_uv[..., 1]/(self.inputs['images'].shape[-2]-1) * 2.0 - 1.0

        self.landmarks_dense_mediapipe_uv = self.landmarks_dense_mediapipe.clone()
        self.landmarks_dense_mediapipe_uv[..., 0] = self.landmarks_dense_mediapipe_uv[..., 0]/(self.inputs['images'].shape[-1]-1) * 2.0 - 1.0
        self.landmarks_dense_mediapipe_uv[..., 1] = self.landmarks_dense_mediapipe_uv[..., 1]/(self.inputs['images'].shape[-2]-1) * 2.0 - 1.0

        self.global_landmarks_projected_dense = None

        if self.args.enable_local:
            # for coarse, downsample the images
            suffix = ''
            BB, L, C, H, W = data['color_images' + suffix].shape
            flattened_images = data['color_images' + suffix].to(self.device).view(BB*L, C, H, W).clone()
            downsampled_images = F.interpolate(flattened_images, scale_factor=0.5, mode='bilinear', align_corners=False)
            downsampled_images = downsampled_images.view(BB, L, C, H//2, W//2)

            scale = 0.5 
            downscaled_intrinsics = data['color_camera_intrinsics' + suffix].to(self.device).clone()
            downscaled_intrinsics[..., 0, 0] *= scale  # fx
            downscaled_intrinsics[..., 0, 1] *= scale  # fy
            downscaled_intrinsics[..., 1, 0] *= scale  # fy
            downscaled_intrinsics[..., 1, 1] *= scale  # fy

            downscaled_intrinsics[..., 0, 2] *= scale  # cx
            downscaled_intrinsics[..., 1, 2] *= scale  # cy

            self.inputs_coarse = {
                'images': downsampled_images,
                'camera_intrinsics': downscaled_intrinsics,
                'camera_extrinsics': data['color_camera_extrinsics'].to(self.device),
                'camera_distortions': data['color_camera_distortions'].to(self.device),
            }



    def forward(self, model=None, local_model=None):
        random_grid = False

        with torch.inference_mode():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.coarse_results = model(**self.inputs_coarse, random_grid=False)
            self.coarse_points = self.coarse_results['vertices']

        results = local_model(  images=self.inputs['images'],
                                    camera_intrinsics=self.inputs['camera_intrinsics'],
                                    camera_extrinsics=self.inputs['camera_extrinsics'],
                                    camera_distortions=self.inputs['camera_distortions'],
                                    camera_centers=self.camera_centers,
                                    global_points=self.coarse_results['vertices'],
                                    random_grid=random_grid)

        self.global_points = results[-1]

        # self.global_points = self.coarse_points



        self.unit_factor = 1000.0

        if getattr(self.args, 'use_pliks_refinement', False):
            # Run PLIKS refinement strictly in float32 to avoid AMP/bfloat16 instabilities
            self.unit_factor 
            pliks_in = (self.global_points).to(dtype=torch.float32) 

            if not self.args.to_meters:
                pliks_in = pliks_in/self.unit_factor

            self.pliks_out = self.pliks_solver(
                pliks_in, iters=1, lsq_method='ne', estimate_root=False
            )  # dict: beta, t, Rk, V_fit
            # keep outputs as float32; convert back to mm for V_fit
            self.pliks_out['V_fit'] = self.pliks_out['V_fit'].to(dtype=torch.float32) 
            if not self.args.to_meters:
                self.pliks_out['V_fit'] = self.pliks_out['V_fit'] * self.unit_factor  # convert back to mm if needed


            beta_all = self.pliks_out['beta']  # (B, n_shape+n_exp)
            n_id, n_exp = self.flame.n_shape, self.flame.n_exp
            beta_id  = beta_all[:, :n_id]
            beta_exp = beta_all[:, n_id:n_id+n_exp]

            self.beta_reg = (beta_id  ** 2).mean()
            self.exp_reg  = (beta_exp ** 2).mean()


            t_hat    = self.pliks_out['t']                                          # [B, 3]
            Rk       = self.pliks_out['Rk']                                         # [B, K, 3, 3]

            V_fit = self.pliks_out['V_fit']

            from models.FLAME.pliks_flame import build_R_world_from_segments, mat_to_axis_angle, world_to_relative_rotations
            J     = self.flame.J_regressor.shape[0]

            R_world = build_R_world_from_segments(Rk, self.pliks_solver.seg_list, J)   # [B,J,3,3]
            R_rel   = world_to_relative_rotations(R_world, self.flame.parents)   # [B,J,3,3]

            # swapped with the pytorch3d version
            from pytorch3d.transforms import matrix_to_axis_angle
            aa_all = matrix_to_axis_angle(R_rel)

            # # pick only stable joints (good curriculum): neck + jaw + eyes
            NECK_ID = 1
            # # TODO: fill your indices once you confirm them
            JAW_ID, LEYE_ID, REYE_ID = 2, 3, 4

            pose_params      = torch.zeros(beta_all.shape[0], 3, device=beta_all.device)          # global/root
            neck_pose_params = aa_all[:, NECK_ID]
            if self.no_jaw:
                jaw_params = torch.zeros_like(aa_all[:, JAW_ID])
            else:
                jaw_params       = aa_all[:, JAW_ID]

            eye_pose_params  = torch.cat([aa_all[:, LEYE_ID], aa_all[:, REYE_ID]], dim=-1)

            def rigid_align_no_scale(X: torch.Tensor, Y: torch.Tensor):
                """
                Solve R,t (no scale) s.t. R X + t ≈ Y   using SVD (batched).
                X,Y: [B,V,3]
                Returns:
                R: [B,3,3], t: [B,3]
                """
                Xc = X - X.mean(dim=1, keepdim=True)
                Yc = Y - Y.mean(dim=1, keepdim=True)
                H  = Xc.transpose(1,2) @ Yc                         # [B,3,3]
                U, S, Vt = torch.linalg.svd(H)                      # Vt=V^T
                R = Vt.transpose(1,2) @ U.transpose(1,2)            # [B,3,3]
                # reflection handling
                det = torch.det(R)
                if (det < 0).any():
                    Vt_fix = Vt.clone()
                    Vt_fix[det < 0, -1, :] *= -1
                    R[det < 0] = Vt_fix[det < 0].transpose(1,2) @ U[det < 0].transpose(1,2)
                t = Y.mean(dim=1) - (R @ X.mean(dim=1).unsqueeze(-1)).squeeze(-1)  # [B,3]
                return R, t


            out = self.flame.forward({
                'shape_params':      beta_id,
                'expression_params': beta_exp,
                'pose_params':       pose_params,
                'neck_pose_params':  neck_pose_params,
                'jaw_params':        jaw_params,
                'eye_pose_params':   eye_pose_params,
            })
            V_flame = out['vertices'] #+ t_hat[:, None, :]   # meters
            fan_landmarks_pliks = out['landmarks_fan_3d']  # pixels, projected inside FLAME module
            mediapipe_landmarks_pliks = out['landmarks_mp']
            # V_flame_mm = V_flame * 1000.0    
            # do float32 
            with torch.cuda.amp.autocast(enabled=False):
                if self.args.to_meters:
                    global_points = self.global_points.float() 
                else:
                    global_points = self.global_points.float() / self.unit_factor

                R_root, t_root = rigid_align_no_scale(V_flame.float(), global_points)

            V_flame_mm = (R_root[:,None] @ V_flame.unsqueeze(-1)).squeeze(-1) + t_root[:,None,:]

            if not self.args.to_meters:
                V_flame_mm = V_flame_mm*self.unit_factor

            self.pliks_out['V_flame_mm'] = V_flame_mm

    

    def compute_vertex_losses(self, vertices, target_vertices=None, suffix=""):
        losses = {}

        registrations_are_here = False
        if target_vertices is None:
            registrations_are_here = True
            print('Using self.target_vertices for vertex losses computation')
            print('Current weight for points and edge and laplacian:', self.args.weight_points_recon, self.args.weight_edge_regularizer, self.args.weight_laplacian_smoothing)
            target_vertices = self.target_vertices

        # Points reconstruction loss
        if self.args.weight_points_recon > 0.0:
            points_loss = self.args.weight_points_recon * self.points_loss_function(vertices, target_vertices)
            losses[f'points_loss{suffix}'] = points_loss
        else:
            losses[f'points_loss{suffix}'] = 0.0

        # Edge regularizer loss
        if self.args.weight_edge_regularizer > 0.0:
            edge_loss = self.args.weight_edge_regularizer * self.edge_loss_function(vertices, target_vertices)
            losses[f'edge_regularizer_loss{suffix}'] = edge_loss 
        else:
            losses[f'edge_regularizer_loss{suffix}'] = 0.0
        
        # Laplacian smoothing loss
        smoothed_vertices = uniform_laplacian_smoothing(vertices, self.faces.to(self.device))
        laplacian_coords = vertices - smoothed_vertices
        laplacian_loss = laplacian_coords.pow(2).mean() * self.args.weight_laplacian_smoothing
        losses[f'laplacian_smoothing_loss{suffix}'] = laplacian_loss

        points2surface_loss = self.args.weight_points2surface * self._points2surface_loss_for_vertices(vertices)
        losses['points2surface_loss' + suffix] = points2surface_loss

        # assert that all our losses are 0 if registrations_are_here is True
        if registrations_are_here:
            for key in losses:
                assert (losses[key] == 0.0), f"Loss {key} is not zero despite using registrations: {losses[key].item()}"

        return losses

    def compute_rendering_losses(self, vertices, suffix=""):
        suffix_label = suffix.strip('_') if suffix else 'main'
        suffix_label = suffix_label or 'main'

        bs, num_views, c, height, width = self.inputs['images'].shape
        losses = {}
        predictions = {}
                
        if not self.args.enable_diff_rendering:
            return losses, predictions
        
        visibility_mask = (self.depth_maps_gt[:, :, :, :].squeeze(-1) > 0.0).unsqueeze(2)
        
        try:
            with torch.cuda.amp.autocast(enabled=False):
                # print(vertices.shape, self.inputs['camera_intrinsics'].shape, self.inputs['camera_extrinsics'].shape, 'sfd')
                pred = self.mesh_helper.render_lots_of_stuff(
                    vertices,
                    self.inputs['camera_intrinsics'],
                    self.inputs['camera_extrinsics'],
                    radial_distortions=self.inputs['camera_distortions'],
                    depth_rendering_height=height,
                    depth_rendering_width=width,
                    return_depth=True,
                    normalize_normals=False
                )

        except Exception as e:
            print('Rendering failed:', e)
            return losses, predictions
        
        # Extract rendered outputs
        normal_maps_pred = pred['normal_images'].permute(0, 3, 1, 2).view(bs, num_views, c, height, width)
        
        normal_maps_pred_11 = normal_maps_pred * 2.0 - 1.0  # to [-1,1]
        # normalize
        normal_maps_pred_norm = normal_maps_pred_11 / (normal_maps_pred_11.norm(dim=2, keepdim=True) + 1e-12) * (pred['depth_images'].view(bs, num_views, 1, height, width) > 0.0)

        depth_maps_pred = pred['depth_images'].view(bs, num_views, height, width)
        pointmaps_pred = (
            depth_to_pointmap_robust(
                depth=depth_maps_pred,
                K=self.inputs['camera_intrinsics'],
                extr=self.inputs['camera_extrinsics']
            )
            .permute(0, 1, 4, 2, 3)
        )
        
        predictions[f'normal_maps_pred{suffix}'] = normal_maps_pred
        predictions[f'depth_maps_pred{suffix}'] = depth_maps_pred
        predictions[f'pointmaps_pred{suffix}'] = pointmaps_pred
        
        # Compute rendering losses
        normals_loss, normals_loss_per_pixel = calculate_map_loss(
            normal_maps_pred_norm.view(bs * num_views, c, height, width),
            self.normal_maps_gt_norm.view(bs * num_views, c, height, width),
            mask=visibility_mask.view(bs * num_views, height, width),
            robust=True,
            gmo_sigma=10
        )

        # # normal_maps_pred: (B, V, 3, H, W) in [-1,1], already renormalized in your code
        # loss_geo, angle_map = geodesic_normal_loss_p3d(
        #     normal_maps_pred,
        #     self.normal_maps_gt,               # (B, V, 3, H, W)
        #     mask=visibility_mask,              # (B, V, H, W)
        #     reduction="mean",
        #     return_per_pixel=True
        # )
        # losses[f'normal_geo_loss{suffix}'] = loss_geo * getattr(self.args, 'weight_normals_images_geo', 1.0)
        # predictions[f'normal_geo_angle_map{suffix}'] = angle_map  # radians

        # --- NEW: gradient loss on normal maps ---
        normals_pred_flat = normal_maps_pred_norm.view(bs * num_views, c, height, width)
        normals_gt_flat   = self.normal_maps_gt_norm.view(bs * num_views, c, height, width)

        normals_grad_loss, normals_grad_pp, grad_mag_pred, grad_mag_gt = \
            calculate_gradient_map_loss(
                normals_pred_flat, normals_gt_flat,
                mask=visibility_mask.view(bs * num_views, height, width),
                robust=True, gmo_sigma=10
            )

        # weight via self.args.weight_normals_grad (add this arg; see §4)
        losses[f'normals_grad_loss{suffix}'] = normals_grad_loss * getattr(self.args, 'weight_normals_grad')


        # store gradient magnitudes for visualization
        predictions[f'normals_grad_mag_pred{suffix}'] = grad_mag_pred.view(bs, num_views, height, width)
        predictions[f'normals_grad_mag_gt{suffix}']   = grad_mag_gt.view(bs, num_views, height, width)
        # --- END NEW ---


        depth_maps_loss, depth_maps_loss_per_pixel = calculate_map_loss(
            depth_maps_pred.view(bs * num_views, 1, height, width),
            self.depth_maps_gt.view(bs * num_views, 1, height, width),
            mask=visibility_mask.view(bs * num_views, 1, height, width),
            robust=True,
            gmo_sigma=10
        )
        
        # Scale Z and compute point-map loss
        p_pred = pointmaps_pred.clone()
        p_gt = self.pointmaps_gt.clone()
        p_pred[:, :, 2] /= 30
        p_gt[:, :, 2] /= 30
        
        point_maps_loss, point_maps_loss_per_pixel = calculate_map_loss(
            p_pred.view(bs * num_views, 3, height, width),
            p_gt.view(bs * num_views, 3, height, width),
            mask=visibility_mask.view(bs * num_views, height, width),
            robust=True,
            gmo_sigma=10
        )
        
        # Apply weighting
        losses[f'normals_loss{suffix}'] = normals_loss * self.args.weight_normals_images
        losses[f'depth_maps_loss{suffix}'] = depth_maps_loss * self.args.weight_depth_maps
        losses[f'point_maps_loss{suffix}'] = point_maps_loss * self.args.weight_point_maps
        
        # Store per-pixel losses for visualization
        # if (self.global_step % self.args.visualize_frequency == 0) and (self.global_step > 0):
        setattr(self, f'normals_loss_per_pixel{suffix}', normals_loss_per_pixel.view(bs, num_views, height, width).detach().cpu())
        setattr(self, f'depth_maps_loss_per_pixel{suffix}', depth_maps_loss_per_pixel.view(bs, num_views, height, width).detach().cpu())
        setattr(self, f'point_maps_loss_per_pixel{suffix}', point_maps_loss_per_pixel.view(bs, num_views, height, width).detach().cpu())
    
        # store per-pixel loss for visualization
        setattr(self, f'normals_grad_loss_per_pixel{suffix}',
                normals_grad_pp.view(bs, num_views, height, width).detach().cpu())

        # SMIRK loss computation
        if self.args.enable_smirk_loss and normal_maps_pred is not None:
            # smirk_losses, smirk_predictions = self.compute_smirk_loss(normal_maps_pred_11, visibility_mask, suffix)
            smirk_losses, smirk_predictions = self.compute_smirk_loss(normal_maps_pred, visibility_mask, suffix) # 01 is for the 300-400 smirk model, 11 for the 150-200
            losses.update(smirk_losses)
            predictions.update(smirk_predictions)


        return losses, predictions

    def compute_smirk_loss(self, normal_maps_pred, visibility_mask, suffix=""):
        """
        Compute SMIRK loss for given normal maps.
        """
        bs, num_views, c, height, width = self.inputs['images'].shape
        losses = {}
        predictions = {}
        
        self.fuse_generator.eval()
        visibility_mask_flat = visibility_mask.view(bs * num_views, height, width)
        self.visibility_mask_flat = visibility_mask_flat #.detach().cpu()
        inverse_visibility_mask = 1.0 - visibility_mask_flat.float()
        
        gt_images = self.inputs['images'].view(bs * num_views, c, height, width) 
        # denormalize 
        gt_images = gt_images * self.dataset_train.std.to(self.device).view(1, c, 1, 1) + self.dataset_train.mean.to(self.device).view(1, c, 1, 1)
        # im = self.dataset_train.denormalize_image(im)
        # print(gt_images.min(), gt_images.max(), 'gt images stats for smirk')

        masked_images, _ = random_pixel_mask(gt_images, visibility_mask=inverse_visibility_mask)
        
        pred_maps = torch.cat([masked_images, normal_maps_pred.view(bs * num_views, c, height, width)], dim=1)

        if self.args.image_resize_factor == 2:
            pred_maps = torch.nn.functional.pad(pred_maps, (8,0,10,0), mode='constant', value=0)

        pred_fused_images = self.fuse_generator(pred_maps)

        if self.args.image_resize_factor == 2:
            pred_fused_images = pred_fused_images[:, :, 10:, 8:]

        pred_fused_images = pred_fused_images.view(bs, num_views, c, height, width)

        gt_maps = torch.cat([masked_images, self.normal_maps_gt_01.view(bs * num_views, c, height, width)], dim=1)

        if self.args.image_resize_factor == 2:
            gt_maps = torch.nn.functional.pad(gt_maps, (8,0,10,0), mode='constant', value=0)

        gt_fused_images = self.fuse_generator(gt_maps)

        if self.args.image_resize_factor == 2:
            gt_fused_images = gt_fused_images[:, :, 10:, 8:]

        gt_fused_images = gt_fused_images.view(bs, num_views, c, height, width)

        gt_images = gt_images.view(bs, num_views, c, height, width)

        # get the area 
        # eye_region_landmarks = self.global_landmarks_projected_dense_out[:,:,self.flame_masks['eye_region']]


        # smirk_loss_l1 = torch.nn.functional.l1_loss(pred_fused_images[:,self.non_rotated_views], gt_images[:,self.non_rotated_views]) * 100
        # smirk_loss_vgg = self.vgg_loss(pred_fused_images[:,self.non_rotated_views], gt_images[:,self.non_rotated_views], needs_normalization=True)
        # smirk_loss = (smirk_loss_l1 + smirk_loss_vgg) * self.args.weight_smirk_loss

        # --- Build a per-view eye-region mask from dense landmarks (bbox) ---
        bs, num_views, c, height, width = self.inputs['images'].shape
        device = self.device

        # landmarks in pixel coords: (B, V, K, 2)
        eye_region_landmarks = self.global_landmarks_projected_dense_out[:, :, np.concatenate((self.flame_masks['left_eyeball'],self.flame_masks['right_eyeball']))]  # (B,V,K,2)

        # Compute bbox per (B,V)
        x = eye_region_landmarks[..., 0]
        y = eye_region_landmarks[..., 1]

        xmin = torch.clamp(torch.floor(x.min(dim=2).values).long(), 0, width  - 1)
        xmax = torch.clamp(torch.ceil( x.max(dim=2).values).long(), 0, width  - 1)
        ymin = torch.clamp(torch.floor(y.min(dim=2).values).long(), 0, height - 1)
        ymax = torch.clamp(torch.ceil( y.max(dim=2).values).long(), 0, height - 1)

        # Optional padding around the eyes (in pixels)
        pad = 0
        xmin = (xmin - pad).clamp(min=0)
        xmax = (xmax + pad).clamp(max=width - 1)
        ymin = (ymin - pad).clamp(min=0)
        ymax = (ymax + pad).clamp(max=height - 1)

        # Create binary mask (B,V,1,H,W), 1 inside bbox
        yy = torch.arange(height, device=device).view(1, 1, height, 1)
        xx = torch.arange(width,  device=device).view(1, 1, 1, width)

        # Broadcast comparisons per view
        in_y = (yy >= ymin.unsqueeze(-1).unsqueeze(-1)) & (yy <= ymax.unsqueeze(-1).unsqueeze(-1))
        in_x = (xx >= xmin.unsqueeze(-1).unsqueeze(-2)) & (xx <= xmax.unsqueeze(-1).unsqueeze(-2))
        eye_mask = (in_y & in_x).float().unsqueeze(2)  # (B,V,1,H,W)

        # If you only train on non-rotated views for SMIRK, we’ll index those later.
        # --- Masked L1 on images (non-rotated views only) ---
        pred_imgs = pred_fused_images[:, self.non_rotated_views]     # (B, Vnr, C, H, W)
        gt_imgs   = gt_images[:, self.non_rotated_views]             # (B, Vnr, C, H, W)
        mask_imgs = eye_mask[:, self.non_rotated_views]              # (B, Vnr, 1, H, W)

        # Standardize shapes
        diff = torch.abs(pred_imgs - gt_imgs)                        # (B, Vnr, C, H, W)
        num = (diff * mask_imgs).sum()
        den = (mask_imgs.sum() * diff.shape[2]).clamp_min(1e-8)      # multiply by channels
        smirk_loss_l1 = num / den

        # --- Per-pixel map for SMIRK L1 (masked), like your normals/pointmap maps ---
        # diff: (B, Vnr, C, H, W), mask_imgs: (B, Vnr, 1, H, W)
        l1_map = diff.mean(dim=2) * mask_imgs.squeeze(2)          # (B, Vnr, H, W)
        self.smirk_l1_loss_per_pixel = l1_map.detach().cpu()


        # --- Masked VGG perceptual loss over the same area ---
        # Flatten views to batch for the perceptual loss
        pred_vgg_in = pred_imgs.reshape(-1, c, height, width)
        gt_vgg_in   = gt_imgs.reshape(-1, c, height, width)
        mask_vgg_in = mask_imgs.reshape(-1, 1, height, width)

        smirk_loss_vgg = self.vgg_loss(pred_vgg_in, gt_vgg_in, needs_normalization=True, mask=mask_vgg_in)

        smirk_loss_vgg, vgg_map = self.vgg_loss(
            pred_vgg_in, gt_vgg_in, needs_normalization=True, mask=mask_vgg_in, return_map=True
        )
        # vgg_map: (B*Vnr, H, W) -> reshape to (B, Vnr, H, W)
        smirk_vgg_map = vgg_map.view(bs, -1, height, width)
        self.smirk_vgg_loss_per_pixel = smirk_vgg_map.detach().cpu()

        # Total SMIRK loss
        smirk_loss = (smirk_loss_l1 + smirk_loss_vgg) * self.args.weight_smirk_loss


        losses[f'smirk_loss{suffix}'] = smirk_loss
        losses[f'smirk_loss_l1{suffix}'] = smirk_loss_l1
        losses[f'smirk_loss_vgg{suffix}'] = smirk_loss_vgg
        
        predictions[f'pred_fused_images{suffix}'] = pred_fused_images.detach().cpu()
        predictions[f'gt_fused_images{suffix}'] = gt_fused_images.detach().cpu()
        predictions[f'masked_images{suffix}'] = masked_images.view(bs, num_views, c, height, width).detach().cpu()
        
        
        return losses, predictions

    def compute_landmarks_loss(self, vertices, suffix="", gt_landmarks=None, gt_mask=None):
        """
        Compute landmarks loss for given vertices.
        """
        suffix_lower = suffix.lower()
        loss_weight = self.args.weight_landmarks
        if "dense" in suffix_lower:
            loss_weight = getattr(self.args, 'weight_dense_landmarks', 0.0)

        # print(loss_weight, suffix)
        if loss_weight <= 0.0 and "dense" not in suffix_lower:
            return torch.tensor(0.0, device=vertices.device), None
        
        num_views = self.inputs['camera_extrinsics'].shape[1]
        vertices_expanded = vertices.unsqueeze(1).repeat(1, num_views, 1, 1)
        vertices_flat = vertices_expanded.view(-1, vertices_expanded.shape[2], vertices_expanded.shape[3])
        
        if "mediapipe" in suffix_lower:
            landmarks3d = self.flame.select_mediapipe(vertices_flat)
        elif "dense" in suffix_lower:
            landmarks3d = vertices_flat
        else:
            landmarks3d = self.flame.seletec_3d68(vertices_flat)

        bs, _, _, height, width = self.data['color_images_augmented'].shape
        
        camera_extrinsics = self.inputs['camera_extrinsics'].view(-1, self.inputs['camera_extrinsics'].shape[2], self.inputs['camera_extrinsics'].shape[3])
        camera_intrinsics = self.inputs['camera_intrinsics'].view(-1, self.inputs['camera_intrinsics'].shape[2], self.inputs['camera_intrinsics'].shape[3])
        camera_distortions = self.inputs['camera_distortions'].view(-1, self.inputs['camera_distortions'].shape[2])
        

        from modules.volumetric_feature_sampler import VolumetricFeatureSampler

        landmarks_x, landmarks_y = VolumetricFeatureSampler.project(
            landmarks3d, camera_intrinsics, camera_extrinsics, camera_distortions,
            height=height, width=width
        )
        
        if self.args.to_meters:
            landmarks_projected = torch.stack((landmarks_x, landmarks_y), dim=-1)
            landmarks_projected = landmarks_projected.view(-1, num_views, landmarks_projected.shape[1], landmarks_projected.shape[2])

            # Inverse transform
            with torch.no_grad():
                landmarks_x_vis = (landmarks_x.detach() + 1.0) * (width - 1.) / 2.0
                landmarks_y_vis = (landmarks_y.detach() + 1.0) * (height - 1.) / 2.0
                landmarks_projected_vis = torch.stack((landmarks_x_vis, landmarks_y_vis), dim=-1)
                landmarks_projected_vis = landmarks_projected_vis.view(-1, num_views, landmarks_projected_vis.shape[1], landmarks_projected_vis.shape[2])
        else:
            landmarks_x_vis = (landmarks_x + 1.0) * (width - 1.) / 2.0
            landmarks_y_vis = (landmarks_y + 1.0) * (height - 1.) / 2.0
            landmarks_projected = torch.stack((landmarks_x_vis, landmarks_y_vis), dim=-1)
            landmarks_projected = landmarks_projected.view(-1, num_views, landmarks_projected.shape[1], landmarks_projected.shape[2])
            landmarks_projected_vis = landmarks_projected.clone()


        # Determine ground truth landmarks and mask
        # target_landmarks = self.landmarks if gt_landmarks is None else gt_landmarks
        # target_mask = self.landmarks_mask if gt_mask is None else gt_mask
        target_landmarks = gt_landmarks
        if gt_mask is not None:
            target_mask = gt_mask
        else:
            target_mask = None

        # # Compute per-point losses
        # if "mediapipe" in suffix_lower:
        #     pred_pts = landmarks_projected[:,self.non_rotated_views, :, :2]
        #     gt_pts = target_landmarks[:, self.non_rotated_views, :, :2]
        # Compute per-point losses
        if "mediapipe" in suffix_lower:
            # print(self.mp_indices_for_loss, landmarks_projected.shape, target_landmarks.shape, 'mp indices shape')
            pred_pts = landmarks_projected[:,self.non_rotated_views, :, :2]
            # print(pred_pts.shape,'aa')
            pred_pts = pred_pts[:, :, self.mp_indices_for_loss, :]
            # print(pred_pts.shape,'bb')
            gt_pts = target_landmarks[:, self.non_rotated_views, :, :2]
            gt_pts = gt_pts[:, :, self.mp_indices_for_loss, :]
            # L1 for this !
            diff_sq = torch.abs(pred_pts - gt_pts).sum(-1)
        elif "dense" in suffix_lower:
            # print(landmarks_projected.shape)
            pred_pts = landmarks_projected[:, self.non_rotated_views, :, :2]
            gt_pts = target_landmarks[:, self.non_rotated_views, :, :2]
            target_mask = target_mask[:, self.non_rotated_views]
            diff_sq = (pred_pts - gt_pts).pow(2).sum(-1)  # (B,V,N)
        
        else:
            pred_pts = landmarks_projected[:, self.non_rotated_views, :17, :2]
            gt_pts = target_landmarks[:, self.non_rotated_views, :17, :2]

        
        # calculate the relative and not absolute error


        # --- Relative (intra-set) landmark distance loss ---
        # pred_pts, gt_pts: (B, V, N, 2)

        B, Vv, N, _ = pred_pts.shape
        device = pred_pts.device
        eps = 1e-8

        # Pairwise Euclidean distances within each set (translation/rotation invariant)
        # Shapes: (B, V, N, N)
        D_pred = torch.cdist(pred_pts, pred_pts, p=2)
        D_gt   = torch.cdist(gt_pts,   gt_pts,   p=2)

        # Keep only upper triangle (exclude diagonal) to avoid double counting/self-pairs
        tri = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        # Shapes: (B, V, K) where K = N*(N-1)/2
        D_pred_ut = D_pred[..., tri]
        D_gt_ut   = D_gt[...,   tri]

        # Robust per-view scale (median non-zero GT pairwise distance) for scale invariance
        # scale: (B, V, 1)
        scale = D_gt_ut.median(dim=-1, keepdim=True).values.clamp_min(eps)

        # Normalize pairwise distances
        D_pred_rel = D_pred_ut / scale
        D_gt_rel   = D_gt_ut   / scale

        # Relative difference
        rel_diff = (D_pred_rel - D_gt_rel).abs()  # (B, V, K)

        # Optional hinge threshold in relative units (default 0.0 if not provided)
        tau_rel = getattr(self.args, 'landmarks_rel_tau', 0.0)
        if tau_rel > 0.0:
            rel_diff = torch.clamp(rel_diff - tau_rel, min=0.0)

        # Use L2 on the relative differences (change to .abs() for L1 if you prefer)
        loss_point = rel_diff.pow(2)               # (B, V, K)

        # Reduce over pairs -> per-view
        per_view_loss = loss_point.mean(dim=-1)    # (B, V)


        # diff_sq = (pred_pts - gt_pts).pow(2).sum(-1)  # (B,V,N)
        # # if "mediapipe" in suffix_lower:
        # #     print(diff_sq)
        # #     print(diff)

        # # diff_sq: (B, V, N)
        # diff = diff_sq.sqrt()  # Euclidean distance in pixels or normalized coords

        # # threshold τ (in same units as diff)
        # tau = 2.5 # getattr(self.args, 'landmarks_hinge_thresh', 0.0)
        # # print(diff.shape, 'diff', diff_sq.shape, 'dsq')
        # # print(diff)
        # # hinge: 0 if diff <= τ, (diff - τ) otherwise
        # hinge = torch.clamp(diff - tau, min=0.0)
        # # print(hinge.shape)

        # # choose your “shape” of the penalty:
        # #  - L1-like:      loss_point = hinge
        # #  - L2-like:      loss_point = hinge ** 2
        # #  - smooth-ish:   loss_point = hinge ** 2 / 2, etc.
        # loss_point = hinge ** 2     # (B, V, N)

        # loss_point = loss_point[loss_point > 0] 

        # # if no elements, then 0
        # if loss_point.numel() == 0:
        #     loss_point = torch.tensor(0.0, device=vertices.device)

        # per_point_loss = loss_point
        # per_view_loss  = per_point_loss.mean(-1)  # (B, V)
        # print(per_view_loss.shape, 'pp')

        # DENSE_LANDMARKS_WEIGHTS = {
        #     'face': 0.0,
        #     'ears': 0.0,
        #     'eyeballs': 0.0,
        #     'eye_region': 0.0,
        #     'lips': 100.0,
        #     'neck': 0.0,
        #     'nostrils': 0.0,
        #     'scalp': 0.0, # changed from 1.0 to 10.0
        #     'boundary': 0.0,
        #     'left_eyeball': 0.0,
        #     'right_eyeball': 0.0,
        #     'right_ear': 0.0,
        #     'right_eye_region': 0.0,
        #     'left_ear': 0.0,
        #     'left_eye_region': 0.0,
        # }


        # self.dense_landmarks_weights_mask = torch.zeros((5023,), dtype=torch.float32).to(self.device)
        # for key in self.vertex_masks_tempeh.keys():
        #     if 'vertex_count' in key:
        #         continue
        #     for idx in self.vertex_masks_tempeh[key].tolist():
        #         self.dense_landmarks_weights_mask[idx] = self.args.dense_mask_weights.get(key)
        # self.dense_landmarks_weights_mask = self.dense_landmarks_weights_mask.unsqueeze(0).unsqueeze(0)  # (1,1,V)
        # # print(sum(self.dense_landmarks_weights_mask > 0.0), 'vertices with dense landmarks supervision')

        # if "dense" in suffix_lower and getattr(self, 'dense_landmarks_weights_mask', None) is not None:
        #     # print('Applying dense landmarks weights mask')
        #     weights = self.dense_landmarks_weights_mask.to(diff_sq.device)
        #     diff_sq = diff_sq * weights


        # # -------- NEW: hard upweighting for big errors --------
        # thr_sq   = getattr(self.args, 'dense_landmarks_err_thresh_sq', 1.0)     # e.g., 3.0**2
        # hi_w     = getattr(self.args, 'dense_landmarks_high_weight', 100.0)     # e.g., 100.0

        # # elementwise multiplier: 100 if diff_sq > thr_sq, else 1
        # mult = torch.where(
        #     diff_sq > thr_sq,
        #     torch.as_tensor(hi_w, device=diff_sq.device, dtype=diff_sq.dtype),
        #     torch.as_tensor(1.0, device=diff_sq.device, dtype=diff_sq.dtype)
        # )

        # diff_sq = diff_sq * mult

        # print(diff_sq.shape)
        # # get the mean of nonzero
        # mean_nonzero = diff_sq[diff_sq > 0].mean()
        # print('Landmarks loss - mean nonzero:', mean_nonzero.item())

        # per_point_loss = diff_sq

        # per_view_loss = per_point_loss.mean(-1)  # (B,V)

        if target_mask is not None:
            mask = target_mask.to(per_view_loss.dtype)
            valid = mask.sum()
            # print(mask.shape)
            if valid.item() > 0:
                landmarks_loss = (per_view_loss * mask).sum() / valid
            else:
                landmarks_loss = torch.tensor(0.0, device=per_view_loss.device)
        else:
            landmarks_loss = per_view_loss.mean()
        
        landmarks_loss = landmarks_loss * loss_weight
        
        return landmarks_loss, landmarks_projected_vis

    def _points2surface_loss_for_vertices(self, vertices):
        """
        Compute points2surface loss for specific vertices.
        """
        if 'v_scan' not in self.data:
            print("No scan vertices available")
            return 0.0
        
        if isinstance(self.data['v_scan'], list):
            total_loss = 0.0
            for i in range(len(self.data['v_scan'])):
                scan_vertices = self.data['v_scan'][i].to(self.device).unsqueeze(0)
                predicted_vertices = vertices[i].unsqueeze(0).float()
                predicted_faces = self.data['f_reg_global'][0].to(self.device)
                loss = self.points2surface_loss_function(scan_vertices, predicted_vertices, predicted_faces)
                total_loss += loss
            return total_loss / len(self.data['v_scan'])
        else:
            scan_vertices = self.data['v_scan'].to(self.device)
            predicted_faces = self.data['f_reg_global'][0].to(self.device)
            return self.points2surface_loss_function(scan_vertices, vertices, predicted_faces)

    def compute_losses(self):
        all_losses = {}

        if self.args.enable_local:
            global_vertex_losses = self.compute_vertex_losses(self.global_points, self.coarse_points)
        else:
            global_vertex_losses = self.compute_vertex_losses(self.global_points, suffix="")
        
        all_losses.update(global_vertex_losses)


        landmarks_loss_dense_out, global_landmarks_projected_dense_out = self.compute_landmarks_loss(
            self.global_points,
            suffix="_dense",
            gt_landmarks=self.landmarks_dense_uv if self.args.to_meters else self.landmarks_dense,
            gt_mask=self.landmarks_dense_mask
        )
        all_losses['landmarks_loss_dense_out'] = landmarks_loss_dense_out
        self.global_landmarks_projected_dense_out = global_landmarks_projected_dense_out


        # landmarks_loss_dense_fan_out, global_landmarks_projected_dense_fan_out = self.compute_landmarks_loss(
        #     self.global_points,
        #     suffix="_fan",
        #     gt_landmarks=self.landmarks_dense_fan_uv if self.args.to_meters else self.landmarks_dense_fan,
        #     # gt_mask=self.landmarks_dense_fan_mask
        # )
        # all_losses['landmarks_loss_dense_fan_out'] = landmarks_loss_dense_fan_out
        # self.global_landmarks_projected_dense_fan_out = global_landmarks_projected_dense_fan_out

        landmarks_loss_dense_mediapipe_out, global_landmarks_projected_dense_mediapipe_out = self.compute_landmarks_loss(
            self.global_points,
            suffix="_mediapipe",
            gt_landmarks=self.landmarks_dense_mediapipe_uv if self.args.to_meters else self.landmarks_dense_mediapipe,
            # gt_mask=self.landmarks_dense_mediapipe_mask
        )
        all_losses['landmarks_loss_dense_mediapipe_out'] = landmarks_loss_dense_mediapipe_out
        self.global_landmarks_projected_dense_mediapipe_out = global_landmarks_projected_dense_mediapipe_out

        if self.args.enable_diff_rendering:
            global_rendering_losses, global_predictions = self.compute_rendering_losses(self.global_points, suffix="")
            all_losses.update(global_rendering_losses)
        
            for key, value in global_predictions.items():
                if key.endswith(""):  
                    attr_name = key.replace("", "") 
                    if isinstance(value, torch.Tensor) and (value.dtype == torch.float16 or value.dtype == torch.bfloat16):
                        setattr(self, attr_name, value.float())
                    else:
                        setattr(self, attr_name, value)
            

        
        if getattr(self.args, 'enable_virtual_views_loss', False):
            vv_loss, vv_pred = self.compute_virtual_views_loss(self.global_points)
        # print(vv_loss)
        # self.losses['virtual_views_loss'] = vv_loss
            for k, v in vv_pred.items(): setattr(self, k, v)  # optional (for visualize)
            all_losses['virtual_views_loss'] = vv_loss


        # ---------- PLIKS ---------- #
        beta_reg = torch.tensor(0.0, device=self.device)
        exp_reg  = torch.tensor(0.0, device=self.device)

        # weights (put these in your args/config)
        w_beta = getattr(self.args, 'weight_shape_regularization', 0.0)      # e.g., 1e-4
        w_exp  = getattr(self.args, 'weight_expression_regularization', 0.0) # e.g., 1e-4
        w_pose = getattr(self.args, 'weight_pose_regularization', 0.0)       # e.g., 1e-4

        have_flame_branch_params = hasattr(self, 'shape_params') and self.shape_params is not None \
                                and hasattr(self, 'expression_params') and self.expression_params is not None

        if have_flame_branch_params:
            beta_reg = (self.shape_params ** 2).mean() * w_beta
            exp_reg  = (self.expression_params ** 2).mean() * w_exp
            all_losses['beta_regularizer'] = beta_reg
            all_losses['exp_regularizer']  = exp_reg
    
        elif hasattr(self, 'pliks_out') and self.pliks_out is not None:
            beta_all = self.pliks_out['beta']  # (B, n_shape+n_exp)
            n_id, n_exp = self.flame.n_shape, self.flame.n_exp
            beta_id  = beta_all[:, :n_id]
            beta_exp = beta_all[:, n_id:n_id+n_exp]
            beta_reg = (beta_id  ** 2).mean() * w_beta
            exp_reg  = (beta_exp ** 2).mean() * w_exp

            t_hat    = self.pliks_out['t']                                          # [B, 3]
            Rk       = self.pliks_out['Rk']                                         # [B, K, 3, 3]

            V_fit = self.pliks_out['V_fit'] #.detach()                                       # [B, V, 3] ! detach to avoid degeneracy
            # V_fit = inv.get('V_fit_root', inv['V_fit'])

            from models.FLAME.pliks_flame import build_R_world_from_segments, mat_to_axis_angle, world_to_relative_rotations
            J     = self.flame.J_regressor.shape[0]

            R_world = build_R_world_from_segments(Rk, self.pliks_solver.seg_list, J)   # [B,J,3,3]
            R_rel   = world_to_relative_rotations(R_world, self.flame.parents)   # [B,J,3,3]

            # aa_all  = mat_to_axis_angle(R_rel)                               # [B,J,3]

            # swapped with the pytorch3d version
            from pytorch3d.transforms import matrix_to_axis_angle
            aa_all = matrix_to_axis_angle(R_rel)

            # # pick only stable joints (good curriculum): neck + jaw + eyes
            NECK_ID = 1
            # # TODO: fill your indices once you confirm them
            JAW_ID, LEYE_ID, REYE_ID = 2, 3, 4

            pose_params      = torch.zeros(beta_all.shape[0], 3, device=beta_all.device)          # global/root
            neck_pose_params = aa_all[:, NECK_ID]
            if self.no_jaw:
                jaw_params = torch.zeros_like(aa_all[:, JAW_ID])
            else:
                jaw_params       = aa_all[:, JAW_ID]

            # print(jaw_params,'jaw')
            # print(jaw_params)
            eye_pose_params  = torch.cat([aa_all[:, LEYE_ID], aa_all[:, REYE_ID]], dim=-1)
            # print(aa_all.shape)

            def rigid_align_no_scale(X: torch.Tensor, Y: torch.Tensor):
                """
                Solve R,t (no scale) s.t. R X + t ≈ Y   using SVD (batched).
                X,Y: [B,V,3]
                Returns:
                R: [B,3,3], t: [B,3]
                """
                Xc = X - X.mean(dim=1, keepdim=True)
                Yc = Y - Y.mean(dim=1, keepdim=True)
                H  = Xc.transpose(1,2) @ Yc                         # [B,3,3]
                U, S, Vt = torch.linalg.svd(H)                      # Vt=V^T
                R = Vt.transpose(1,2) @ U.transpose(1,2)            # [B,3,3]
                # reflection handling
                det = torch.det(R)
                if (det < 0).any():
                    Vt_fix = Vt.clone()
                    Vt_fix[det < 0, -1, :] *= -1
                    R[det < 0] = Vt_fix[det < 0].transpose(1,2) @ U[det < 0].transpose(1,2)
                t = Y.mean(dim=1) - (R @ X.mean(dim=1).unsqueeze(-1)).squeeze(-1)  # [B,3]
                return R, t


            out = self.flame.forward({
                'shape_params':      beta_id,
                'expression_params': beta_exp,
                'pose_params':       pose_params,
                'neck_pose_params':  neck_pose_params,
                'jaw_params':        jaw_params,
                'eye_pose_params':   eye_pose_params,
            })
            V_flame = out['vertices'] #+ t_hat[:, None, :]   # meters

            with torch.cuda.amp.autocast(enabled=False):
                if self.args.to_meters:
                    global_points = self.global_points.float() 
                else:
                    global_points = self.global_points.float() / self.unit_factor

                R_root, t_root = rigid_align_no_scale(V_flame.float(), global_points)

            V_flame_mm = (R_root[:,None] @ V_flame.unsqueeze(-1)).squeeze(-1) + t_root[:,None,:]

            if not self.args.to_meters:
                V_flame_mm = V_flame_mm*self.unit_factor


            all_losses['beta_regularizer_pliks'] = beta_reg
            all_losses['exp_regularizer_pliks']  = exp_reg
            all_losses['pose_regularizer_pliks']  = aa_all.pow(2).sum(dim=(1,2)).mean() * w_pose

            # edge_mask = {
            #     'w_edge_face': 2.0,
            #     'w_edge_ears': 3.0,
            #     'w_edge_eyeballs': 50.0,
            #     'w_edge_eye_region': 10.0,
            #     'w_edge_lips': 3.0,
            #     'w_edge_neck': 1.0,
            #     'w_edge_nostrils': 10.0,
            #     'w_edge_scalp': 1.0,
            #     'w_edge_boundary': 10.0,
            # }


            all_losses['vertices_regularizer_pliks'] = F.mse_loss(V_flame_mm, self.global_points) * self.args.weight_vertices_regularizer_pliks # NO DETACH !!
            all_losses['vertices_regularizer_pliks_edge'] = self.edge_loss_function(V_flame_mm, self.global_points) * self.args.weight_vertices_regularizer_pliks_edge # NO DETACH !!

            landmarks_loss_dense, global_landmarks_projected_dense = self.compute_landmarks_loss(
                V_flame_mm,
                suffix="_dense",
                gt_landmarks=self.landmarks_dense_uv if self.args.to_meters else self.landmarks_dense,
                gt_mask=self.landmarks_dense_mask
            )
            all_losses['landmarks_loss_dense'] = landmarks_loss_dense
            self.global_landmarks_projected_dense = global_landmarks_projected_dense

            self.pliks_out['V_flame_mm'] = V_flame_mm
            

            # landmarks_loss_dense_fan, global_landmarks_projected_dense_fan = self.compute_landmarks_loss(
            #     V_flame_mm,
            #     suffix="_fan",
            #     gt_landmarks=self.landmarks_dense_fan_uv if self.args.to_meters else self.landmarks_dense_fan,
            #     # gt_mask=self.landmarks_dense_fan_mask
            # )
            # all_losses['landmarks_loss_dense_fan'] = landmarks_loss_dense_fan
            # self.global_landmarks_projected_dense_fan = global_landmarks_projected_dense_fan

            landmarks_loss_dense_mediapipe, global_landmarks_projected_dense_mediapipe = self.compute_landmarks_loss(
                self.global_points,
                suffix="_mediapipe",
                gt_landmarks=self.landmarks_dense_mediapipe_uv if self.args.to_meters else self.landmarks_dense_mediapipe,
                # gt_mask=self.landmarks_dense_mediapipe_mask
            )
            all_losses['landmarks_loss_dense_mediapipe'] = landmarks_loss_dense_mediapipe
            self.global_landmarks_projected_dense_mediapipe = global_landmarks_projected_dense_mediapipe

            # i = np.zeros((2000,2000,3))
            # for i in range(self.global_landmarks_projected_dense_mediapipe[0,0].shape[0]):
            #     x = int(self.global_landmarks_projected_dense_mediapipe[0,0,i,0].item())*20
            #     y = int(self.global_landmarks_projected_dense_mediapipe[0,0,i,1].item())*20
            #     if x >=0 and x <2000 and y>=0 and y<2000:
            #         i[y,x,0] = 1.0
            # import imageio
            # imageio.imwrite('landmarks_debug.png', (i*255).astype(np.uint8))
            # raise

            
        # ---------- END NEW ----------

        # Calculate total loss
        # total_loss = sum([loss for loss in all_losses.values() if isinstance(loss, torch.Tensor)])
        total_loss = sum(
            loss
            for loss in all_losses.values()
            if isinstance(loss, torch.Tensor) and not torch.all(loss == 0)
        )

        # print(all_losses)
        # raise
        # Set instance attributes for backward compatibility
        self.loss = total_loss
        self.points_loss = all_losses.get('points_loss', 0.0)
        self.points2surface_loss = all_losses.get('points2surface_loss', 0.0)
        self.edge_regularizer_loss = all_losses.get('edge_regularizer_loss', 0.0)
        self.normals_loss = all_losses.get('normals_loss', 0.0)
        self.depth_maps_loss = all_losses.get('depth_maps_loss', 0.0)
        self.point_maps_loss = all_losses.get('point_maps_loss', 0.0)
        self.smirk_loss = all_losses.get('smirk_loss', 0.0)
        self.smirk_loss_l1 = all_losses.get('smirk_loss_l1', 0.0)
        self.smirk_loss_vgg = all_losses.get('smirk_loss_vgg', 0.0)
        self.laplacian_smoothing_loss = all_losses.get('laplacian_smoothing_loss', 0.0)
        # self.adversarial_loss = adversarial_loss
        # self.landmarks_loss = landmarks_loss
        self.chamfer_distance = all_losses.get('chamfer_distance', 0.0)
        self.flame_regularization_loss = all_losses.get('flame_regularization_loss', 0.0)
        self.normals_grad_loss = all_losses.get('normals_grad_loss', 0.0)
        self.virtual_views_loss = all_losses.get('virtual_views_loss', 0.0)
        # print('asdf',self.virtual_views_loss)

        # Create losses dictionary for logging
        self.main_losses = {
            'Total loss': total_loss,
            'Points loss': self.points_loss,
            'Normals loss': self.normals_loss,
            'PointMaps loss': self.point_maps_loss,
            'Depth loss': self.depth_maps_loss,
            'Points2Surface loss': self.points2surface_loss,
            'Edge regularizer': self.edge_regularizer_loss,
            'Smirk loss': self.smirk_loss if self.args.enable_smirk_loss else 0.0,
            'Smirk loss L1': self.smirk_loss_l1 if self.args.enable_smirk_loss else 0.0,
            'Smirk loss VGG': self.smirk_loss_vgg if self.args.enable_smirk_loss else 0.0,
            'Identity loss': all_losses.get('identity_loss', 0.0) if getattr(self.args, 'enable_arcface_loss', False) else 0.0,
            'Laplacian smoothing loss': self.laplacian_smoothing_loss,
            'Landmarks loss (dense)': all_losses.get('landmarks_loss_dense', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'Landmarks loss (dense) out': all_losses.get('landmarks_loss_dense_out', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'FLAME regularization loss': self.flame_regularization_loss if self.args.enable_flame_branch else 0.0,
            'β regularizer (PLIKS)': all_losses.get('beta_regularizer_pliks', 0.0) if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'ψ regularizer (PLIKS)': all_losses.get('exp_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Pose regularizer (PLIKS)': all_losses.get('pose_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            't regularizer (PLIKS)': all_losses.get('t_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices edge regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks_edge', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Normals grad loss': self.normals_grad_loss if hasattr(self, 'normals_grad_loss') else 0.0,
            'Virtual views loss': self.virtual_views_loss if hasattr(self, 'virtual_views_loss') else 0.0,
            'Landmarks loss (dense fan)': all_losses.get('landmarks_loss_dense_fan', 0.0), #if getattr(self, 'landmarks_dense_fan', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'Landmarks loss (dense fan) out': all_losses.get('landmarks_loss_dense_fan_out', 0.0), # if getattr(self, 'landmarks_dense_fan', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'Landmarks loss (dense mediapipe)': all_losses.get('landmarks_loss_dense_mediapipe', 0.0), # if getattr(self, 'landmarks_dense_mediapipe', None) is not None and self.args.weight_dense_landmarks > 0.0
            # else 0.0,
            'Landmarks loss (dense mediapipe) out': all_losses.get('landmarks_loss_dense_mediapipe_out', 0.0) # if getattr(self, 'landmarks_dense_mediapipe', None) is not None and self.args.weight_dense_landmarks > 0.0
            # else 0.0,

        }
        if hasattr(self, 'losses'):
            self.losses.update(self.main_losses)
        

    def backward(self):
        if torch.isnan(self.loss) or torch.isinf(self.loss):
            print("FLAME total loss is NaN or Inf, skipping backward step.")
            print(self.losses)
            raise

        if self.args.enable_local:
            self.optimizer_model_local.zero_grad()
            self.loss.backward()

            if self.args.gradient_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.local_model.module.optimizable_parms(), max_norm=self.args.gradient_max_norm, norm_type=2)
            self.optimizer_model_local.step()
        else:
            self.optimizer_model.zero_grad()
            self.loss.backward()
            if self.args.gradient_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.module.optimizable_parms(), max_norm=self.args.gradient_max_norm, norm_type=2)
            self.optimizer_model.step()

    def run(self):
        batch_size = self.args.batch_size
        num_epoch = int(np.ceil(self.args.num_iterations / float(len(self.dataset_train)) * batch_size))
        start_epoch = int(self.global_step / float(len(self.dataset_train)) * batch_size)+1
        print("expect to run for %d epoches" % (num_epoch-start_epoch))

        for epoch_idx in range(start_epoch, num_epoch+1):
            np.random.seed() # reset seed
            print('************************')
            print('Epoch %d / %d' % (epoch_idx, num_epoch))
            print('************************')
            self.train_one_epoch()

    def train_one_epoch(self):
        for data in tqdm(self.dataloader_train, desc='Training', total=len(self.dataloader_train)):
            self.train_step(data)


    # def train_step(self, data):
    #     self.losses = {}

    #     self.feed_data(data)

    #     for param_group in self.optimizer_model.param_groups:
    #         lr = param_group['lr']

    #     self.model.train()
    #     self.forward()
    #     self.compute_losses()
    #     self.backward()

    #     if self.global_step % self.args.print_frequency == 0:        
    #         print('%s, step %d, total loss: %f, ' %(get_time_string(), self.global_step, to_numpy(self.loss)))
    #         d= {
    #             'Learning rate/train': lr,
    #         }
    #         # print(self.data['f_reg_global'])
    #         # Only compute main model metrics if not in FLAME-only mode
    #         # def _points2surface_metric(self, points_to_use, v_scan, v_registration, f_reg_global, flame_masks_triangles):
    #         distances = _points2surface_metric(self.global_points, self.data['v_scan'], self.data['v_registration'], self.data['f_reg_global'], self.flame_masks_triangles)
    #         for key in distances:
    #             d['Points2Surface distance %s/train' % key] = np.mean(distances[key])
    #             d['Points2Surface distance %s/train median' % key] = np.median(distances[key])
    #             d['Points2Surface distance %s/train std' % key] = np.std(distances[key])

    #         for key in self.losses:
    #             d['%s/train' % key] = to_numpy(self.losses[key])
            
    #         pprint.pprint(d)
            
    #         if self.args.wandb:
    #             wandb.log(d, step=self.global_step)

    #     if (self.global_step % self.args.visualize_frequency == 0):# and (self.global_step > 0):
    #         try:
    #             # self.visualize_virtual_views(out_dir_suffix='vv', sample_idx=0)

    #             self.visualize('train')
    #         except Exception as e:
    #             print(f"Error occurred during visualization: {e}")
    #             raise e
    #     if (self.global_step > 0) and (self.global_step % self.args.validate_frequency == 0):               
    #         with torch.no_grad():
    #             self.validate()
    #     if (self.global_step > 0) and (self.global_step % self.args.save_frequency == 0):
    #         self.save_checkpoint()
        

    #     if self.global_step % self.args.print_frequency == 0:        
    #         self.cleanup_memory()
    #         print_memory(self.device, prefix='FW')

    #     self.global_step += 1


    # def train_step(self, data):
    #     self.losses = {}

    #     self.feed_data(data)

    #     for param_group in self.optimizer_model.param_groups:
    #         lr = param_group['lr']

    #     self.model.train()
    #     self.forward()
    #     self.compute_losses()
    #     self.backward()

    #     if self.global_step % self.args.print_frequency == 0:        
    #         print('%s, step %d, total loss: %f, ' %(get_time_string(), self.global_step, to_numpy(self.loss)))
    #         d= {
    #             'Learning rate/train': lr,
    #         }
    #         # print(self.data['f_reg_global'])
    #         # Only compute main model metrics if not in FLAME-only mode
    #         # def _points2surface_metric(self, points_to_use, v_scan, v_registration, f_reg_global, flame_masks_triangles):
    #         distances = _points2surface_metric(self.global_points, self.data['v_scan'], self.data['v_registration'], self.data['f_reg_global'], self.flame_masks_triangles)
    #         for key in distances:
    #             d['Points2Surface distance %s/train' % key] = np.mean(distances[key])
    #             d['Points2Surface distance %s/train median' % key] = np.median(distances[key])
    #             d['Points2Surface distance %s/train std' % key] = np.std(distances[key])

    #         for key in self.losses:
    #             d['%s/train' % key] = to_numpy(self.losses[key])
            
    #         pprint.pprint(d)
            
    #         if self.args.wandb:
    #             wandb.log(d, step=self.global_step)

    #     if (self.global_step % self.args.visualize_frequency == 0):# and (self.global_step > 0):
    #         try:
    #             # self.visualize_virtual_views(out_dir_suffix='vv', sample_idx=0)

    #             self.visualize('train')
    #         except Exception as e:
    #             print(f"Error occurred during visualization: {e}")
    #             raise e
    #     if (self.global_step > 0) and (self.global_step % self.args.validate_frequency == 0):               
    #         with torch.no_grad():
    #             self.validate()
    #     if (self.global_step > 0) and (self.global_step % self.args.save_frequency == 0):
    #         self.save_checkpoint()
        

    #     if self.global_step % self.args.print_frequency == 0:        
    #         self.cleanup_memory()
    #         print_memory(self.device, prefix='FW')

    #     self.global_step += 1


    # def validate(self):
    #     self.losses = {}

    #     validation_losses = {}

    #     vis_validation_every = 25

    #     for i, data in enumerate(self.dataloader_val):
    #         self.feed_data(data, mode='val')

    #         self.model.eval()
    #         if self.args.enable_local:
    #             self.local_model.eval()

    #         self.forward()
    #         self.compute_losses()

    #         for key in self.losses:
    #             if key not in validation_losses:
    #                 validation_losses[key] = []
    #             validation_losses[key].append(to_numpy(self.losses[key]))

    #         distances = _points2surface_metric(self.global_points, self.data['v_scan'], self.data['v_registration'], self.data['f_reg_global'], self.flame_masks_triangles)
    #         for key in distances:
    #             distance_key = 'Points2Surface distance %s' % key
    #             if distance_key not in validation_losses:
    #                 validation_losses[distance_key] = np.array([], dtype=np.float32)
    #             validation_losses[distance_key] = np.concatenate([validation_losses[distance_key], to_numpy(distances[key])])


    #         if i % vis_validation_every == 0:
    #             print('Visualizing validation sample %d' % i)
    #             try:
    #                 self.visualize('val', val_idx=i)
    #             except Exception as e:
    #                 print(f"Error occurred during validation visualization: {e}")

    #             self.export_mesh('val', i)

    #             print_memory(self.device, prefix='FW')
    #         # break

    #     d = {}
    #     for key in validation_losses:
    #         validation_losses[key] = np.array(validation_losses[key])
    #         # print(validation_losses[key].shape)
    #         d['%s/validation' % key] = np.mean(validation_losses[key])
    #         d['%s/validation median' % key] = np.median(validation_losses[key])
    #         d['%s/validation std' % key] = np.std(validation_losses[key])

    #     print('Validation results:')
    #     pprint.pprint(d)
    #     if self.args.wandb:
    #         wandb.log(d, step=self.global_step)

    def visualize(self, mode='train', val_idx=0):
        if mode == 'train':
            key = "_augmented"
        else:
            key = "" # timo had validation with augmented images as well
        
        with torch.no_grad():
            for idx in range(len(self.inputs['images']))[:1]:
                # if we have base model
                if self.args.enable_local:
                    self.global_points_base = self.coarse_results['vertices']
                    reconstructed_vertices_base = to_numpy(self.global_points_base[idx])
                else:
                    if hasattr(self, 'base_model'):
                        self.base_model.eval()
                        self.base_results = self.base_model(self.inputs['images'], self.inputs['camera_intrinsics'], self.inputs['camera_extrinsics'], 
                                                            camera_distortions=self.inputs['camera_distortions'])
                        self.global_points_base = self.base_results['vertices']
                        reconstructed_vertices_base = to_numpy(self.global_points_base[idx])

                target_vertices = to_numpy(self.target_vertices[idx])
                
                reconstructed_vertices = to_numpy(self.global_points[idx])
                
                faces = to_numpy(self.faces)

                vertex_distance = np.linalg.norm(target_vertices-reconstructed_vertices, axis=-1)


                # After you compute reconstructed_* and faces
                reconstructed_vertices_pliks = None
                if hasattr(self, 'pliks_out') and self.pliks_out is not None:
                    # V_fit is the FLAME-reprojected mesh from the recovered params (already aligned to your vertex space)
                    reconstructed_vertices_pliks = to_numpy(self.pliks_out['V_fit'][idx])

                reconstructed_vertices_pliks_flame = None
                if hasattr(self, 'pliks_out') and self.pliks_out is not None and 'V_flame_mm' in self.pliks_out:
                    # V_fit is the FLAME-reprojected mesh from the recovered params (already aligned to your vertex space)
                    reconstructed_vertices_pliks_flame = to_numpy(self.pliks_out['V_flame_mm'][idx])


                scan_vertices = self.data['v_scan'][idx].to(self.device).unsqueeze(0)
                predicted_faces = torch.Tensor(self.faces).to(self.device)

                # Compute scan colors for main model (only if available)
                vertex_colors_scan = None
                if hasattr(self, 'global_points') and self.global_points is not None:
                    predicted_vertices = self.global_points[idx].unsqueeze(0).float()
                    vertex_colors_scan = compute_s2m_distance(scan_vertices, predicted_vertices, predicted_faces, masks=self.flame_masks_triangles)['full']
                    vertex_colors_scan = dist_to_rgb(vertex_colors_scan.detach().cpu().numpy(), min_dist=0.0, max_dist=3.0)

                # Base model scan colors
                vertex_colors_scan_base = None
                if hasattr(self, 'global_points_base'):
                    predicted_vertices_base = self.global_points_base[idx].unsqueeze(0).float()
                    vertex_colors_scan_base = compute_s2m_distance(scan_vertices, predicted_vertices_base, predicted_faces, masks=self.flame_masks_triangles)['full']
                    vertex_colors_scan_base = dist_to_rgb(vertex_colors_scan_base.detach().cpu().numpy(), min_dist=0.0, max_dist=3.0)

                # FLAME scan colors


                vertex_colors_scan_pliks = None
                if reconstructed_vertices_pliks is not None:
                    predicted_vertices_pliks = torch.from_numpy(reconstructed_vertices_pliks).unsqueeze(0).float().to(self.device)
                    vertex_colors_scan_pliks = compute_s2m_distance(
                        scan_vertices, predicted_vertices_pliks, predicted_faces, masks=self.flame_masks_triangles
                    )['full']
                    vertex_colors_scan_pliks = dist_to_rgb(vertex_colors_scan_pliks.detach().cpu().numpy(), min_dist=0.0, max_dist=3.0)

                scan_vertices = self.data['v_scan'][idx]

                for view_id in self.visualization_view_ids:
                    
                    if view_id >= self.data['color_images'][idx].shape[0]:
                        continue
                    input_image = to_numpy(self.data[f'color_images{key}'][idx][view_id].permute(1,2,0))
                    camera_intrinsics = to_numpy(self.data[f'color_camera_intrinsics{key}'][0][view_id])
                    camera_extrinsics = to_numpy(self.data['color_camera_extrinsics'][idx][view_id])
                    radial_distortion = to_numpy(self.data['color_camera_distortions'][idx][view_id]) 

                    if mode == 'train':
                        input_image = self.dataset_train.denormalize_image(input_image)
                    else:
                        input_image = self.dataset_val.denormalize_image(input_image)

                    relative_view_id = np.where(self.current_batch_view_ids==view_id)[0][0]

                    if self.args.enable_diff_rendering:
                        gt_normals = to_numpy(self.normal_maps_gt_01[idx][relative_view_id].permute(1,2,0))
                        # gt_normals = (gt_normals + 1.0) / 2.0
                        gt_normals = (255*gt_normals).astype(np.uint8)

                        depth_gt = self.depth_maps_gt[idx][relative_view_id].cpu().numpy().squeeze(-1)
                        depth_gt = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min())
                        depth_gt = (255*depth_gt).astype(np.uint8)
                        depth_gt = np.stack((depth_gt, depth_gt, depth_gt), axis=-1)  # Convert to 3-channel image
                        
                        point_maps_gt = self.pointmaps_gt[idx][relative_view_id].cpu().numpy()
                        point_maps_gt, _min, _max = pointmap_to_rgb(point_maps_gt)


                    if self.args.enable_diff_rendering and hasattr(self, 'normal_maps_pred') and self.normal_maps_pred is not None:
                        pred_normals = to_numpy(self.normal_maps_pred[idx][relative_view_id].permute(1,2,0))

                        # pred_normals = (pred_normals + 1.0) / 2.0
                        pred_normals = (255*pred_normals).astype(np.uint8)
                        pred_normals = np.clip(pred_normals, 0, 255)

                        normals_loss_per_pixel = self.normals_loss_per_pixel[idx][relative_view_id].cpu().numpy()
                        height, width = normals_loss_per_pixel.shape
                        normals_loss_per_pixel_vis = dist_to_rgb(normals_loss_per_pixel.reshape(-1), min_dist=0.0, max_dist=3)
                        normals_loss_per_pixel_vis = normals_loss_per_pixel_vis.reshape(height, width, 3)

                        point_map_loss_per_pixel = self.point_maps_loss_per_pixel[idx][relative_view_id].cpu().numpy()
                        height, width = point_map_loss_per_pixel.shape
                        point_map_loss_per_pixel_vis = dist_to_rgb(point_map_loss_per_pixel.reshape(-1), min_dist=0.0, max_dist=3)
                        point_map_loss_per_pixel_vis = point_map_loss_per_pixel_vis.reshape(height, width, 3)

                        depth_maps_loss_per_pixel = self.depth_maps_loss_per_pixel[idx][relative_view_id].cpu().numpy()
                        height, width = depth_maps_loss_per_pixel.shape
                        depth_maps_loss_per_pixel_vis = dist_to_rgb(depth_maps_loss_per_pixel.reshape(-1), min_dist=0.0, max_dist=3)
                        depth_maps_loss_per_pixel_vis = depth_maps_loss_per_pixel_vis.reshape(height, width, 3)

                        # normal_geo_angle_map_per_pixel = self.normal_geo_angle_map[idx][relative_view_id].cpu().numpy()
                        # height, width = normal_geo_angle_map_per_pixel.shape
                        # normal_geo_angle_map_per_pixel_vis = dist_to_rgb(normal_geo_angle_map_per_pixel.reshape
                        #                                                 (-1), min_dist=0.0, max_dist=6.0)
                        # normal_geo_angle_map_per_pixel_vis = normal_geo_angle_map_per_pixel_vis.reshape(height, width, 3)

                        depth_vis = self.depth_maps_pred[idx][relative_view_id].cpu().numpy()
                        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
                        depth_vis = (255*depth_vis).astype(np.uint8)
                        depth_vis = np.stack((depth_vis, depth_vis, depth_vis), axis=-1)  # Convert to 3-channel image

                        point_maps_pred = self.pointmaps_pred[idx][relative_view_id].cpu().numpy()
                        point_maps_pred, _, _ = pointmap_to_rgb(point_maps_pred, _min,_max)

                        # --- NEW: normals gradient visualizations (main) ---
                        normals_grad_pp = getattr(self, 'normals_grad_loss_per_pixel', None)
                        if normals_grad_pp is not None:
                            normals_grad_pp = normals_grad_pp[idx][relative_view_id].cpu().numpy()
                            H, W = normals_grad_pp.shape
                            normals_grad_pp_vis = dist_to_rgb(normals_grad_pp.reshape(-1), min_dist=0.0, max_dist=3.0).reshape(H, W, 3)

                            grad_mag_pred = getattr(self, 'normals_grad_mag_pred', None)
                            grad_mag_gt   = getattr(self, 'normals_grad_mag_gt', None)
                            if grad_mag_pred is not None and grad_mag_gt is not None:
                                gmp = grad_mag_pred[idx][relative_view_id].cpu().numpy()
                                gmg = grad_mag_gt[idx][relative_view_id].cpu().numpy()

                                def to_gray3(x):
                                    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                                    x = (255 * x).astype(np.uint8)
                                    return np.stack((x, x, x), axis=-1)

                                normals_grad_mag_pred_vis = to_gray3(gmp)
                                normals_grad_mag_gt_vis   = to_gray3(gmg)
                        # --- END NEW ---





                    if self.args.enable_smirk_loss:
                        relative_view_id = np.where(self.current_batch_view_ids==view_id)[0][0]
                        pred_fused_image = self.pred_fused_images[idx][relative_view_id].cpu().numpy()
                        gt_fused_image = self.gt_fused_images[idx][relative_view_id].cpu().numpy()
                        masked_image = self.masked_images[idx][relative_view_id].cpu().numpy()
                        # print(pred_fused_image.max(), pred_fused_image.min(),'fused image stats')
                        pred_fused_image = (255*pred_fused_image).astype(np.uint8)
                        # print(pred_fused_image.max(), pred_fused_image.min(),'fused image stats')

                        gt_fused_image = (255*gt_fused_image).astype(np.uint8)
                        masked_image = (255*masked_image).astype(np.uint8)

                        # transpose them !
                        pred_fused_image = pred_fused_image.transpose(1, 2, 0)
                        gt_fused_image = gt_fused_image.transpose(1, 2, 0)
                        masked_image = masked_image.transpose(1, 2, 0)
                        # cv2.imwrite('pred_fused_image.png', pred_fused_image[:,:,::-1])
                        # cv2.imwrite('gt_fused_image.png', gt_fused_image[:,:,::-1])
                        # cv2.imwrite('masked_image.png', masked_image[:,:,::-1])
                        # raise


                        # FLAME SMIRK predictions

                    input_image = (255*input_image).astype(np.uint8)

                    camera_args = {
                        'camera_intrinsics': camera_intrinsics,
                        'camera_extrinsics': camera_extrinsics,
                        'radial_distortion': radial_distortion,
                        'frustum': {'near': 0.01, 'far': 3000.0},
                        'image_size': input_image.shape[:2]
                    }

                    target_rendering = render_mesh(vertices=target_vertices, faces=faces, vertex_colors=None, **camera_args)
                    
                    # Only render main model if available
                    reconstruction_rendering = None
                    if reconstructed_vertices is not None:
                        reconstruction_rendering = render_mesh(vertices=reconstructed_vertices, faces=faces, vertex_colors=None, needs_projection=True, **camera_args)


                    # Only render error if we have main model predictions
                    target_error_rendering = None
                    if vertex_colors_scan is not None:
                        target_error_rendering = render_mesh(vertices=scan_vertices, faces=self.data['f_scan'][idx], vertex_colors=vertex_colors_scan, **camera_args)
                    
                    base_error_rendering = None
                    if hasattr(self, 'base_model') and vertex_colors_scan_base is not None:
                        base_error_rendering = render_mesh(vertices=scan_vertices, faces=self.data['f_scan'][idx], vertex_colors=vertex_colors_scan_base, **camera_args)
                    
                    base_reconstruction_rendering = None
                    if hasattr(self, 'base_model'):
                        base_reconstruction_rendering = render_mesh(vertices=reconstructed_vertices_base, faces=faces, vertex_colors=None, needs_projection=True, **camera_args)

                    pliks_reconstruction_rendering = None
                    if reconstructed_vertices_pliks is not None:
                        pliks_reconstruction_rendering = render_mesh(vertices=reconstructed_vertices_pliks, faces=faces, vertex_colors=None, needs_projection=True, **camera_args)

                    pliks_flame_reconstruction_rendering = None
                    if reconstructed_vertices_pliks_flame is not None:
                        pliks_flame_reconstruction_rendering = render_mesh(vertices=reconstructed_vertices_pliks_flame, faces=faces, vertex_colors=None, needs_projection=True, **camera_args)

                    vertex_colors_scan_pliks = None
                    if reconstructed_vertices_pliks is not None:
                        pliks_error_rendering = render_mesh(vertices=scan_vertices, faces=self.data['f_scan'][idx], vertex_colors=vertex_colors_scan_pliks, **camera_args)

                    # FLAME renderings
                    flame_reconstruction_rendering = None
                    flame_error_rendering = None

                    if 'f_scan' in self.data:
                        target_scan = render_mesh(vertices=scan_vertices, faces=self.data['f_scan'][idx], vertex_colors=None, **camera_args)
                    else:
                        target_scan = render_mesh(vertices=scan_vertices, faces=None, vertex_colors=None, **camera_args)

                    dense_gt_image = None
                    dense_pred_image = None

                    vis_landmarks = True
                    if vis_landmarks:
                        vis_image = input_image.copy()
                        vis_image_pred = input_image.copy()
                        # print(self.landmarks_dense_fan.shape)
                        # for landmark in self.landmarks_dense_fan[idx][relative_view_id]:
                        # landmarks_dense_fan = to_numpy(self.landmarks_dense_fan[idx][relative_view_id])
                        # for i in range(landmarks_dense_fan.shape[0]):
                            # vis_image = cv2.circle(vis_image, (int(landmarks_dense_fan[i][0]), int(landmarks_dense_fan[i][1])), 2, (0, 0, 255), -1)

                        landmarks_dense_mediapipe = to_numpy(self.landmarks_dense_mediapipe[idx][relative_view_id])
                        for i in range(landmarks_dense_mediapipe.shape[0]):
                            vis_image = cv2.circle(vis_image, (int(landmarks_dense_mediapipe[i][0]), int(landmarks_dense_mediapipe[i][1])), 2, (255, 0, 255), -1)

                        # projected_landmarks_fan = None
                        # if hasattr(self, 'global_landmarks_projected_dense_fan_out') and self.global_landmarks_projected_dense_fan_out is not None:
                        #     projected_landmarks_fan = to_numpy(self.global_landmarks_projected_dense_fan_out[idx][relative_view_id])
                        #     for i in range(projected_landmarks_fan.shape[0]):
                        #         vis_image_pred = cv2.circle(vis_image_pred, (int(projected_landmarks_fan[i][0]), int(projected_landmarks_fan[i][1])), 2, (255, 0, 0), -1)

                        projected_landmarks_mediapipe = None
                        if hasattr(self, 'global_landmarks_projected_dense_mediapipe') and self.global_landmarks_projected_dense_mediapipe is not None:
                            projected_landmarks_mediapipe = to_numpy(self.global_landmarks_projected_dense_mediapipe[idx][relative_view_id])
                            for i in range(projected_landmarks_mediapipe.shape[0]):
                                vis_image_pred = cv2.circle(vis_image_pred, (int(projected_landmarks_mediapipe[i][0]), int(projected_landmarks_mediapipe[i][1])), 2, (255, 0, 0), -1)   

                        # target_landmarks = to_numpy(self.landmarks[idx][relative_view_id])
                        # target_landmarks_mediapipe = to_numpy(self.landmarks_mediapipe[idx][relative_view_id])
                        # for i in range(0,17):
                        #     vis_image = cv2.circle(vis_image, (int(target_landmarks[i][0]), int(target_landmarks[i][1])), 3, (0, 255, 0), -1)

                        # for i in range(target_landmarks_mediapipe.shape[0]):
                        #     vis_image = cv2.circle(vis_image, (int(target_landmarks_mediapipe[i][0]), int(target_landmarks_mediapipe[i][1])), 3, (0, 255, 255), -1)

                        # if hasattr(self, 'global_landmarks_projected') and self.global_landmarks_projected is not None:
                        #     projected_landmarks = to_numpy(self.global_landmarks_projected[idx][relative_view_id])
                        #     for i in range(projected_landmarks.shape[0]):
                        #         vis_image = cv2.circle(vis_image, (int(projected_landmarks[i][0]), int(projected_landmarks[i][1])), 3, (255, 0, 0), -1)

                        # if hasattr(self, 'global_landmarks_projected_mp') and self.global_landmarks_projected_mp is not None:
                        #     projected_landmarks = to_numpy(self.global_landmarks_projected_mp[idx][relative_view_id])
                        #     for i in range(projected_landmarks.shape[0]):
                        #         vis_image = cv2.circle(vis_image, (int(projected_landmarks[i][0]), int(projected_landmarks[i][1])), 3, (255, 0, 0), -1)
                        

                        # FLAME landmarks (red circles)

                        if getattr(self, 'landmarks_dense', None) is not None:
                            dense_gt = to_numpy(self.landmarks_dense[idx][relative_view_id])
                            view_mask_val = None
                            if getattr(self, 'landmarks_dense_mask', None) is not None:
                                view_mask_val = float(to_numpy(self.landmarks_dense_mask[idx][relative_view_id]).item())
                            if dense_gt.shape[0] > 0 and (view_mask_val is None or view_mask_val > 0):
                                colors = get_dense_vertex_colors(dense_gt.shape[0], self.flame_masks)
                                base_rgb = (np.ones_like(input_image)*255.0).astype(np.uint8)
                                base_rgb = input_image.copy()
                                gt_bgr = cv2.cvtColor(base_rgb.copy(), cv2.COLOR_RGB2BGR)
                                dense_gt_drawn = draw_dense_points(gt_bgr.copy(), dense_gt, colors)
                                dense_gt_image = cv2.cvtColor(dense_gt_drawn, cv2.COLOR_BGR2RGB)

                                dense_pred_image = None
                                if getattr(self, 'global_landmarks_projected_dense', None) is not None:
                                    dense_pred = to_numpy(self.global_landmarks_projected_dense[idx][relative_view_id])
                                    pred_bgr = draw_dense_points(gt_bgr.copy(), dense_pred, colors)
                                    dense_pred_image = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

                                dense_pred_out_image = None
                                if getattr(self, 'global_landmarks_projected_dense_out', None) is not None:
                                    dense_pred = to_numpy(self.global_landmarks_projected_dense_out[idx][relative_view_id])
                                    pred_bgr = draw_dense_points(gt_bgr.copy(), dense_pred, colors)
                                    dense_pred_out_image = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

   
                    else:
                        vis_image = input_image.copy()


                    # --- SMIRK loss maps (per-pixel) ---
                    if hasattr(self, 'smirk_l1_loss_per_pixel') and len(self.smirk_l1_loss_per_pixel[idx]) > relative_view_id:
                        smirk_l1_pp = self.smirk_l1_loss_per_pixel[idx][relative_view_id].cpu().numpy()  # (H,W)
                        H_, W_ = smirk_l1_pp.shape
                        smirk_l1_vis = dist_to_rgb(smirk_l1_pp.reshape(-1), min_dist=0.0, max_dist=3.0).reshape(H_, W_, 3)
                        smirk_l1_vis = (smirk_l1_vis * 255).astype(np.uint8)

                    if hasattr(self, 'smirk_l1_loss_per_pixel') and len(self.smirk_l1_loss_per_pixel[idx]) > relative_view_id:
                        smirk_vgg_pp = self.smirk_vgg_loss_per_pixel[idx][relative_view_id].cpu().numpy()
                        H_, W_ = smirk_vgg_pp.shape
                        smirk_vgg_vis = dist_to_rgb(smirk_vgg_pp.reshape(-1), min_dist=0.0, max_dist=3.0).reshape(H_, W_, 3)
                        smirk_vgg_vis = (smirk_vgg_vis * 255).astype(np.uint8)


                    # Pair up images and labels
                    pairs = [
                        (input_image, "Input"),
                        *([(dense_gt_image, "Dense GT")] if dense_gt_image is not None else []),
                        *([(dense_pred_image, "Dense Pred")] if dense_pred_image is not None else []),
                        *([(dense_pred_out_image, "Dense Pred Out")] if dense_pred_out_image is not None else []),
                        (target_scan, "Target Scan"),
                        (target_rendering, "Target"),
                        *([(base_reconstruction_rendering, "Base Recon")] if base_reconstruction_rendering is not None else []),
                        *([(reconstruction_rendering, "Recon")] if reconstruction_rendering is not None else []),
                        *([(flame_reconstruction_rendering, "FLAME Recon")] if flame_reconstruction_rendering is not None else []),
                        *([(target_error_rendering, "Error")] if target_error_rendering is not None else []),
                        *([(base_error_rendering, "Base Error")] if base_error_rendering is not None else []),
                        *([(flame_error_rendering, "FLAME Error")] if flame_error_rendering is not None else []),
                        *([(pred_normals, "Pred Normals")] if self.args.enable_diff_rendering and hasattr(self, 'normal_maps_pred') and self.normal_maps_pred is not None else []),
                        *([(gt_normals, "GT Normals")] if self.args.enable_diff_rendering and hasattr(self, 'normal_maps_pred') and self.normal_maps_pred is not None else []),
                        *([(normals_loss_per_pixel_vis, "Normals Loss")] if self.args.enable_diff_rendering and hasattr(self, 'normal_maps_pred') and self.normal_maps_pred is not None else []),
                        # *([(normal_geo_angle_map_per_pixel_vis, "Normals Geo Angle")] if self.args.enable_diff_rendering and hasattr(self, 'normal_geo_angle_map_per_pixel') and self.normal_geo_angle_map_per_pixel is not None else []),
                        *([(depth_vis, "Pred Depth")] if self.args.enable_diff_rendering and hasattr(self, 'depth_maps_pred') and self.depth_maps_pred is not None else []),
                        *([(depth_gt, "GT Depth")] if self.args.enable_diff_rendering and hasattr(self, 'depth_maps_pred') and self.depth_maps_pred is not None else []),
                        *([(depth_maps_loss_per_pixel_vis, "Depth Loss")] if self.args.enable_diff_rendering and hasattr(self, 'depth_maps_pred') and self.depth_maps_pred is not None else []),
                        *([(point_maps_pred, "Pred PointMap")] if self.args.enable_diff_rendering and hasattr(self, 'pointmaps_pred') and self.pointmaps_pred is not None else []),
                        *([(point_maps_gt, "GT PointMap")] if self.args.enable_diff_rendering and hasattr(self, 'pointmaps_pred') and self.pointmaps_pred is not None else []),
                        *([(point_map_loss_per_pixel_vis, "PointMap Loss")] if self.args.enable_diff_rendering and hasattr(self, 'pointmaps_pred') and self.pointmaps_pred is not None else []),
                        *([(pred_fused_image, "Pred Fused")] if self.args.enable_smirk_loss else []),
                        *([(gt_fused_image, "GT Fused")] if self.args.enable_smirk_loss else []),
                        *([(smirk_l1_vis,  "SMIRK L1 Loss")]  if 'smirk_l1_vis'  in locals() else []),
                        *([(smirk_vgg_vis, "SMIRK VGG Loss")] if 'smirk_vgg_vis' in locals() else []),
                        *([(masked_image, "Masked Input")] if self.args.enable_smirk_loss else []),
                        *([(pliks_reconstruction_rendering, "PLIKS Recon")] if pliks_reconstruction_rendering is not None else []),   # <-- NEW
                        *([(pliks_error_rendering, "PLIKS Error")] if reconstructed_vertices_pliks is not None else []),  # <-- NEW
                        *([(pliks_flame_reconstruction_rendering, "PLIKS FLAME Recon")] if pliks_flame_reconstruction_rendering is not None else []),  # <-- NEW
                        *([(normals_grad_mag_pred_vis, "Pred Normals ∇")] if 'normals_grad_mag_pred_vis' in locals() else []),
                        *([(normals_grad_mag_gt_vis,   "GT Normals ∇")]   if 'normals_grad_mag_gt_vis' in locals() else []),
                        *([(normals_grad_pp_vis,       "Normals ∇ Loss")] if 'normals_grad_pp_vis' in locals() else []),
                        *([(normals_grad_mag_pred_vis_flame, "FLAME Normals ∇")] if 'normals_grad_mag_pred_vis_flame' in locals() else []),
                        *([(normals_grad_mag_gt_vis_flame,   "FLAME GT Normals ∇")] if 'normals_grad_mag_gt_vis_flame' in locals() else []),
                        *([(normals_grad_pp_vis_flame,       "FLAME Normals ∇ Loss")] if 'normals_grad_pp_vis_flame' in locals() else []),
                        *([(vis_image, "Input Image with Landmarks")] if 'vis_image' in locals() else []),
                        *([(vis_image_pred, "Input Image with Predicted Landmarks")] if 'vis_image_pred' in locals() else []),
                    ]

                    images, labels = map(list, zip(*pairs))

                    labeled_images = add_labels_to_images(images, labels)

                    # Grid config: 4 rows to accommodate FLAME results
                    n = len(labeled_images)
                    nrows = 4
                    ncols = math.ceil(n / nrows)

                    # Pad the list if needed to make a full grid
                    while len(labeled_images) < nrows * ncols:
                        h, w, c = labeled_images[0].shape
                        labeled_images.append(np.ones((h, w, c), dtype=np.uint8) * 255)  # white padding image

                    # Organize into rows
                    rows = []
                    for i in range(nrows):
                        row_imgs = labeled_images[i*ncols:(i+1)*ncols]
                        rows.append(np.hstack(row_imgs))

                    # Stack rows to form final image
                    visualization = np.vstack(rows)
                    visualization = visualization.transpose(2, 0, 1)  # C x H x W

                    os.makedirs(os.path.join(self.directory_output, mode + '_images'), exist_ok=True)
                    # if mode == 'val':
                    cv2.imwrite(f'{self.directory_output}/{mode}_images/view_id_{view_id:02d}_{val_idx}.jpg', cv2.cvtColor(visualization.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(f'{self.directory_output}/{mode}_images/view_id_{view_id:02d}_{self.global_step}_{idx}.jpg', cv2.cvtColor(visualization.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR))


                    if self.args.wandb and idx == 0 and (self.global_step % 500 == 0):
                        wandb.log({f'{mode.capitalize()}/view_id_{view_id:02d}': wandb.Image(visualization.transpose(1, 2, 0))}, step=self.global_step)


        # --- Virtual-views compact panels ---
        # Only if we computed vv loss in this step (attributes present)
        try:
            self.visualize_virtual_views(out_dir_suffix='vv', sample_idx=0)
        except Exception as e:
            raise e
            print('[visualize_virtual_views] Error:', e)


    def visualize_multiple_front(self, vertices_list, faces_list, labels, out_path=None, sample_idx=0,
                                image_size=(800, 800), max_err_mm=3.0):
        """
        Build one big HxW row:
        [ mesh_0 gray | mesh_0 error_on_scan | mesh_1 gray | mesh_1 error_on_scan | ... ]
        - vertices_list: list of (V,3) np.float32/float64 arrays (pred meshes)
        - faces_list:    list of (F,3) int arrays
        - out_path:      where to save (defaults into {self.directory_output}/val_images)
        - sample_idx:    which scan from current batch to use as reference
        - image_size:    per-tile size (H,W)
        - max_err_mm:    colorbar max for dist_to_rgb
        """
        import numpy as np, cv2, torch
        from pytorch3d.transforms import axis_angle_to_matrix

        assert len(vertices_list) == len(faces_list), "vertices_list and faces_list must have same length."

        # --- Camera (front-ish) ---
        # Base intrinsics from current batch; then zoom-in by s
        K = self.inputs['camera_intrinsics'][0, 0].detach().cpu().numpy().copy()
        s = 3.0
        K[0, 0] *= s; K[1, 1] *= s
        K[0, 2] *= s; K[1, 2] *= s

        # Start from an existing extrinsic, then overwrite with a canonical head-on pose
        Extr = self.inputs['camera_extrinsics'][0, 0].detach().cpu().numpy().copy()
        # Slight tilt from above; feel free to tweak the 1st component
        R = axis_angle_to_matrix(torch.tensor([[3.125, 0.0, 0.0]], dtype=torch.float32)).squeeze(0).numpy()
        Extr[:3, :3] = R
        Extr[0, 3] = -50.0
        Extr[1, 3] = -38.0
        Extr[2, 3] = 1350.0

        # print(K.shape, Extr.shape)
        # proj = K @ Extr
        # print('K')
        # print(K)
        # print('Extr')
        # print(Extr)
        # print('K@Extr = ')
        # print(proj)
        # raise
        

        # --- Scan geometry for error coloring ---
        if isinstance(self.data['v_scan'], list):
            scan_v = self.data['v_scan'][sample_idx].detach().cpu()
            scan_f = self.data['f_scan'][sample_idx].detach().cpu()
        else:
            scan_v = self.data['v_scan'][sample_idx].detach().cpu()
            scan_f = self.data['f_scan'][sample_idx].detach().cpu()
        scan_v_np = scan_v.numpy()
        scan_f_np = scan_f.numpy()

        tiles = []
        H, W = image_size

        for idx, (V_np, F_np) in enumerate(zip(vertices_list, faces_list)):
            # 1) Gray render of the predicted mesh
            gray_img = render_mesh(
                vertices=V_np, faces=F_np, vertex_colors=None,
                camera_extrinsics=Extr, camera_intrinsics=K,
                radial_distortion=None, image_size=image_size, needs_projection=True
            )

            # 2) Error colors on the scan w.r.t. the predicted mesh
            with torch.no_grad():
                pred_v = torch.as_tensor(V_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                pred_f = torch.as_tensor(F_np, dtype=torch.int64, device=self.device)
                scan_v_t = scan_v.unsqueeze(0).to(self.device)

                dist = compute_s2m_distance(
                    scan_v_t, pred_v, pred_f, masks=self.flame_masks_triangles
                )['full']  # (N_scan_v,)
                err_rgb = dist_to_rgb(
                    dist.detach().cpu().numpy(), min_dist=0.0, max_dist=1.0
                )  # (N_scan_v, 3) in [0,1]

            err_img = render_mesh(
                vertices=scan_v_np, faces=scan_f_np, vertex_colors=err_rgb,
                camera_extrinsics=Extr, camera_intrinsics=K,
                radial_distortion=None, image_size=image_size, needs_projection=True
            )

            # Label overlays
            def put_label(img, txt):
                out = img.copy()
                cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
                return out

            gray_img = put_label(gray_img, f"Mesh {labels[idx]} gray")
            err_img  = put_label(err_img,  "Scan error")

            # pair horizontally
            pair = np.hstack([gray_img, err_img])
            tiles.append(pair)

        if not tiles:
            return

        row = np.hstack(tiles)
        os.makedirs(os.path.join(self.directory_output, 'val_images'), exist_ok=True)
        if out_path is None:
            out_path = os.path.join(self.directory_output, 'val_images',
                                    f'front_row_{self.global_step:06d}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f'[visualize_multiple_front] Saved: {out_path}')


    def export_mesh(self, mode, idx):

        out_sequence_dir = os.path.join(self.directory_output, 'validation_meshes')
        os.makedirs(out_sequence_dir, exist_ok=True)

        to_meter_scale_factor = 0.001  # Convert to meters

        target_vertices = to_numpy(self.target_vertices[0]) * to_meter_scale_factor
        scan_vertices = self.data['v_scan'][0] * to_meter_scale_factor
        scan_faces = self.data['f_scan'][0]
        faces = to_numpy(self.faces)


        scene = trimesh.Scene()

        mesh1 = trimesh.Trimesh(vertices=target_vertices, faces=faces, process=False)
        scan_mesh = trimesh.Trimesh(vertices=scan_vertices, faces=scan_faces, process=False)
        scene.add_geometry(mesh1, node_name="target")
        scene.add_geometry(scan_mesh, node_name="scan")


        if hasattr(self, 'global_points_flame') and self.global_points_flame is not None:
            reconstructed_vertices_flame = to_numpy(self.global_points_flame[0]) * to_meter_scale_factor
            mesh_flame = trimesh.Trimesh(vertices=reconstructed_vertices_flame, faces=faces, process=False)
            scene.add_geometry(mesh_flame, node_name="flame")
         
        if hasattr(self, 'global_points') and self.global_points is not None:
            reconstructed_vertices = to_numpy(self.global_points[0]) * to_meter_scale_factor
            mesh_2 = trimesh.Trimesh(vertices=reconstructed_vertices, faces=faces, process=False)
            scene.add_geometry(mesh_2, node_name="reconstructed")

        out_fname = os.path.join(out_sequence_dir, f'{mode}_mesh_{self.global_step}_{idx:04d}.ply')

        # you can export to .obj, .glb, .gltf, etc.
        scene.export(out_fname.replace('.ply','.glb'))


    def compute_virtual_views_loss( # experiment v2
        self,
        vertices_pred,
        # zoom_range=(1.10, 1.90),   # we are training with this the local
        zoom_range=(1.50, 3.50),   # fx, fy *= s
        rot_deg=2.5,              # random ±rot_deg around each cam axis (cam frame)
        weight=None
    ):
        import torch, math
        device = vertices_pred.device
        dtype  = vertices_pred.dtype

        if weight is None:
            weight = getattr(self.args, 'weight_virtual_views', 10.0)
        if weight <= 0.0:
            return torch.tensor(0.0, device=device, dtype=dtype), {}

        Bp, V, _ = vertices_pred.shape
        H = int(self.inputs['images'].shape[-2])
        W = int(self.inputs['images'].shape[-1])

        K_real    = self.inputs['camera_intrinsics']   # (B_inp, N, 3, 3)
        Extr_real = self.inputs['camera_extrinsics']   # (B_inp, N, 3, 4)
        B_inp, N_real = K_real.shape[:2]

        # batch-align
        if B_inp != Bp:
            rep = int((Bp + B_inp - 1) // B_inp)
            K_real    = K_real.repeat(rep, 1, 1, 1)[:Bp]
            Extr_real = Extr_real.repeat(rep, 1, 1, 1)[:Bp]

        # drop the last two (back views)
        keep_idx = torch.arange(max(0, N_real), device=device)
        n_vv = keep_idx.numel()
        if n_vv == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), {}

        Kv_all   = K_real[:, keep_idx].clone()         # (B, n_vv, 3, 3)
        Extr_all = Extr_real[:, keep_idx].clone()      # (B, n_vv, 3, 4)

        # ---- zoom intrinsics (fx, fy) ----
        s_lo, s_hi = float(zoom_range[0]), float(zoom_range[1])
        s = (s_lo + (s_hi - s_lo) * torch.rand(Bp, n_vv, device=device, dtype=dtype))
        Kv_all[..., 0, 0] = Kv_all[..., 0, 0] * s
        Kv_all[..., 0, 1] = Kv_all[..., 0, 1] * s
        Kv_all[..., 1, 1] = Kv_all[..., 1, 1] * s
        Kv_all[..., 1, 0] = Kv_all[..., 1, 0] * s
        # cx, cy unchanged

        # ---- small random rotation in camera frame; keep camera center fixed ----
        # [DELETE the old R_delta-based code block entirely]
        # R = Extr_all[..., :3]
        # t = Extr_all[..., 3]
        # ... ax/ay/az, R_delta, R_new, t_new ...

        # ---- NEW: Dolly/orbit sampler while keeping the same look-at target ----
        R = Extr_all[..., :3].contiguous()         # (B, n, 3, 3) world->cam
        t = Extr_all[..., 3].contiguous()          # (B, n, 3)

        # Camera center in world: C = -R^T t
        Rt = R.transpose(-2, -1)                   # (B, n, 3, 3)
        C = -(Rt @ t.unsqueeze(-1)).squeeze(-1)    # (B, n, 3)

        # Original forward (optical axis) in world.
        # Convention: camera looks along +Z_cam. So ez_cam = [0,0,1].
        ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        ez = ez.view(1, 1, 3, 1).expand(R.shape[0], R.shape[1], 3, 1)      # (B,n,3,1)
        f0 = (Rt @ ez).squeeze(-1)                                         # (B,n,3), world forward
        f0 = torch.nn.functional.normalize(f0, dim=-1, eps=1e-9)

        # Original up in world (to preserve roll). For image coords (x right, y down),
        # up_cam is -Y: [0,-1,0]. If your renderer uses +Y up, change to [0,1,0].
        up_cam = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        up_cam = up_cam.view(1,1,3,1).expand(R.shape[0], R.shape[1], 3, 1)  # (B,n,3,1)
        up0 = (Rt @ up_cam).squeeze(-1)                                     # (B,n,3), world up
        # In case up0 ~ f0, fix degeneracy with a global up
        global_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1,1,3).expand_as(up0)
        colin = (torch.abs((up0 * f0).sum(-1)) > 0.99).unsqueeze(-1)        # (B,n,1)
        up0 = torch.where(colin, global_up, up0)
        up0 = torch.nn.functional.normalize(up0, dim=-1, eps=1e-9)

        # Choose a gaze target G ≈ intersection of the original optical axis with the mesh region.
        # Use predicted-vertex centroid as an anchor for depth along f0:
        mesh_ctr = vertices_pred.mean(dim=1, keepdim=True)                  # (B,1,3)
        mesh_ctr = mesh_ctr.expand(-1, R.shape[1], -1)                      # (B,n,3)
        s = ((mesh_ctr - C) * f0).sum(-1, keepdim=True)                     # (B,n,1) signed distance along f0
        G = C + s * f0                                                      # (B,n,3)
        rad = torch.norm(C - G, dim=-1, keepdim=True).clamp_min(1e-9)  # (B,n,1)

        # Build an orthonormal basis (u,v,f0) for the tangent plane around f0
        u = torch.nn.functional.normalize(torch.cross(up0, f0, dim=-1), dim=-1, eps=1e-9)  # (B,n,3)
        v = torch.nn.functional.normalize(torch.cross(f0,  u, dim=-1), dim=-1, eps=1e-9)   # (B,n,3)

        # --- Spherical orbit around G by small angles (no K change) ---
        max_deg = getattr(self.args, 'vv_sphere_deg', 25.0)   # e.g., up to ±8°
        max_rad = math.radians(max_deg)

        # Sample angular offsets: dtheta (elevation about v), dphi (azimuth about u)
        dtheta = (torch.rand(Bp, n_vv, device=device, dtype=dtype) * 2 - 1) * max_rad  # (-max,+max)
        dphi   = (torch.rand(Bp, n_vv, device=device, dtype=dtype) * 2 - 1) * max_rad

        # Original direction from target to camera
        dir0 = torch.nn.functional.normalize(C - G, dim=-1, eps=1e-9)  # (B,n,3)

        def rodrigues(axis, angle):
            # axis: (B,n,3), angle: (B,n)
            a = torch.nn.functional.normalize(axis, dim=-1, eps=1e-9)
            ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
            zeros = torch.zeros_like(ax)
            K = torch.stack([
                torch.stack([    zeros, -az,     ay], dim=-1),
                torch.stack([      az, zeros,   -ax], dim=-1),
                torch.stack([     -ay,   ax,  zeros], dim=-1),
            ], dim=-2)  # (B,n,3,3)

            c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)  # (B,n,1,1)
            s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
            I = torch.eye(3, device=device, dtype=dtype).view(1,1,3,3).expand(a.shape[0], a.shape[1], 3, 3)
            aaT = a.unsqueeze(-1) * a.unsqueeze(-2)  # (B,n,3,3)
            return c * I + (1.0 - c) * aaT + s * K   # (B,n,3,3)

        # Apply two small rotations: first around u (azimuth), then around v (elevation)
        R_az = rodrigues(u, dphi)          # rotate around 'u'
        R_el = rodrigues(v, dtheta)        # then around 'v'
        R_delta = R_el @ R_az              # (B,n,3,3)

        new_dir = (R_delta @ dir0.unsqueeze(-1)).squeeze(-1)         # (B,n,3)
        new_dir = torch.nn.functional.normalize(new_dir, dim=-1, eps=1e-9)

        # Keep the same radius r and target G, move camera center on the sphere
        # C_new = G + r * new_dir  # (B,n,3)
                                               # (B,n,3)
        C_new = G + rad * new_dir  # (B,n,3)

        # New forward to keep looking at the same G
        f = torch.nn.functional.normalize(G - C_new, dim=-1, eps=1e-9)          # (B,n,3)

        # Preserve original roll: compute right from original up0, then recompute up to be orthogonal to f
        r = torch.nn.functional.normalize(torch.cross(up0, f, dim=-1), dim=-1, eps=1e-9)  # (B,n,3)
        up = torch.nn.functional.normalize(torch.cross(f, r, dim=-1), dim=-1, eps=1e-9)   # (B,n,3)

        # Camera-to-world (columns are axes): [r, up, f], then world->cam is transpose
        R_c2w = torch.stack([r, up, f], dim=-1)                                   # (B,n,3,3)
        R_new = R_c2w.transpose(-2, -1).contiguous()                              # (B,n,3,3)

        # New translation: t' = -R' * C'
        t_new = -(R_new @ C_new.unsqueeze(-1)).squeeze(-1)                        # (B,n,3)

        Extr_all[..., :3] = R_new
        Extr_all[..., 3]  = t_new



        # ---- render predicted depth under (Kv_all, Extr_all) ----
        try:
            pred = self.mesh_helper.render_lots_of_stuff(
                vertices_pred, Kv_all, Extr_all,
                radial_distortions=None,
                depth_rendering_height=H, depth_rendering_width=W,
                return_depth=True, normalize_normals=False
            )
        except Exception as e:
            print("[VV minimal] Pred rendering failed:", e)
            return torch.tensor(0.0, device=device, dtype=dtype), {}

        depth_pred = pred['depth_images'].view(Bp, n_vv, H, W)

        # ---- render scan depth (same K/Extr) ----
        if isinstance(self.data['v_scan'], list):
            v_list = [v.to(device=device, dtype=dtype) for v in self.data['v_scan']]
        else:
            v_all = self.data['v_scan'].to(device=device, dtype=dtype)
            v_list = [v_all[i] for i in range(v_all.shape[0])]
        if isinstance(self.data['f_scan'], list):
            f_list = [f.to(device=device) for f in self.data['f_scan']]
        else:
            f_all = self.data['f_scan'].to(device=device)
            f_list = [f_all[i] for i in range(f_all.shape[0])]

        Bscan = min(len(v_list), len(f_list))
        if Bscan >= Bp:
            v_scan_list = v_list[:Bp]; f_scan_list = f_list[:Bp]
        else:
            reps = (Bp + Bscan - 1) // Bscan
            v_scan_list = (v_list * reps)[:Bp]
            f_scan_list = (f_list * reps)[:Bp]

        depth_scan_chunks = []
        for i in range(Bp):
            v_i, f_i = v_scan_list[i], f_scan_list[i]
            mh_i = MeshHelper(num_vertices=v_i.shape[0], faces=f_i.detach().cpu().numpy())
            try:
                out_i = mh_i.render_lots_of_stuff(
                    v_i.unsqueeze(0), Kv_all[i:i+1], Extr_all[i:i+1],
                    radial_distortions=None,
                    depth_rendering_height=H, depth_rendering_width=W,
                    return_depth=True, normalize_normals=False
                )
                depth_scan_chunks.append(out_i['depth_images'].view(1, n_vv, H, W))
            except Exception as e:
                print(f"[VV minimal] Scan rendering failed @sample {i}:", e)
                depth_scan_chunks.append(torch.zeros(1, n_vv, H, W, device=device, dtype=dtype))
        depth_scan = torch.cat(depth_scan_chunks, dim=0)

        # ---- pointmaps & robust loss (with dataset-specific rotated views) ----
        vis = ((depth_pred > 0.0) & (depth_scan > 0.0)).float()

        rotated_views_for_pointmaps = [6, 7]

        p_pred = depth_to_pointmap_robust(
            depth_pred, K=Kv_all, extr=Extr_all, rotated_views=rotated_views_for_pointmaps
        ).permute(0, 1, 4, 2, 3)
        p_scan = depth_to_pointmap_robust(
            depth_scan, K=Kv_all, extr=Extr_all, rotated_views=rotated_views_for_pointmaps
        ).permute(0, 1, 4, 2, 3)

        # keep your Z scaling convention
        p_pred = p_pred.clone(); p_scan = p_scan.clone()
        p_pred[:, :, 2] /= 30.0; p_scan[:, :, 2] /= 30.0

        # loss_pm, _ = calculate_map_loss(
        #     p_pred.reshape(Bp * n_vv, 3, H, W),
        #     p_scan.reshape(Bp * n_vv, 3, H, W),
        #     mask=vis.reshape(Bp * n_vv, H, W),
        #     robust=True, gmo_sigma=10
        # )
        loss_pm, pm_loss_pp = calculate_map_loss(
            p_pred.reshape(Bp * n_vv, 3, H, W),
            p_scan.reshape(Bp * n_vv, 3, H, W),
            mask=vis.reshape(Bp * n_vv, H, W),
            robust=True, gmo_sigma=10
        )

        # cache minimal vis state
        self.vv_K    = Kv_all.detach().cpu()
        self.vv_extr = Extr_all.detach().cpu()
        self.vv_size = (H, W)

        # --- NEW: cache pointmaps and per-pixel loss for visualization ---
        self.vv_pointmaps_pred = p_pred.detach().cpu()              # (B, n_vv, 3, H, W)
        self.vv_pointmaps_scan = p_scan.detach().cpu()              # (B, n_vv, 3, H, W)
        self.vv_pm_loss_per_pixel = pm_loss_pp.view(Bp, n_vv, H, W).detach().cpu()
        self.vv_vis_mask = vis.detach().cpu()                       # (B, n_vv, H, W)
        # print(loss_pm, weight)

        return loss_pm * weight, {}


    # def compute_virtual_views_loss( # v3 1 !important
    #     self,
    #     vertices_pred,
    #     # zoom_range=(1.10, 1.90),   # we are training with this the local
    #     zoom_range=(1.40, 1.90),   # fx, fy *= s
    #     rot_deg=2.5,              # random ±rot_deg around each cam axis (cam frame)
    #     weight=None
    # ):
    #     import torch, math
    #     device = vertices_pred.device
    #     dtype  = vertices_pred.dtype

    #     if weight is None:
    #         weight = getattr(self.args, 'weight_virtual_views', 10.0)
    #     if weight <= 0.0:
    #         return torch.tensor(0.0, device=device, dtype=dtype), {}

    #     Bp, V, _ = vertices_pred.shape
    #     H = int(self.inputs['images'].shape[-2])
    #     W = int(self.inputs['images'].shape[-1])

    #     K_real    = self.inputs['camera_intrinsics']   # (B_inp, N, 3, 3)
    #     Extr_real = self.inputs['camera_extrinsics']   # (B_inp, N, 3, 4)
    #     B_inp, N_real = K_real.shape[:2]

    #     # batch-align
    #     if B_inp != Bp:
    #         rep = int((Bp + B_inp - 1) // B_inp)
    #         K_real    = K_real.repeat(rep, 1, 1, 1)[:Bp]
    #         Extr_real = Extr_real.repeat(rep, 1, 1, 1)[:Bp]

    #     # drop the last two (back views)
    #     keep_idx = torch.arange(max(0, N_real - 2), device=device)
    #     n_vv = keep_idx.numel()
    #     if n_vv == 0:
    #         return torch.tensor(0.0, device=device, dtype=dtype), {}

    #     Kv_all   = K_real[:, keep_idx].clone()         # (B, n_vv, 3, 3)
    #     Extr_all = Extr_real[:, keep_idx].clone()      # (B, n_vv, 3, 4)

    #     # ---- zoom intrinsics (fx, fy) ----
    #     s_lo, s_hi = float(zoom_range[0]), float(zoom_range[1])
    #     s = (s_lo + (s_hi - s_lo) * torch.rand(Bp, n_vv, device=device, dtype=dtype))
    #     Kv_all[..., 0, 0] = Kv_all[..., 0, 0] * s
    #     Kv_all[..., 1, 1] = Kv_all[..., 1, 1] * s
    #     # cx, cy unchanged

    #     # ---- small random rotation in camera frame; keep camera center fixed ----
    #     R = Extr_all[..., :3]                  # (B,n,3,3)
    #     t = Extr_all[..., 3]                   # (B,n,3)

    #     # random Euler angles (roll z, pitch y, yaw x)
    #     max_rad = math.radians(float(rot_deg))
    #     ax = (torch.rand(Bp, n_vv, device=device, dtype=dtype) * 2*max_rad) - max_rad  # yaw (x)
    #     ay = (torch.rand(Bp, n_vv, device=device, dtype=dtype) * 2*max_rad) - max_rad  # pitch (y)
    #     az = (torch.rand(Bp, n_vv, device=device, dtype=dtype) * 2*max_rad) - max_rad  # roll (z)

    #     cx, sx = torch.cos(ax), torch.sin(ax)
    #     cy, sy = torch.cos(ay), torch.sin(ay)
    #     cz, sz = torch.cos(az), torch.sin(az)

    #     # Rx, Ry, Rz (B,n,3,3)
    #     Rx = torch.stack([
    #         torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
    #         torch.stack([torch.zeros_like(cx),        cx,                 -sx],            dim=-1),
    #         torch.stack([torch.zeros_like(cx),        sx,                  cx],            dim=-1),
    #     ], dim=-2)
    #     Ry = torch.stack([
    #         torch.stack([      cy, torch.zeros_like(cy),  sy], dim=-1),
    #         torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
    #         torch.stack([     -sy, torch.zeros_like(cy),  cy], dim=-1),
    #     ], dim=-2)
    #     Rz = torch.stack([
    #         torch.stack([ cz, -sz, torch.zeros_like(cz)], dim=-1),
    #         torch.stack([ sz,  cz, torch.zeros_like(cz)], dim=-1),
    #         torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    #     ], dim=-2)

    #     # cam-frame delta: R_delta = Rz @ Ry @ Rx
    #     R_delta = Rz @ Ry @ Rx

    #     # keep center C invariant -> R' = R_delta @ R, t' = R_delta @ t
    #     R_new = R_delta @ R
    #     t_new = (R_delta @ t.unsqueeze(-1)).squeeze(-1)
    #     Extr_all[..., :3] = R_new
    #     Extr_all[..., 3]  = t_new

    #     # ---- render predicted depth under (Kv_all, Extr_all) ----
    #     try:
    #         pred = self.mesh_helper.render_lots_of_stuff(
    #             vertices_pred, Kv_all, Extr_all,
    #             radial_distortions=None,
    #             depth_rendering_height=H, depth_rendering_width=W,
    #             return_depth=True, normalize_normals=False
    #         )
    #     except Exception as e:
    #         print("[VV minimal] Pred rendering failed:", e)
    #         return torch.tensor(0.0, device=device, dtype=dtype), {}

    #     depth_pred = pred['depth_images'].view(Bp, n_vv, H, W)

    #     # ---- render scan depth (same K/Extr) ----
    #     if isinstance(self.data['v_scan'], list):
    #         v_list = [v.to(device=device, dtype=dtype) for v in self.data['v_scan']]
    #     else:
    #         v_all = self.data['v_scan'].to(device=device, dtype=dtype)
    #         v_list = [v_all[i] for i in range(v_all.shape[0])]
    #     if isinstance(self.data['f_scan'], list):
    #         f_list = [f.to(device=device) for f in self.data['f_scan']]
    #     else:
    #         f_all = self.data['f_scan'].to(device=device)
    #         f_list = [f_all[i] for i in range(f_all.shape[0])]

    #     Bscan = min(len(v_list), len(f_list))
    #     if Bscan >= Bp:
    #         v_scan_list = v_list[:Bp]; f_scan_list = f_list[:Bp]
    #     else:
    #         reps = (Bp + Bscan - 1) // Bscan
    #         v_scan_list = (v_list * reps)[:Bp]
    #         f_scan_list = (f_list * reps)[:Bp]

    #     depth_scan_chunks = []
    #     for i in range(Bp):
    #         v_i, f_i = v_scan_list[i], f_scan_list[i]
    #         mh_i = MeshHelper(num_vertices=v_i.shape[0], faces=f_i.detach().cpu().numpy())
    #         try:
    #             out_i = mh_i.render_lots_of_stuff(
    #                 v_i.unsqueeze(0), Kv_all[i:i+1], Extr_all[i:i+1],
    #                 radial_distortions=None,
    #                 depth_rendering_height=H, depth_rendering_width=W,
    #                 return_depth=True, normalize_normals=False
    #             )
    #             depth_scan_chunks.append(out_i['depth_images'].view(1, n_vv, H, W))
    #         except Exception as e:
    #             print(f"[VV minimal] Scan rendering failed @sample {i}:", e)
    #             depth_scan_chunks.append(torch.zeros(1, n_vv, H, W, device=device, dtype=dtype))
    #     depth_scan = torch.cat(depth_scan_chunks, dim=0)

    #     # ---- pointmaps & robust loss (with rotated_views=[]) ----
    #     vis = ((depth_pred > 0.0) & (depth_scan > 0.0)).float()

    #     p_pred = depth_to_pointmap_robust(
    #         depth_pred, K=Kv_all, extr=Extr_all, rotated_views=[]
    #     ).permute(0, 1, 4, 2, 3)
    #     p_scan = depth_to_pointmap_robust(
    #         depth_scan, K=Kv_all, extr=Extr_all, rotated_views=[]
    #     ).permute(0, 1, 4, 2, 3)

    #     # keep your Z scaling convention
    #     p_pred = p_pred.clone(); p_scan = p_scan.clone()
    #     p_pred[:, :, 2] /= 30.0; p_scan[:, :, 2] /= 30.0

    #     # loss_pm, _ = calculate_map_loss(
    #     #     p_pred.reshape(Bp * n_vv, 3, H, W),
    #     #     p_scan.reshape(Bp * n_vv, 3, H, W),
    #     #     mask=vis.reshape(Bp * n_vv, H, W),
    #     #     robust=True, gmo_sigma=10
    #     # )
    #     loss_pm, pm_loss_pp = calculate_map_loss(
    #         p_pred.reshape(Bp * n_vv, 3, H, W),
    #         p_scan.reshape(Bp * n_vv, 3, H, W),
    #         mask=vis.reshape(Bp * n_vv, H, W),
    #         robust=True, gmo_sigma=10
    #     )

    #     # cache minimal vis state
    #     self.vv_K    = Kv_all.detach().cpu()
    #     self.vv_extr = Extr_all.detach().cpu()
    #     self.vv_size = (H, W)

    #     # --- NEW: cache pointmaps and per-pixel loss for visualization ---
    #     self.vv_pointmaps_pred = p_pred.detach().cpu()              # (B, n_vv, 3, H, W)
    #     self.vv_pointmaps_scan = p_scan.detach().cpu()              # (B, n_vv, 3, H, W)
    #     self.vv_pm_loss_per_pixel = pm_loss_pp.view(Bp, n_vv, H, W).detach().cpu()
    #     self.vv_vis_mask = vis.detach().cpu()                       # (B, n_vv, H, W)
    #     # print(loss_pm, weight)

    #     return loss_pm * weight, {}


    # def compute_virtual_views_loss( # v 4
    #     self,
    #     vertices_pred,
    #     # zoom_range=(1.10, 1.90),   # we are training with this the local
    #     zoom_range=(1.50, 2.50),   # fx, fy *= s
    #     rot_deg=2.5,              # random ±rot_deg around each cam axis (cam frame)
    #     weight=None
    # ):
    #     import torch, math
    #     device = vertices_pred.device
    #     dtype  = vertices_pred.dtype

    #     if weight is None:
    #         weight = getattr(self.args, 'weight_virtual_views', 10.0)
    #     if weight <= 0.0:
    #         return torch.tensor(0.0, device=device, dtype=dtype), {}

    #     Bp, V, _ = vertices_pred.shape
    #     H = int(self.inputs['images'].shape[-2])
    #     W = int(self.inputs['images'].shape[-1])

    #     K_real    = self.inputs['camera_intrinsics']   # (B_inp, N, 3, 3)
    #     Extr_real = self.inputs['camera_extrinsics']   # (B_inp, N, 3, 4)
    #     B_inp, N_real = K_real.shape[:2]

    #     # batch-align
    #     if B_inp != Bp:
    #         rep = int((Bp + B_inp - 1) // B_inp)
    #         K_real    = K_real.repeat(rep, 1, 1, 1)[:Bp]
    #         Extr_real = Extr_real.repeat(rep, 1, 1, 1)[:Bp]

    #     # drop the last two (back views)
    #     keep_idx = torch.arange(max(0, N_real), device=device)
    #     n_vv = keep_idx.numel()
    #     if n_vv == 0:
    #         return torch.tensor(0.0, device=device, dtype=dtype), {}

    #     Kv_all   = K_real[:, keep_idx].clone()         # (B, n_vv, 3, 3)
    #     Extr_all = Extr_real[:, keep_idx].clone()      # (B, n_vv, 3, 4)

    #     # ---- zoom intrinsics (fx, fy) ----
    #     s_lo, s_hi = float(zoom_range[0]), float(zoom_range[1])
    #     s = (s_lo + (s_hi - s_lo) * torch.rand(Bp, n_vv, device=device, dtype=dtype))
    #     Kv_all[..., 0, 0] = Kv_all[..., 0, 0] * s
    #     Kv_all[..., 0, 1] = Kv_all[..., 0, 1] * s
    #     Kv_all[..., 1, 1] = Kv_all[..., 1, 1] * s
    #     Kv_all[..., 1, 0] = Kv_all[..., 1, 0] * s
    #     # cx, cy unchanged

    #     # ---- small random rotation in camera frame; keep camera center fixed ----
    #     # [DELETE the old R_delta-based code block entirely]
    #     # R = Extr_all[..., :3]
    #     # t = Extr_all[..., 3]
    #     # ... ax/ay/az, R_delta, R_new, t_new ...

    #     # ---- NEW: Dolly/orbit sampler while keeping the same look-at target ----
    #     R = Extr_all[..., :3].contiguous()         # (B, n, 3, 3) world->cam
    #     t = Extr_all[..., 3].contiguous()          # (B, n, 3)

    #     # Camera center in world: C = -R^T t
    #     Rt = R.transpose(-2, -1)                   # (B, n, 3, 3)
    #     C = -(Rt @ t.unsqueeze(-1)).squeeze(-1)    # (B, n, 3)

    #     # Original forward (optical axis) in world.
    #     # Convention: camera looks along +Z_cam. So ez_cam = [0,0,1].
    #     ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    #     ez = ez.view(1, 1, 3, 1).expand(R.shape[0], R.shape[1], 3, 1)      # (B,n,3,1)
    #     f0 = (Rt @ ez).squeeze(-1)                                         # (B,n,3), world forward
    #     f0 = torch.nn.functional.normalize(f0, dim=-1, eps=1e-9)

    #     # Original up in world (to preserve roll). For image coords (x right, y down),
    #     # up_cam is -Y: [0,-1,0]. If your renderer uses +Y up, change to [0,1,0].
    #     up_cam = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    #     up_cam = up_cam.view(1,1,3,1).expand(R.shape[0], R.shape[1], 3, 1)  # (B,n,3,1)
    #     up0 = (Rt @ up_cam).squeeze(-1)                                     # (B,n,3), world up
    #     # In case up0 ~ f0, fix degeneracy with a global up
    #     global_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1,1,3).expand_as(up0)
    #     colin = (torch.abs((up0 * f0).sum(-1)) > 0.99).unsqueeze(-1)        # (B,n,1)
    #     up0 = torch.where(colin, global_up, up0)
    #     up0 = torch.nn.functional.normalize(up0, dim=-1, eps=1e-9)

    #     # Choose a gaze target G ≈ intersection of the original optical axis with the mesh region.
    #     # Use predicted-vertex centroid as an anchor for depth along f0:
    #     mesh_ctr = vertices_pred.mean(dim=1, keepdim=True)                  # (B,1,3)
    #     mesh_ctr = mesh_ctr.expand(-1, R.shape[1], -1)                      # (B,n,3)
    #     s = ((mesh_ctr - C) * f0).sum(-1, keepdim=True)                     # (B,n,1) signed distance along f0
    #     G = C + s * f0                                                      # (B,n,3)
    #     rad = torch.norm(C - G, dim=-1, keepdim=True).clamp_min(1e-9)  # (B,n,1)

    #     # Build an orthonormal basis (u,v,f0) for the tangent plane around f0
    #     u = torch.nn.functional.normalize(torch.cross(up0, f0, dim=-1), dim=-1, eps=1e-9)  # (B,n,3)
    #     v = torch.nn.functional.normalize(torch.cross(f0,  u, dim=-1), dim=-1, eps=1e-9)   # (B,n,3)

    #     # --- Spherical orbit around G by small angles (no K change) ---
    #     max_deg = getattr(self.args, 'vv_sphere_deg', 15.0)   # e.g., up to ±8°
    #     max_rad = math.radians(max_deg)

    #     # Sample angular offsets: dtheta (elevation about v), dphi (azimuth about u)
    #     dtheta = (torch.rand(Bp, n_vv, device=device, dtype=dtype) * 2 - 1) * max_rad  # (-max,+max)
    #     dphi   = (torch.rand(Bp, n_vv, device=device, dtype=dtype) * 2 - 1) * max_rad

    #     # Original direction from target to camera
    #     dir0 = torch.nn.functional.normalize(C - G, dim=-1, eps=1e-9)  # (B,n,3)

    #     def rodrigues(axis, angle):
    #         # axis: (B,n,3), angle: (B,n)
    #         a = torch.nn.functional.normalize(axis, dim=-1, eps=1e-9)
    #         ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    #         zeros = torch.zeros_like(ax)
    #         K = torch.stack([
    #             torch.stack([    zeros, -az,     ay], dim=-1),
    #             torch.stack([      az, zeros,   -ax], dim=-1),
    #             torch.stack([     -ay,   ax,  zeros], dim=-1),
    #         ], dim=-2)  # (B,n,3,3)

    #         c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)  # (B,n,1,1)
    #         s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    #         I = torch.eye(3, device=device, dtype=dtype).view(1,1,3,3).expand(a.shape[0], a.shape[1], 3, 3)
    #         aaT = a.unsqueeze(-1) * a.unsqueeze(-2)  # (B,n,3,3)
    #         return c * I + (1.0 - c) * aaT + s * K   # (B,n,3,3)

    #     # Apply two small rotations: first around u (azimuth), then around v (elevation)
    #     R_az = rodrigues(u, dphi)          # rotate around 'u'
    #     R_el = rodrigues(v, dtheta)        # then around 'v'
    #     R_delta = R_el @ R_az              # (B,n,3,3)

    #     new_dir = (R_delta @ dir0.unsqueeze(-1)).squeeze(-1)         # (B,n,3)
    #     new_dir = torch.nn.functional.normalize(new_dir, dim=-1, eps=1e-9)

    #     # Keep the same radius r and target G, move camera center on the sphere
    #     # C_new = G + r * new_dir  # (B,n,3)
    #                                            # (B,n,3)
    #     C_new = G + rad * new_dir  # (B,n,3)

    #     # New forward to keep looking at the same G
    #     f = torch.nn.functional.normalize(G - C_new, dim=-1, eps=1e-9)          # (B,n,3)

    #     # Preserve original roll: compute right from original up0, then recompute up to be orthogonal to f
    #     r = torch.nn.functional.normalize(torch.cross(up0, f, dim=-1), dim=-1, eps=1e-9)  # (B,n,3)
    #     up = torch.nn.functional.normalize(torch.cross(f, r, dim=-1), dim=-1, eps=1e-9)   # (B,n,3)

    #     # Camera-to-world (columns are axes): [r, up, f], then world->cam is transpose
    #     R_c2w = torch.stack([r, up, f], dim=-1)                                   # (B,n,3,3)
    #     R_new = R_c2w.transpose(-2, -1).contiguous()                              # (B,n,3,3)

    #     # New translation: t' = -R' * C'
    #     t_new = -(R_new @ C_new.unsqueeze(-1)).squeeze(-1)                        # (B,n,3)

    #     Extr_all[..., :3] = R_new
    #     Extr_all[..., 3]  = t_new



    #     # ---- render predicted depth under (Kv_all, Extr_all) ----
    #     try:
    #         pred = self.mesh_helper.render_lots_of_stuff(
    #             vertices_pred, Kv_all, Extr_all,
    #             radial_distortions=None,
    #             depth_rendering_height=H, depth_rendering_width=W,
    #             return_depth=True, normalize_normals=False
    #         )
    #     except Exception as e:
    #         print("[VV minimal] Pred rendering failed:", e)
    #         return torch.tensor(0.0, device=device, dtype=dtype), {}

    #     depth_pred = pred['depth_images'].view(Bp, n_vv, H, W)

    #     # ---- render scan depth (same K/Extr) ----
    #     if isinstance(self.data['v_scan'], list):
    #         v_list = [v.to(device=device, dtype=dtype) for v in self.data['v_scan']]
    #     else:
    #         v_all = self.data['v_scan'].to(device=device, dtype=dtype)
    #         v_list = [v_all[i] for i in range(v_all.shape[0])]
    #     if isinstance(self.data['f_scan'], list):
    #         f_list = [f.to(device=device) for f in self.data['f_scan']]
    #     else:
    #         f_all = self.data['f_scan'].to(device=device)
    #         f_list = [f_all[i] for i in range(f_all.shape[0])]

    #     Bscan = min(len(v_list), len(f_list))
    #     if Bscan >= Bp:
    #         v_scan_list = v_list[:Bp]; f_scan_list = f_list[:Bp]
    #     else:
    #         reps = (Bp + Bscan - 1) // Bscan
    #         v_scan_list = (v_list * reps)[:Bp]
    #         f_scan_list = (f_list * reps)[:Bp]

    #     depth_scan_chunks = []
    #     for i in range(Bp):
    #         v_i, f_i = v_scan_list[i], f_scan_list[i]
    #         mh_i = MeshHelper(num_vertices=v_i.shape[0], faces=f_i.detach().cpu().numpy())
    #         try:
    #             out_i = mh_i.render_lots_of_stuff(
    #                 v_i.unsqueeze(0), Kv_all[i:i+1], Extr_all[i:i+1],
    #                 radial_distortions=None,
    #                 depth_rendering_height=H, depth_rendering_width=W,
    #                 return_depth=True, normalize_normals=False
    #             )
    #             depth_scan_chunks.append(out_i['depth_images'].view(1, n_vv, H, W))
    #         except Exception as e:
    #             print(f"[VV minimal] Scan rendering failed @sample {i}:", e)
    #             depth_scan_chunks.append(torch.zeros(1, n_vv, H, W, device=device, dtype=dtype))
    #     depth_scan = torch.cat(depth_scan_chunks, dim=0)

    #     # ---- pointmaps & robust loss (with rotated_views=[]) ----
    #     vis = ((depth_pred > 0.0) & (depth_scan > 0.0)).float()

    #     p_pred = depth_to_pointmap_robust(
    #         depth_pred, K=Kv_all, extr=Extr_all, rotated_views=[6,7]
    #     ).permute(0, 1, 4, 2, 3)
    #     p_scan = depth_to_pointmap_robust(
    #         depth_scan, K=Kv_all, extr=Extr_all, rotated_views=[6,7]
    #     ).permute(0, 1, 4, 2, 3)

    #     # keep your Z scaling convention
    #     p_pred = p_pred.clone(); p_scan = p_scan.clone()
    #     p_pred[:, :, 2] /= 30.0; p_scan[:, :, 2] /= 30.0

    #     # loss_pm, _ = calculate_map_loss(
    #     #     p_pred.reshape(Bp * n_vv, 3, H, W),
    #     #     p_scan.reshape(Bp * n_vv, 3, H, W),
    #     #     mask=vis.reshape(Bp * n_vv, H, W),
    #     #     robust=True, gmo_sigma=10
    #     # )
    #     loss_pm, pm_loss_pp = calculate_map_loss(
    #         p_pred.reshape(Bp * n_vv, 3, H, W),
    #         p_scan.reshape(Bp * n_vv, 3, H, W),
    #         mask=vis.reshape(Bp * n_vv, H, W),
    #         robust=True, gmo_sigma=10
    #     )

    #     # cache minimal vis state
    #     self.vv_K    = Kv_all.detach().cpu()
    #     self.vv_extr = Extr_all.detach().cpu()
    #     self.vv_size = (H, W)

    #     # --- NEW: cache pointmaps and per-pixel loss for visualization ---
    #     self.vv_pointmaps_pred = p_pred.detach().cpu()              # (B, n_vv, 3, H, W)
    #     self.vv_pointmaps_scan = p_scan.detach().cpu()              # (B, n_vv, 3, H, W)
    #     self.vv_pm_loss_per_pixel = pm_loss_pp.view(Bp, n_vv, H, W).detach().cpu()
    #     self.vv_vis_mask = vis.detach().cpu()                       # (B, n_vv, H, W)
    #     # print(loss_pm, weight)

    #     return loss_pm * weight, {}

    def visualize_virtual_views(self, out_dir_suffix='vv', sample_idx=0, max_views=8):
        """
        Minimal visualization with pointmaps:
          Row 1: [Pred render | Scan render]
          Row 2: [Pred PointMap | Scan PointMap | PointMap Loss]
        Requires compute_virtual_views_loss to have run this step.
        """
        import os, numpy as np, cv2

        need = ['vv_K','vv_extr','vv_size','global_points','faces','data',
                'vv_pointmaps_pred','vv_pointmaps_scan','vv_pm_loss_per_pixel']
        if not all(hasattr(self, k) for k in need):
            print('[visualize_virtual_views] Missing caches; run compute_virtual_views_loss first.')
            return

        H, W = self.vv_size
        B = int(self.vv_K.shape[0])
        b = int(max(0, min(sample_idx, B-1)))

        Kb    = self.vv_K[b].numpy()         # (n_vv, 3,3)
        Extrb = self.vv_extr[b].numpy()      # (n_vv, 3,4)
        n_vv  = Kb.shape[0]

        # Pred mesh (current model)
        faces_pred = self.faces.cpu().numpy()
        verts_pred = self.global_points[b].detach().cpu().numpy()

        # Scan mesh
        if isinstance(self.data['v_scan'], list):
            scan_v = self.data['v_scan'][b].cpu().numpy()
            scan_f = self.data['f_scan'][b].cpu().numpy()
        else:
            scan_v = self.data['v_scan'][b].cpu().numpy()
            scan_f = self.data['f_scan'][b].cpu().numpy()

        # Cached pointmaps & loss
        p_pred_b = self.vv_pointmaps_pred[b].numpy()      # (n_vv, 3, H, W)
        p_scan_b = self.vv_pointmaps_scan[b].numpy()      # (n_vv, 3, H, W)
        pm_loss_b = self.vv_pm_loss_per_pixel[b].numpy()  # (n_vv, H, W)

        os.makedirs(os.path.join(self.directory_output, f'{out_dir_suffix}_images'), exist_ok=True)

        def put_label(img_rgb, txt):
            img = img_rgb.copy()
            cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            return img

        rows_all = []
        show_n = min(n_vv, int(max_views))
        for v in range(show_n):
            K = Kb[v]; extr = Extrb[v]

            cam_args = dict(
                camera_intrinsics=K,
                camera_extrinsics=extr,
                radial_distortion=None,
                frustum={'near': 0.01, 'far': 3000.0},
                image_size=(H, W)
            )

            # --- Row 1: renders ---
            scan_img = render_mesh(vertices=scan_v,   faces=scan_f,       vertex_colors=None, **cam_args)
            pred_img = render_mesh(vertices=verts_pred, faces=faces_pred, vertex_colors=None, needs_projection=True, **cam_args)
            row1 = np.hstack([
                put_label(np.clip(pred_img, 0, 255).astype(np.uint8), 'Pred render'),
                put_label(np.clip(scan_img, 0, 255).astype(np.uint8), 'Scan render'),
            ])

            # --- Row 2: pointmaps & loss (match your other vis: use GT min/max) ---
            # p_*: (3, H, W)
            p_gt   = p_scan_b[v]
            p_pred = p_pred_b[v]

            # Colorize GT first to get consistent channel-wise min/max
            pm_gt_rgb, ch_min, ch_max = pointmap_to_rgb(p_gt)
            pm_pr_rgb, _, _ = pointmap_to_rgb(p_pred, ch_min, ch_max)

            # Per-pixel loss -> heatmap
            pm_loss = pm_loss_b[v]
            # print(pm_loss.min(), pm_loss.max())
            pm_loss_vis = dist_to_rgb(pm_loss.reshape(-1), min_dist=0.0, max_dist=3.0).reshape(H, W, 3)
            # print(pm_loss_vis)
            pm_loss_vis = (pm_loss_vis * 255).astype(np.uint8)

            row2 = np.hstack([
                put_label(pm_pr_rgb.astype(np.uint8), 'Pred PointMap'),
                put_label(pm_gt_rgb.astype(np.uint8), 'Scan PointMap'),
                put_label(pm_loss_vis.astype(np.uint8), 'PointMap Loss'),
            ])

            rows_all.append(np.hstack([row1, row2]))

        grid = np.vstack(rows_all).astype(np.uint8)
        out_path = os.path.join(self.directory_output, f'{out_dir_suffix}_images/all_views_{self.global_step}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    def _save_color_grid(self, idx, n_select=4, nrows=2, ncols=4):
        """
        Save a 2x4 grid with 4 randomly chosen color images (pads remaining tiles white).
        Output: .../val_images/color_grid_{idx:04d}.jpg
        """
        out_dir = os.path.join(self.directory_output, 'val_images')
        os.makedirs(out_dir, exist_ok=True)

        sel_view_ids = list(range(8))

        imgs = []
        for vid in sel_view_ids:
            im = to_numpy(self.data['color_images'][0][vid].permute(1, 2, 0))
            im = self.dataset_train.denormalize_image(im)
            im = (255 * im).astype(np.uint8)
            imgs.append(im)

        if not imgs:
            return

        H, W, C = imgs[0].shape
        pad = np.ones((H, W, C), dtype=np.uint8) * 255
        while len(imgs) < nrows * ncols:
            imgs.append(pad.copy())

        rows = [np.hstack(imgs[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
        grid = np.vstack(rows)

        out_path = os.path.join(out_dir, f'color_grid_{idx:04d}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))



    def refine_eval_bigloop(self):
        """
        Per-sample refinement & evaluation with:
        1) per-sample prints (losses + P2S stats),
        2) running summary,
        3) FINAL summary that aggregates ALL raw P2S distances per region (like Tester).
        """
        import copy, csv, numpy as np, pprint, os, torch
        from collections import defaultdict
        from utils import utils
        
        # ---- config ----
        refinement_steps = self.args.refinement_steps 
        refinement_lr    = self.args.refinement_lr
        refine_layers    = 'all' 
        save_csv         = True
        refine_vis       = self.args.refine_vis
        # print('Params for refinement:', refinement_steps, refinement_lr, refine_layers, 'Save CSV:', save_csv)
        # raise
        # Figure out how many items to loop over
        split_list = utils.load_json(self.args.train_data_list_fname)
        num_items  = len(split_list)
        print(f'[Refine] Total items: {num_items}')


        validation_losses_per_epoch = {}
        validation_losses = {}

        vertices_for_viz = []
        faces_for_viz = []
        labels_for_viz = []
        VIZ_PAPER = True

        steps_to_gather = [5, 10, 20, 50, 100]
        # steps_to_gather = [0, 50, -100] #, 50, -100]
        gather_intermediate = True

        # for idx in [6,147]: #range(147,num_items,10):
        # the guy with the teeth bug (and the teaser is 1506 in the test set ! )
        start = getattr(self.args, 'refine_start_index', 0)
        end   = getattr(self.args, 'refine_end_index', -1)
        print('Running refinement from index', start, 'to', end)

        # assert end > 0, "Please set refine_end_index to a positive integer."

        if end < 0:
            end = num_items - 1

        # radek = [7870,223,7410,7030,7350,7880]
        for idx in range(start, end + 1): #[2,3,27,35,88] : # range(start, end + 1):
        # for idx in radek: # range(start,end + 1): #[2,3,27,35,88] : # range(start, end + 1):
            # for idx in range(0,num_items): #,10):
            print(f'\n[Refine] >>> Sample {idx+1}/{num_items}')
            # 1) re-init dataset for this index (creates loaders for the single item)

            self.register_dataset(index=idx) #+300*5)

            # grab the only batch (enforce batch_size=1 in your config; if not, we take the first item)
            batch = next(iter(self.dataloader_train))
            if batch['color_images'].shape[0] != 1:
                batch = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in batch.items()}

            # feed w/o augs -> stable refinement target
            self.feed_data(batch, mode='val')

            # 2) clone the LOCAL (or global) model you want to refine
            base = self.local_model.module if (hasattr(self, 'local_model') and isinstance(self.local_model, torch.nn.DataParallel)) else getattr(self, 'local_model', None)
            refine_model = copy.deepcopy(base).to(self.device)
            coarse_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            coarse_model = coarse_model.to(self.device)

            # --- LoRA for local refinement ---
            use_lora_local: bool = False
            if use_lora_local:
                lora_r: int = 4
                lora_alpha: int = 16
                lora_dropout: float = 0.0
                lora_lr: float = 1e-3

                from models.model_aligner.prototypes.lora_local import inject_lora_3d

                # If you know where the 3D convs live, you can target just that submodule.
                # Commonly: self.local_model.module.local_densify_net.local_net (or similar).

                # Adapt ALL Conv3d/ConvTranspose3d inside the local model:
                inject_lora_3d(refine_model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                refine_model = refine_model.to(self.device)

                # Freeze everything, then unfreeze only LoRA params
                for n, p in refine_model.named_parameters():
                    p.requires_grad_(False)
                for n, p in refine_model.named_parameters():
                    if "down.weight" in n or "up.weight" in n or "upT.weight" in n:  # LoRA params
                        p.requires_grad_(True)

                # Optional: sanity print
                lora_trainable = sum(p.numel() for p in refine_model.parameters() if p.requires_grad)
                print(f"[LoRA] Trainable params in refine_model: {lora_trainable/1e6:.3f} M")

                # Optimizer just for LoRA
                lora_params = [p for p in refine_model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(lora_params, lr=refinement_lr)
                params = lora_params
            else:
                # 3) choose trainable params & optimizer
                if refinement_lr == 1e-3:
                    print('Using special refinement LR scheme for grid refiner @1e-3 LR')
                    base_params, special_params = [], []
                    for name, param in refine_model.named_parameters(): 
                        if 'grid_refiner' in name:
                            special_params.append(param)
                        else:
                            base_params.append(param)

                    opt = torch.optim.AdamW([
                                                {'params': base_params, 'lr': refinement_lr, 'group_id': 'base'}, 
                                                {'params': special_params, 'lr': 1e-4, 'group_id': 'special'}])
                    params = base_params + special_params
                else:            
                    params = []
                    for name, p in refine_model.named_parameters():
                        p.requires_grad = (refine_layers == 'all') or ('grid_refiner' in name)
                        if p.requires_grad:
                            params.append(p)
                    opt = torch.optim.AdamW(params, lr=refinement_lr)

            if VIZ_PAPER:
                self.forward_tempeh()

            if 0 in steps_to_gather:
                with torch.no_grad():
                    refine_model.eval()
                    coarse_model.eval()
                    self.forward(coarse_model, refine_model)   # <— pass models
                    self.compute_losses()

                    # --- Points2Surface for Base ---
                    distances = self._points2surface_metric(self.global_points)

                    # --- NEW: save per-sample per-step distances ---
                    save_dir = os.path.join(self.directory_output, "refine_distances")
                    os.makedirs(save_dir, exist_ok=True)
                    t = 0
                    sample_tag = f"sample_{idx:05d}_step_{t:03d}.npy"
                    np.save(os.path.join(save_dir, sample_tag), distances)
                    # -----------------------------------------------

            if -100 in steps_to_gather:
                self.global_points = self.target_vertices
                self.compute_losses()

                # --- Points2Surface for Base ---
                distances = self._points2surface_metric(self.global_points)

                # --- NEW: save per-sample per-step distances ---
                save_dir = os.path.join(self.directory_output, "refine_distances")
                os.makedirs(save_dir, exist_ok=True)
                t = -100
                sample_tag = f"sample_{idx:05d}_step_{t:03d}.npy"
                np.save(os.path.join(save_dir, sample_tag), distances)
                # -----------------------------------------------

            # 4) N refinement steps on this single sample
            # from tqdm import tqdm
            # for t in tqdm(range(refinement_steps)):
            for t in range(refinement_steps):
                self.losses = {}
                refine_model.eval()
                coarse_model.eval()
                self.forward(coarse_model, refine_model)

                # self.export_mesh('val', idx)
                # raise
                self.compute_losses()

                # if t % 5 == 0:
                #     d = {}
                #     for key in self.losses:
                #         d['%s/train' % key] = to_numpy(self.losses[key])
                    
                #     pprint.pprint(d)
                    

                if VIZ_PAPER:
                    if t == 0:
                        # vertices_for_viz.append(self.coarse_points_tempeh[0].cpu())
                        # faces_for_viz.append(self.faces.cpu())
                        # labels_for_viz.append(f'TEMPEH GLOBAL')

                        # vertices_for_viz.append(self.coarse_points[0].detach().cpu())
                        # faces_for_viz.append(self.faces.detach().cpu())
                        # labels_for_viz.append(f'Initial Coarse')

                        vertices_for_viz.append(self.global_points_tempeh[0].detach().cpu())
                        faces_for_viz.append(self.faces.detach().cpu())
                        labels_for_viz.append(f'TEMPEH LOCAL')

                        vertices_for_viz.append(self.global_points[0].detach().cpu())
                        faces_for_viz.append(self.faces.detach().cpu())
                        labels_for_viz.append(f'Initial Refined')


                    # if t == 10 or t == 20 or t == 50 or t == refinement_steps -1:
                    # if t == refinement_steps -1:
                    if t in steps_to_gather:
                        vertices_for_viz.append(self.global_points[0].detach().cpu())
                        faces_for_viz.append(self.faces.detach().cpu())
                        labels_for_viz.append(f'TTO {t}')

                opt.zero_grad()
                self.loss.backward()
                if self.args.gradient_max_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, self.args.gradient_max_norm)
                opt.step()
                print(f'  [Refine] Step {t+1}/{refinement_steps}, Loss: {self.loss.item():.6f}')
            


                if gather_intermediate:
                    if t in steps_to_gather:
                        # 5) final eval pass (no grad) and collect metrics
                        with torch.no_grad():
                            refine_model.eval()
                            coarse_model.eval()
                            self.forward(coarse_model, refine_model)   # <— pass models
                            self.compute_losses()

                            # --- Points2Surface for Base ---
                            distances = self._points2surface_metric(self.global_points)

                            # --- NEW: save per-sample per-step distances ---
                            save_dir = os.path.join(self.directory_output, "refine_distances")
                            os.makedirs(save_dir, exist_ok=True)

                            sample_tag = f"sample_{idx:05d}_step_{t:03d}.npy"
                            np.save(os.path.join(save_dir, sample_tag), distances)
                            # -----------------------------------------------


                            for key in distances:
                                distance_key = f'Points2Surface distance {key}'
                                # if distance_key not in validation_losses:
                                    # validation_losses[distance_key] = []

                                vals = distances[key]
                                if t not in validation_losses_per_epoch:
                                    validation_losses_per_epoch[t] = {}
                                if distance_key not in validation_losses_per_epoch[t]:
                                    validation_losses_per_epoch[t][distance_key] = []

                                validation_losses_per_epoch[t][distance_key].extend(vals)
                                # print lengths
                                print(f'Key: {distance_key}, vals length: {len(vals)}, iteration: {t},  total length: {len(validation_losses_per_epoch[t][distance_key])}')
                        
                        # Final summary (convert & print once)

                        # if self.args.refine_start_index == 0 and self.args.refine_end_index == -1:
                        #     if idx % 5 == 0 or (idx == num_items - 1): 
                        #         # print('Current IDX', idx)
                        #         # print('\n[Refine] Running validation summary so far:')
                        #         # for t in validation_losses_per_epoch.keys():
                        #         #     print('At step', t)
                        #         #     self._print_validation_table(validation_losses_per_epoch[t])
                                                    

                        #         print("\n==== FINAL COMBINED VALIDATION TABLE ====")
                        #         csv_out = os.path.join(self.directory_output, "validation_combined.csv")
                        #         self._print_validation_table_multi(validation_losses, validation_losses_per_epoch, csv_path=csv_out)


            if VIZ_PAPER and self.args.refine_start_index == 0 and self.args.refine_end_index == -1:
                # self.forward_tempeh()
                # vertices_for_viz.append(self.global_points_tempeh[0].cpu())
                # faces_for_viz.append(self.faces.cpu())
                # labels_for_viz.append(f'TEMPEH')

                print('Saving refinement visualization for paper...')
                # raise
                vertices_for_viz.append(self.data['v_registration'][0].cpu())
                faces_for_viz.append(self.data['f_registration'][0].cpu())
                labels_for_viz.append(f'Traditional Registration')


                vertices_for_viz.append(self.data['v_scan'][0].cpu())
                faces_for_viz.append(self.data['f_scan'][0].cpu())
                labels_for_viz.append(f'Initial Scan')

                self.visualize_multiple_front(
                    vertices_list=vertices_for_viz, faces_list=faces_for_viz, labels=labels_for_viz
                )

                vertices_for_viz = []
                faces_for_viz = []
                
            # lightweight vis (optional)
            if refine_vis and (idx % self.args.refine_visualization_freq == 0 or idx == num_items - 1):
                if self.args.refine_start_index == 0 and self.args.refine_end_index == -1:
                    self._save_color_grid(idx)
                    self.export_mesh('val', idx)

            if self.args.refine_start_index == 0 and self.args.refine_end_index == -1:
                # 5) final eval pass (no grad) and collect metrics
                with torch.no_grad():
                    refine_model.eval()
                    coarse_model.eval()
                    self.forward(coarse_model, refine_model)   # <— pass models
                    self.compute_losses()

                    # --- Points2Surface for Base ---
                    distances = self._points2surface_metric(self.global_points)
                    for key in distances:
                        distance_key = f'Points2Surface distance {key}'
                        if distance_key not in validation_losses:
                            validation_losses[distance_key] = []

                        vals = distances[key]
                        validation_losses[distance_key].extend(vals)
                        # print lengths
                        print(f'Key: {distance_key}, vals length: {len(vals)}, total length: {len(validation_losses[distance_key])}')
                
            # Final summary (convert & print once)

            # if idx % 5 == 0 or (idx == num_items - 1): 
            #     print('Current IDX', idx)
            #     print('\n[Refine] Running validation summary so far:')
            # self._print_validation_table(validation_losses)
        

            del refine_model
            torch.cuda.empty_cache()

            self.global_step += 1
        print('Finished with params', refinement_lr, refinement_steps, refine_layers)
        # self._print_validation_table(validation_losses)


    def _points2surface_metric(self, points_to_use):
        losses = {}
        for i in range(len(self.data['v_scan'])):
            # print(len(self.data['v_scan'][i].shape), points_to_use.shape, len(self.data['v_registration']))
            scan_vertices = self.data['v_scan'][i].to(self.device).unsqueeze(0)
            predicted_vertices = points_to_use[i].unsqueeze(0).float()
            predicted_faces = self.data['f_reg_global'][0].to(self.device)
            assert predicted_faces.shape[0] == 9976
            registration_vertices = self.data['v_registration'][i].to(self.device).float().unsqueeze(0) 

            with torch.no_grad():
                distances_dict = compute_s2m_distance(scan_vertices, predicted_vertices, predicted_faces, 
                                                    masks=self.flame_masks_triangles,
                                                    registration_vertices_for_regions=registration_vertices,
                                                    exclude_regions=['boundary'], exclude_from_full=['scalp'])
                for key in distances_dict:
                    if key not in losses:
                        losses[key] = []
                    losses[key].extend(to_numpy(distances_dict[key]).tolist())

        return losses

    def _print_validation_table_multi(self, overall_losses, per_step_losses, csv_path=None):
        """
        Build one wide table:
        rows  = metric names (e.g., 'Points2Surface distance face', 'Points2Surface distance full', ...)
        cols  = ['Overall mean','Overall median','Overall std',
                't=5 mean','t=5 median','t=5 std', ... for every recorded step]
        `overall_losses`: dict[str, List[float]]
        `per_step_losses`: dict[int, dict[str, List[float]]]  (step -> metric -> values)
        """
        import numpy as np
        import pandas as pd

        def summarize(loss_dict):
            out = {}
            for k, vals in loss_dict.items():
                arr = np.asarray(vals, dtype=np.float32)
                if arr.size == 0:
                    out[k] = dict(mean=np.nan, median=np.nan, std=np.nan)
                else:
                    out[k] = dict(mean=float(arr.mean()),
                                median=float(np.median(arr)),
                                std=float(arr.std()))
            return out

        overall_sum = summarize(overall_losses)

        # Collect all metric keys across overall + every step
        all_metrics = set(overall_sum.keys())
        for t, d in per_step_losses.items():
            all_metrics.update(d.keys())
        all_metrics = sorted(all_metrics)

        # Build columns
        cols = ["Overall mean", "Overall median", "Overall std"]
        ordered_steps = sorted(per_step_losses.keys())
        for t in ordered_steps:
            cols += [f"t={t} mean", f"t={t} median", f"t={t} std"]

        # Fill rows
        table = []
        for metric in all_metrics:
            row = []
            # overall
            o = overall_sum.get(metric, dict(mean=np.nan, median=np.nan, std=np.nan))
            row += [o["mean"], o["median"], o["std"]]
            # per-step
            for t in ordered_steps:
                ssum = summarize(per_step_losses[t]).get(metric, dict(mean=np.nan, median=np.nan, std=np.nan))
                row += [ssum["mean"], ssum["median"], ssum["std"]]
            table.append([metric] + row)

        colnames = ["Metric"] + cols
        df = pd.DataFrame(table, columns=colnames)

        print("\nValidation results — ONE BIG TABLE (mean / median / std):")
        print(df.to_string(index=False, float_format="%.4f"))

        if csv_path:
            try:
                df.to_csv(csv_path, index=False, float_format="%.6f")
                print(f"\nSaved combined CSV to: {csv_path}")
            except Exception as e:
                print(f"Could not save CSV to {csv_path}: {e}")


    def _print_validation_table(self, validation_losses):
        """Pretty-print validation results with mean, median, and std for Base, PLIKS, and PLIKS-FLAME."""

        def summarize_losses(loss_dict):
            summary = {}
            for key in loss_dict:
                arr = np.array(loss_dict[key])
                summary[key] = {
                    "mean": np.mean(arr),
                    "median": np.median(arr),
                    "std": np.std(arr),
                }
            return summary

        d_base = summarize_losses(validation_losses)

        # Collect all metric keys
        all_keys = sorted(set(d_base.keys())) #

        rows = []
        for key in all_keys:
            rows.append([
                key,
                d_base.get(key, {}).get("mean", float("nan")),
                d_base.get(key, {}).get("median", float("nan")),
                d_base.get(key, {}).get("std", float("nan")),
            ])

        df = pd.DataFrame(
            rows,
            columns=[
                "Metric",
                "Base Mean", "Base Median", "Base Std",
            ]
        )

        print("\nValidation results (CSV, mean / median / std):")
        print(df.to_csv(index=False, float_format="%.4f").strip())


# -----------------------------------------------------------------------------
def run(config_fname=''):
    parser = TrainOptions()
    args = parser.parse(config_filename=config_fname)
    parser.print_options()

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    if args.wandb:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project='tempeh_final_refine', config=vars(args), name=args.experiment_id)
        wandb.run.log_code("trainer/")

    trainer = Trainer(args, device)
    trainer.initialize()

    trainer.refine_eval_bigloop()

if __name__ == '__main__':
    run()
    print('Done')
