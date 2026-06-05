import os
import wandb
import torch
import numpy as np
from torch.autograd import Variable
from utils.utils import to_numpy
from utils.point_to_point_loss import PointToPointLoss
from utils.point_to_surface_loss import PointToSurfaceLoss, compute_s2m_distance
from utils.edge_loss import EdgeLoss
from utils.mesh_helper import MeshHelper, depth_to_pointmap_robust
from trainer.base_trainer import BaseTrainer
from option_handler.train_options_global import TrainOptions
from models.FLAME.FLAME import FLAME
import cv2
import trimesh
from trainer.utils import pixels_to_uv
from utils.losses import calculate_map_loss, calculate_gradient_map_loss
import torch.nn.functional as F
import pandas as pd
from modules.volumetric_feature_sampler import VolumetricFeatureSampler
from trainer.visualizer import RefinerVisualizer



class Trainer(BaseTrainer):

    def __init__(self, args, device):
        super().__init__(args)
        self.args = args
        self.device = device

        self.no_jaw = True

        from models.FLAME.pliks_flame import PliksFlameSolver

        self.flame = FLAME(flame_model_path='assets/FLAME2023/flame2023_no_jaw.pkl', no_jaw=True).to(device)

        self.pliks_solver = PliksFlameSolver(self.flame, locked_joint_ids=torch.Tensor([2]).long()).to(device)

        self.faces = self.flame.faces_tensor.cpu()

        self.mesh_helper = MeshHelper(num_vertices=5023, faces=self.faces)

        self.load_flame_masks()

        self.unit_factor = 1000.0

        if self.args.to_meters:
            self.args.global_voxel_inc = self.args.global_voxel_inc / self.unit_factor
            self.args.global_origin = [x/self.unit_factor for x in self.args.global_origin]


        self.rotated_views_global = [6, 7]  # hardcoded for 8-view setup
        self.non_rotated_views_global = [i for i in range(8) if i not in self.rotated_views_global]
            
            
        self.dense_landmarks_weights_mask = torch.zeros((5023,), dtype=torch.float32).to(self.device)
        for key in self.vertex_masks_tempeh.keys():
            if 'vertex_count' in key:
                continue
            for idx in self.vertex_masks_tempeh[key].tolist():
                self.dense_landmarks_weights_mask[idx] = self.args.dense_mask_weights.get(key)
        self.dense_landmarks_weights_mask = self.dense_landmarks_weights_mask.unsqueeze(0).unsqueeze(0)  # (1,1,V)
        

        mp_indices_for_loss = np.arange(0,105).tolist()
        indices_to_ignore = [17,10,14,12,18,11,16,15,19,13,3,9,6,5,8,1,2,4,7,
                    52,53,54,55,56,57,58,59,60,61,62,63,64]

        self.mp_indices_for_loss = [x for x in mp_indices_for_loss if x not in indices_to_ignore]

        self.visualizer = self._make_visualizer()

    def _make_visualizer(self):
        renderer = getattr(self.args, 'visualization_renderer', 'pyrender').lower()
        if renderer == 'pyrender':
            return RefinerVisualizer(self)
        if renderer == 'blender':
            from trainer.visualizer_blender import RefinerBlenderVisualizer
            return RefinerBlenderVisualizer(self)
        raise ValueError(f"Unsupported visualization renderer: {renderer}")

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

    def register_dataset(self, index=0):
        from utils import mesh_sampling, utils

        if not self.args.val_data_list_fname:
            raise ValueError('Refinement requires -vdl/--val-data-list-fname.')
        self.split_list = utils.load_json(self.args.val_data_list_fname)

        # create a temporary
        import json
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
            image_dir=self.args.image_directory,
            calibration_dir=self.args.calibration_directory,
            scan_dir=self.args.scan_directory,
            registration_root_dir=self.args.processed_directory,
            image_resize_factor=self.args.image_resize_factor,
            brightness_sigma=self.args.brightness_sigma,
            image_file_ext=self.args.image_file_ext,
            dense_landmarks_dir=self.args.dense_landmarks_dir,
            dense_semantic_landmarks_dir=self.args.dense_semantic_landmarks_dir,
            normals_dir=self.args.normals_image_directory,
            to_meters=self.args.to_meters,
            depths_dir=self.args.depths_image_directory,
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


    def feed_data(self, data, mode='train'):
        suffix = '_augmented' if mode == 'train' else ''

        depth_maps = data['color_images_depth' + suffix]

        self.current_subjects = data['subject']
        self.current_sequences = data['sequence']
        self.current_frames = data['frame']

        views = np.arange(data['color_images'].shape[1])

        # positions in the subsampled 'views' array
        self.rotated_views = [i for i, v in enumerate(views) if v in self.rotated_views_global]
        self.non_rotated_views = [i for i, v in enumerate(views) if v not in self.rotated_views_global]


        self.visualization_view_ids = views
        self.data = data
        self.inputs = {
            'images': Variable(data['color_images' + suffix][:, views, ...]).to(self.device),
            'camera_intrinsics': Variable(data['color_camera_intrinsics' + suffix][:, views, ...]).to(self.device),
            'camera_extrinsics': Variable(data['color_camera_extrinsics'][:, views, ...]).to(self.device),
            'camera_distortions': Variable(data['color_camera_distortions'][:, views, ...]).to(self.device),
        }
        self.camera_centers = Variable(data['color_camera_centers'][:, views, ...]).to(self.device)

        self.target_vertices = data['v_registration'].to(self.device)

        if depth_maps is not None:

            self.depth_maps_gt = Variable(depth_maps[:, views, ...]).to(self.device) #/self.unit_factor
            if self.args.to_meters:
                self.depth_maps_gt = self.depth_maps_gt / self.unit_factor
            self.normal_maps_gt = Variable(data['color_images_normals' + suffix][:, views, ...]).to(self.device)

            self.normal_maps_gt_norm = self.normal_maps_gt / self.normal_maps_gt.norm(dim=2, keepdim=True).clamp(min=1e-12) * (self.depth_maps_gt.squeeze(-1) > 0.0).unsqueeze(2)

            self.normal_maps_gt_01 = (self.normal_maps_gt + 1.0) / 2.0 * (self.depth_maps_gt.squeeze(-1) > 0.0).unsqueeze(2)


            self.pointmaps_gt = depth_to_pointmap_robust(
                depth=self.depth_maps_gt.squeeze(-1),                # (B,V,H,W)
                K=self.inputs['camera_intrinsics'],
                extr=self.inputs['camera_extrinsics'],
            ).permute(0, 1, 4, 2, 3)                                 # (B,V,H,W,3)

        dense_mask_key = 'color_camera_dense_landmarks_masks' + suffix

        self.landmarks_dense = data['color_camera_dense_landmarks' + suffix][:, views].to(self.device)
        if dense_mask_key in data and data[dense_mask_key] is not None:
            self.landmarks_dense_mask = data[dense_mask_key][:, views].to(self.device).squeeze(-1)
        else:
            self.landmarks_dense_mask = torch.ones(self.landmarks_dense.shape[:2], device=self.device, dtype=torch.float32)

        h, w = self.inputs['images'].shape[-2:]
        self.landmarks_dense_uv = pixels_to_uv(self.landmarks_dense, h, w)

        self.landmarks_dense_mediapipe = data['color_camera_dense_mediapipe_landmarks' + suffix][:, views].to(self.device)
        self.landmarks_dense_mediapipe_uv = pixels_to_uv(self.landmarks_dense_mediapipe, h, w)

        self.global_landmarks_projected_dense = None

        if self.args.enable_local:
            # for coarse, downsample the images
            BB, L, C, H, W = data['color_images'].shape
            downsampled_images = F.interpolate(
                data['color_images'].to(self.device).view(BB * L, C, H, W),
                scale_factor=0.5, mode='bilinear', align_corners=False,
            ).view(BB, L, C, H // 2, W // 2)

            downscaled_intrinsics = data['color_camera_intrinsics'].to(self.device).clone()
            downscaled_intrinsics[..., :2, :] *= 0.5

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


    def compute_vertex_losses(self, vertices, target_vertices=None, suffix=""):
        losses = {}

        registrations_are_here = False
        if target_vertices is None:
            registrations_are_here = True
            print('Using self.target_vertices for vertex losses computation')
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

        if not self.args.enable_diff_rendering:
            return losses
        
        visibility_mask = (self.depth_maps_gt[:, :, :, :].squeeze(-1) > 0.0).unsqueeze(2)
        
        try:
            with torch.cuda.amp.autocast(enabled=False):
                # print(vertices.shape, self.inputs['camera_intrinsics'].shape, self.inputs['camera_extrinsics'].shape, 'sfd')
                pred = self.mesh_helper.render_normals_and_depth(
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
            return losses
        
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
        
        self.normal_maps_pred = normal_maps_pred
        self.depth_maps_pred = depth_maps_pred
        self.pointmaps_pred = pointmaps_pred
        
        # Compute rendering losses
        normals_loss, normals_loss_per_pixel = calculate_map_loss(
            normal_maps_pred_norm.view(bs * num_views, c, height, width),
            self.normal_maps_gt_norm.view(bs * num_views, c, height, width),
            mask=visibility_mask.view(bs * num_views, height, width),
            robust=True,
            gmo_sigma=10
        )


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
        self.normals_grad_mag_pred = grad_mag_pred.view(bs, num_views, height, width)
        self.normals_grad_mag_gt   = grad_mag_gt.view(bs, num_views, height, width)
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

        return losses

    def compute_landmarks_loss(self, vertices, suffix="", gt_landmarks=None, gt_mask=None):
        """Compute landmarks loss for given vertices (scale/translation/rotation-invariant relative pairwise distances)."""
        is_dense = "dense" in suffix.lower()
        is_mediapipe = "mediapipe" in suffix.lower()

        loss_weight = getattr(self.args, 'weight_dense_landmarks', 0.0) if is_dense else self.args.weight_landmarks
        if loss_weight <= 0.0 and not is_dense:
            return torch.tensor(0.0, device=vertices.device), None

        num_views = self.inputs['camera_extrinsics'].shape[1]
        vertices_flat = vertices.unsqueeze(1).repeat(1, num_views, 1, 1).flatten(0, 1)

        if is_mediapipe:
            landmarks3d = self.flame.select_mediapipe(vertices_flat)
        elif is_dense:
            landmarks3d = vertices_flat
        else:
            landmarks3d = self.flame.seletec_3d68(vertices_flat)

        _, _, _, height, width = self.data['color_images_augmented'].shape
        landmarks_x, landmarks_y = VolumetricFeatureSampler.project(
            landmarks3d,
            self.inputs['camera_intrinsics'].flatten(0, 1),
            self.inputs['camera_extrinsics'].flatten(0, 1),
            self.inputs['camera_distortions'].flatten(0, 1),
            height=height, width=width,
        )

        # Pixel-space coords: always used for vis; also the loss target when not to_meters.
        px = (landmarks_x + 1.0) * (width - 1.) / 2.0
        py = (landmarks_y + 1.0) * (height - 1.) / 2.0
        landmarks_pixel = torch.stack((px, py), dim=-1).view(-1, num_views, px.shape[1], 2)
        landmarks_projected_vis = landmarks_pixel.detach()

        if self.args.to_meters:
            landmarks_projected = torch.stack((landmarks_x, landmarks_y), dim=-1).view(-1, num_views, landmarks_x.shape[1], 2)
        else:
            landmarks_projected = landmarks_pixel

        nr = self.non_rotated_views
        pred_v = landmarks_projected[:, nr]
        gt_v = gt_landmarks[:, nr, :, :2]

        if is_mediapipe:
            pred_pts = pred_v[:, :, self.mp_indices_for_loss]
            gt_pts = gt_v[:, :, self.mp_indices_for_loss]
        elif is_dense:
            pred_pts = pred_v
            gt_pts = gt_v
            if gt_mask is not None:
                gt_mask = gt_mask[:, nr]
        else:
            pred_pts = pred_v[:, :, :17]
            gt_pts = gt_v[:, :, :17]

        # Scale/translation/rotation-invariant relative pairwise-distance loss.
        N = pred_pts.shape[2]
        eps = 1e-8
        D_pred = torch.cdist(pred_pts, pred_pts, p=2)                 # (B, V, N, N)
        D_gt   = torch.cdist(gt_pts,   gt_pts,   p=2)
        tri = torch.triu(torch.ones(N, N, device=pred_pts.device, dtype=torch.bool), diagonal=1)
        D_pred_ut = D_pred[..., tri]                                   # (B, V, K)
        D_gt_ut   = D_gt[...,   tri]

        scale = D_gt_ut.median(dim=-1, keepdim=True).values.clamp_min(eps)  # (B, V, 1)
        rel_diff = (D_pred_ut / scale - D_gt_ut / scale).abs()

        tau_rel = getattr(self.args, 'landmarks_rel_tau', 0.0)
        if tau_rel > 0.0:
            rel_diff = torch.clamp(rel_diff - tau_rel, min=0.0)

        per_view_loss = rel_diff.pow(2).mean(dim=-1)                   # (B, V)

        if gt_mask is not None:
            mask_f = gt_mask.to(per_view_loss.dtype)
            landmarks_loss = (per_view_loss * mask_f).sum() / mask_f.sum().clamp(min=1)
        else:
            landmarks_loss = per_view_loss.mean()

        return landmarks_loss * loss_weight, landmarks_projected_vis

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


        all_losses['landmarks_loss_dense_out'], self.global_landmarks_projected_dense_out = self.compute_landmarks_loss(
            self.global_points,
            suffix="_dense",
            gt_landmarks=self.landmarks_dense_uv if self.args.to_meters else self.landmarks_dense,
            gt_mask=self.landmarks_dense_mask
        )

        all_losses['landmarks_loss_dense_mediapipe_out'], self.global_landmarks_projected_dense_mediapipe_out = self.compute_landmarks_loss(
            self.global_points,
            suffix="_mediapipe",
            gt_landmarks=self.landmarks_dense_mediapipe_uv if self.args.to_meters else self.landmarks_dense_mediapipe,
        )

        if self.args.enable_diff_rendering:
            global_rendering_losses = self.compute_rendering_losses(self.global_points, suffix="")
            all_losses.update(global_rendering_losses)
            

        if hasattr(self, 'pliks_out') and self.pliks_out is not None:
            pliks_losses, V_flame_mm = self.pliks_solver.compute_regularizers(
                self.pliks_out, self.global_points, self.flame,
                weight_shape=getattr(self.args, 'weight_shape_regularization', 0.0),
                weight_expression=getattr(self.args, 'weight_expression_regularization', 0.0),
                weight_pose=getattr(self.args, 'weight_pose_regularization', 0.0),
                weight_vertices=self.args.weight_vertices_regularizer_pliks,
                weight_vertices_edge=self.args.weight_vertices_regularizer_pliks_edge,
                edge_loss_fn=self.edge_loss_function,
                no_jaw=self.no_jaw,
                to_meters=self.args.to_meters,
                unit_factor=self.unit_factor,
            )
            all_losses.update(pliks_losses)
            self.pliks_out['V_flame_mm'] = V_flame_mm

            all_losses['landmarks_loss_dense'], self.global_landmarks_projected_dense = self.compute_landmarks_loss(
                V_flame_mm,
                suffix="_dense",
                gt_landmarks=self.landmarks_dense_uv if self.args.to_meters else self.landmarks_dense,
                gt_mask=self.landmarks_dense_mask
            )

            all_losses['landmarks_loss_dense_mediapipe'], self.global_landmarks_projected_dense_mediapipe = self.compute_landmarks_loss(
                self.global_points,
                suffix="_mediapipe",
                gt_landmarks=self.landmarks_dense_mediapipe_uv if self.args.to_meters else self.landmarks_dense_mediapipe,
            )

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
        self.normals_grad_loss = all_losses.get('normals_grad_loss', 0.0)

        # Create losses dictionary for logging
        self.main_losses = {
            'Total loss': total_loss,
            'Points loss': self.points_loss,
            'Normals loss': self.normals_loss,
            'PointMaps loss': self.point_maps_loss,
            'Depth loss': self.depth_maps_loss,
            'Points2Surface loss': self.points2surface_loss,
            'Edge regularizer': self.edge_regularizer_loss,
            'Landmarks loss (dense)': all_losses.get('landmarks_loss_dense', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'Landmarks loss (dense) out': all_losses.get('landmarks_loss_dense_out', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'β regularizer (PLIKS)': all_losses.get('beta_regularizer_pliks', 0.0) if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'ψ regularizer (PLIKS)': all_losses.get('exp_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Pose regularizer (PLIKS)': all_losses.get('pose_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices edge regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks_edge', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Normals grad loss': self.normals_grad_loss if hasattr(self, 'normals_grad_loss') else 0.0,
            'Landmarks loss (dense mediapipe)': all_losses.get('landmarks_loss_dense_mediapipe', 0.0),
            'Landmarks loss (dense mediapipe) out': all_losses.get('landmarks_loss_dense_mediapipe_out', 0.0)

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
        import copy, csv, numpy as np, os, torch
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
        if not self.args.val_data_list_fname:
            raise ValueError('Refinement requires -vdl/--val-data-list-fname.')
        split_list = utils.load_json(self.args.val_data_list_fname)
        num_items  = len(split_list)
        print(f'[Refine] Total items: {num_items}')


        validation_losses_per_epoch = {}
        validation_losses = {}

        steps_to_gather = [5, 10, 20, 50, 100]
        # steps_to_gather = [0, 50, -100] #, 50, -100]
        gather_intermediate = True

        for idx in range(num_items):
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
            vertices_for_viz, faces_for_viz, labels_for_viz = [], [], []

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
                    

                if refine_vis:
                    if t == 0:
                        vertices_for_viz.append(self.global_points[0].detach().cpu())
                        faces_for_viz.append(self.faces.detach().cpu())
                        labels_for_viz.append('Initial Refined')
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
                        

            # lightweight vis (optional) + front-row mesh comparison
            if refine_vis and (idx % self.args.refine_visualization_freq == 0 or idx == num_items - 1):
                vertices_for_viz.append(self.data['v_registration'][0].cpu())
                faces_for_viz.append(self.data['f_registration'][0].cpu())
                labels_for_viz.append('Traditional Registration')

                vertices_for_viz.append(self.data['v_scan'][0].cpu())
                faces_for_viz.append(self.data['f_scan'][0].cpu())
                labels_for_viz.append('Initial Scan')

                self.visualizer.multi_front(
                    vertices_list=vertices_for_viz, faces_list=faces_for_viz,
                    labels=labels_for_viz, dataset_idx=idx,
                )

                self._save_color_grid(idx)
                self.export_mesh('val', idx)

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
