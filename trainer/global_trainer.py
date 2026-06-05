import os
import wandb
import torch
import numpy as np
from utils.utils import print_memory, to_numpy, get_time_string
from utils.point_to_point_loss import PointToPointLoss
from utils.point_to_surface_loss import PointToSurfaceLoss
from utils.edge_loss import EdgeLoss
from utils.mesh_helper import MeshHelper, depth_to_pointmap_robust
from trainer.base_trainer import BaseTrainer
from option_handler.train_options_global import TrainOptions
from datasets.face_align_dataset_mpi_grid import FaceAlignDatasetMPI
from models.FLAME.FLAME import FLAME
import pprint
import trimesh
from tqdm import tqdm
from trainer.utils import pixels_to_uv
from utils.losses import calculate_map_loss, _points2surface_metric
import torch.nn.functional as F
from models.FLAME.pliks_flame import PliksFlameSolver
from modules.volumetric_feature_sampler import VolumetricFeatureSampler
from trainer.visualizer import GlobalTrainerVisualizer


class Trainer(BaseTrainer):

    def __init__(self, args, device):
        super().__init__(args)
        self.args = args
        self.device = device

        self.no_jaw = True

        self.flame = FLAME(flame_model_path='assets/FLAME2023/flame2023_no_jaw.pkl', no_jaw=self.no_jaw).to(device)

        self.pliks_solver = PliksFlameSolver(self.flame, locked_joint_ids=torch.Tensor([2]).long()).to(device)

        self.faces = self.flame.faces_tensor.cpu()

        self.mesh_helper = MeshHelper(num_vertices=5023, faces=self.faces)

        self.load_flame_masks()

        self.unit_factor = 1000.0
        if self.args.to_meters:
            self.args.global_voxel_inc = self.args.global_voxel_inc / self.unit_factor
            self.args.global_origin = [x/self.unit_factor for x in self.args.global_origin]

        self.rotated_views_global = [6, 7]  # hardcoded for 8-view setup of FaMoS
        self.non_rotated_views_global = [i for i in range(8) if i not in self.rotated_views_global]
            
        self.dense_landmarks_weights_mask = torch.zeros((5023,), dtype=torch.float32).to(self.device)
        for key in self.vertex_masks_tempeh.keys():
            if 'vertex_count' in key:
                continue
            for idx in self.vertex_masks_tempeh[key].tolist():
                self.dense_landmarks_weights_mask[idx] = self.args.dense_mask_weights.get(key)
        self.dense_landmarks_weights_mask = self.dense_landmarks_weights_mask.unsqueeze(0).unsqueeze(0)  # (1,1,V)

        # landmarks in mediapipe format -> ignore these and focus on lips and eyelids
        # left eyebrow: 17,10,14,12,18,11,16,15,19,13
        # right eyebrow: 3,9,6,5,8,1,2,4,5,7
        # nose: 52,53,54,55,56,57,58,59,60,61,62,63,64

        mp_indices_for_loss = np.arange(0,105).tolist()
        indices_to_ignore = [17,10,14,12,18,11,16,15,19,13,3,9,6,5,8,1,2,4,7,
                    52,53,54,55,56,57,58,59,60,61,62,63,64]

        self.mp_indices_for_loss = [x for x in mp_indices_for_loss if x not in indices_to_ignore]

        self.visualizer = GlobalTrainerVisualizer(self)

    def register_model(self):
        import models.model_aligner.prototypes.model_global_stage as models
        model = models.Model(args=self.args)
        self.model = model.to(self.device)

        self.base_model = models.Model(self.args).to(self.device)
            
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

            self.local_model = model.to(self.device)
            if self.args.pretrained_local_path:
                self.base_local_model = models.Model(args=self.args, mesh_sampler=self.mesh_sampler, feature_net=feature_net)
                self.base_local_model = self.base_local_model.to(self.device)


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

            base_params, special_params = [], []
            for name, param in self.local_model.named_parameters(): 
                if 'grid_refiner' in name:
                    special_params.append(param)
                else:
                    base_params.append(param)
         
            self.optimizer_model_local = torch.optim.AdamW([
                                        {'params': base_params, 'lr': self.args.learning_rate, 'group_id': 'base'}, 
                                        {'params': special_params, 'lr': 1e-4, 'group_id': 'special'}])


    def register_dataset(self):
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
            depths_dir=self.args.depths_image_directory
        )

        self.dataset_train = FaceAlignDatasetMPI(
            data_list_fname=self.args.train_data_list_fname,
            **common_kwargs
        )

        self.dataset_val = FaceAlignDatasetMPI(
            data_list_fname=self.args.val_data_list_fname,
            **common_kwargs
        )

        self.dataloader_train = self.make_data_loader(self.dataset_train, cuda=True, shuffle=True)
        self.dataloader_val = self.make_data_loader(self.dataset_val, cuda=True, shuffle=False)

        print(len(self.dataloader_train), 'train samples', len(self.dataloader_val), 'validation samples')

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
            'images': data['color_images' + suffix][:, views, ...].to(self.device),
            'camera_intrinsics': data['color_camera_intrinsics' + suffix][:, views, ...].to(self.device),
            'camera_extrinsics': data['color_camera_extrinsics'][:, views, ...].to(self.device),
            'camera_distortions': data['color_camera_distortions'][:, views, ...].to(self.device),
        }
        self.camera_centers = data['color_camera_centers'][:, views, ...].to(self.device)

        self.target_vertices = data['v_registration'].to(self.device)

        if depth_maps is not None:
            self.depth_maps_gt = depth_maps[:, views, ...].to(self.device)
            if self.args.to_meters:
                self.depth_maps_gt = self.depth_maps_gt / self.unit_factor
            self.normal_maps_gt = data['color_images_normals' + suffix][:, views, ...].to(self.device)

            self.normal_maps_gt_norm = self.normal_maps_gt / self.normal_maps_gt.norm(dim=2, keepdim=True).clamp(min=1e-12) * (self.depth_maps_gt.squeeze(-1) > 0.0).unsqueeze(2)

            self.normal_maps_gt_01 = (self.normal_maps_gt + 1.0) / 2.0 * (self.depth_maps_gt.squeeze(-1) > 0.0).unsqueeze(2)

            self.pointmaps_gt = depth_to_pointmap_robust(
                depth=self.depth_maps_gt.squeeze(-1),                # (B,V,H,W)
                K=self.inputs['camera_intrinsics'],
                extr=self.inputs['camera_extrinsics'],
                rotated_views=self.rotated_views_global,
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

        if self.args.enable_local:
            # for coarse, downsample the images
            BB, L, C, H, W = data['color_images'].shape
            downsampled_images = F.interpolate(data['color_images'].to(self.device).view(BB * L, C, H, W), scale_factor=0.5, mode='bilinear', align_corners=False).view(BB, L, C, H // 2, W // 2)

            downscaled_intrinsics = data['color_camera_intrinsics'].to(self.device).clone()
            downscaled_intrinsics[..., :2, :] *= 0.5

            self.inputs_coarse = {
                'images': downsampled_images,
                'camera_intrinsics': downscaled_intrinsics,
                'camera_extrinsics': data['color_camera_extrinsics'],
                'camera_distortions': data['color_camera_distortions'],
            }


    def forward(self):
        random_grid = True if self.model.training else False

        if self.args.enable_local:
            # if local, first infer the global model
            with torch.inference_mode():
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                self.coarse_results = self.model(**self.inputs_coarse, random_grid=False)
                self.coarse_points = self.coarse_results['vertices']
            
            results = self.local_model(  images=self.inputs['images'],
                                        camera_intrinsics=self.inputs['camera_intrinsics'],
                                        camera_extrinsics=self.inputs['camera_extrinsics'],
                                        camera_distortions=self.inputs['camera_distortions'],
                                        camera_centers=self.camera_centers,
                                        global_points=self.coarse_results['vertices'],
                                        random_grid=random_grid)

            self.global_points = results[-1]
            
        else:
            results = self.model(**self.inputs, random_grid=random_grid)
            self.global_points = results['vertices']

        
        if getattr(self.args, 'use_pliks_refinement', False):
            pliks_in = (self.global_points)

            if not self.args.to_meters:
                pliks_in = pliks_in/self.unit_factor

            self.pliks_out = self.pliks_solver(
                pliks_in, iters=1, lsq_method='ne', estimate_root=False
            )

            if not self.args.to_meters:
                self.pliks_out['V_fit'] = self.pliks_out['V_fit'] * self.unit_factor  # convert back to mm if needed


    def compute_vertex_losses(self, vertices, target_vertices):
        losses = {}

        # vertices regularization loss
        if self.args.weight_points_recon > 0.0:
            points_loss = self.args.weight_points_recon * self.points_loss_function(vertices, target_vertices)
            losses[f'points_loss'] = points_loss
        else:
            losses[f'points_loss'] = 0.0

        # Edge regularizer loss
        if self.args.weight_edge_regularizer > 0.0:
            edge_loss = self.args.weight_edge_regularizer * self.edge_loss_function(vertices, target_vertices)
            losses[f'edge_regularizer_loss'] = edge_loss
        else:
            losses[f'edge_regularizer_loss'] = 0.0
        
        # Point to surface loss (for MOCHI it is 0)
        if self.args.weight_points2surface > 0.0:
            points2surface_loss = self.args.weight_points2surface * self._points2surface_loss_for_vertices(vertices)
            losses['points2surface_loss'] = points2surface_loss

        return losses

    def compute_rendering_losses(self, vertices):
        bs, num_views, c, height, width = self.inputs['images'].shape
        losses = {}

        visibility_mask = (self.depth_maps_gt[:, :, :, :].squeeze(-1) > 0.0).unsqueeze(2)
        
        try:
            with torch.cuda.amp.autocast(enabled=False):
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
        self.normal_maps_pred = pred['normal_images'].permute(0, 3, 1, 2).view(bs, num_views, c, height, width)
        
        normal_maps_pred_11 = self.normal_maps_pred * 2.0 - 1.0  # to [-1,1]
        # normalize
        normal_maps_pred_norm = normal_maps_pred_11 / (normal_maps_pred_11.norm(dim=2, keepdim=True) + 1e-12) * (pred['depth_images'].view(bs, num_views, 1, height, width) > 0.0)

        self.depth_maps_pred = pred['depth_images'].view(bs, num_views, height, width)
        self.pointmaps_pred = (
            depth_to_pointmap_robust(
                depth=self.depth_maps_pred,
                K=self.inputs['camera_intrinsics'],
                extr=self.inputs['camera_extrinsics'],
                rotated_views=self.rotated_views_global
            )
            .permute(0, 1, 4, 2, 3)
        )
        
        # Compute rendering losses
        normals_loss, normals_loss_per_pixel = calculate_map_loss(
            normal_maps_pred_norm.view(bs * num_views, c, height, width),
            self.normal_maps_gt_norm.view(bs * num_views, c, height, width),
            mask=visibility_mask.view(bs * num_views, height, width),
            robust=True,
            gmo_sigma=10
        )
        
        # Scale Z and compute point-map loss
        p_pred = self.pointmaps_pred.clone()
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
        losses[f'normals_loss'] = normals_loss * self.args.weight_normals_images
        losses[f'point_maps_loss'] = point_maps_loss * self.args.weight_point_maps
        
        self.normals_loss_per_pixel = normals_loss_per_pixel.view(bs, num_views, height, width).detach().cpu()
        self.point_maps_loss_per_pixel = point_maps_loss_per_pixel.view(bs, num_views, height, width).detach().cpu()
    

        return losses


    def compute_landmarks_loss(self, vertices, suffix="", gt_landmarks=None, gt_mask=None):
        """Compute landmarks loss for given vertices."""
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
            diff = (pred_v[:, :, self.mp_indices_for_loss] - gt_v[:, :, self.mp_indices_for_loss]).abs().sum(-1)
        elif is_dense:
            diff = (pred_v - gt_v).pow(2).sum(-1)
            if gt_mask is not None:
                gt_mask = gt_mask[:, nr]
        else:
            diff = (pred_v[:, :, :17] - gt_v[:, :, :17]).pow(2).sum(-1)

        per_view_loss = diff.mean(-1)
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
                predicted_faces = self.faces.to(self.device)
                loss = self.points2surface_loss_function(scan_vertices, predicted_vertices, predicted_faces)
                total_loss += loss
            return total_loss / len(self.data['v_scan'])
        else:
            scan_vertices = self.data['v_scan'].to(self.device)
            predicted_faces = self.faces.to(self.device)
            return self.points2surface_loss_function(scan_vertices, vertices, predicted_faces)

    def compute_losses(self):
        all_losses = {}

        if self.args.enable_local:
            global_vertex_losses = self.compute_vertex_losses(self.global_points, self.coarse_points)
        
            all_losses.update(global_vertex_losses)

        if self.args.enable_diff_rendering:
            global_rendering_losses = self.compute_rendering_losses(self.global_points)
            all_losses.update(global_rendering_losses)
            

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

        if hasattr(self, 'pliks_out') and self.pliks_out is not None:
            pliks_losses, V_flame_mm = self.pliks_solver.compute_regularizers(
                self.pliks_out, self.global_points, self.flame,
                weight_shape=self.args.weight_shape_regularization,
                weight_expression=self.args.weight_expression_regularization,
                weight_pose=self.args.weight_pose_regularization,
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

        total_loss = sum(
            loss
            for loss in all_losses.values()
            if isinstance(loss, torch.Tensor) and not torch.all(loss == 0)
        )

        # Set instance attributes for backward compatibility
        self.loss = total_loss
        self.points_loss = all_losses.get('points_loss', 0.0)
        self.points2surface_loss = all_losses.get('points2surface_loss', 0.0)
        self.edge_regularizer_loss = all_losses.get('edge_regularizer_loss', 0.0)
        self.normals_loss = all_losses.get('normals_loss', 0.0)
        self.point_maps_loss = all_losses.get('point_maps_loss', 0.0)

        # Create losses dictionary for logging
        self.main_losses = {
            'Total loss': total_loss,
            'Points loss': self.points_loss,
            'Normals loss': self.normals_loss,
            'PointMaps loss': self.point_maps_loss,
            'Points2Surface loss': self.points2surface_loss,
            'Edge regularizer': self.edge_regularizer_loss,
            'Landmarks loss (dense)': all_losses.get('landmarks_loss_dense', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'Landmarks loss (dense) out': all_losses.get('landmarks_loss_dense_out', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'β regularizer (PLIKS)': all_losses.get('beta_regularizer_pliks', 0.0) if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'ψ regularizer (PLIKS)': all_losses.get('exp_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Pose regularizer (PLIKS)': all_losses.get('pose_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            't regularizer (PLIKS)': all_losses.get('t_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices edge regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks_edge', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
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


    def train_step(self, data):
        self.losses = {}

        self.feed_data(data)

        for param_group in self.optimizer_model.param_groups:
            lr = param_group['lr']

        self.model.train()
        self.forward()
        self.compute_losses()
        self.backward()

        if self.global_step % self.args.print_frequency == 0:        
            print('%s, step %d, total loss: %f, ' %(get_time_string(), self.global_step, to_numpy(self.loss)))
            d = {'Total loss/train': to_numpy(self.loss)}
            for key in self.losses:
                val = to_numpy(self.losses[key])
                if val != 0.0:
                    d['%s/train' % key] = val

            pprint.pprint(d)

            if self.args.wandb:
                wandb_d = {'Learning rate/train': lr}
                wandb_d.update(d)
                distances = _points2surface_metric(self.global_points, self.data['v_scan'], self.data['v_registration'], self.faces.unsqueeze(0), self.flame_masks_triangles)
                for key in distances:
                    wandb_d['Points2Surface distance %s/train' % key] = np.mean(distances[key])
                    wandb_d['Points2Surface distance %s/train median' % key] = np.median(distances[key])
                    wandb_d['Points2Surface distance %s/train std' % key] = np.std(distances[key])
                wandb.log(wandb_d, step=self.global_step)

        if (self.global_step % self.args.visualize_frequency == 0):
            try:
                self.visualize('train')
                self.export_mesh('train', self.global_step)
            except Exception as e:
                print(f"Error occurred during visualization: {e}")
                raise e
        if (self.global_step > 0) and (self.global_step % self.args.validate_frequency == 0):               
            with torch.no_grad():
                self.validate()
        if (self.global_step > 0) and (self.global_step % self.args.save_frequency == 0):
            self.save_checkpoint()
        

        if self.global_step % self.args.print_frequency == 0:        
            self.cleanup_memory()
            print_memory(self.device, prefix='FW')

        self.global_step += 1

    def validate(self):
        self.losses = {}

        validation_losses = {}

        vis_validation_every = 25

        for i, data in enumerate(self.dataloader_val):
            self.feed_data(data, mode='val')

            self.model.eval()
            if self.args.enable_local:
                self.local_model.eval()

            self.forward()
            self.compute_losses()

            for key in self.losses:
                if key not in validation_losses:
                    validation_losses[key] = []
                validation_losses[key].append(to_numpy(self.losses[key]))

            distances = _points2surface_metric(self.global_points, self.data['v_scan'], self.data['v_registration'], self.faces.unsqueeze(0), self.flame_masks_triangles)
            for key in distances:
                distance_key = 'Points2Surface distance %s' % key
                if distance_key not in validation_losses:
                    validation_losses[distance_key] = np.array([], dtype=np.float32)
                validation_losses[distance_key] = np.concatenate([validation_losses[distance_key], to_numpy(distances[key])])


            if i % vis_validation_every == 0:
                print('Visualizing validation sample %d' % i)
                try:
                    self.visualize('val', val_idx=i)
                except Exception as e:
                    print(f"Error occurred during validation visualization: {e}")

                self.export_mesh('val', i)

                print_memory(self.device, prefix='FW')
            # break

        d = {}
        for key in validation_losses:
            validation_losses[key] = np.array(validation_losses[key])
            # print(validation_losses[key].shape)
            d['%s/validation' % key] = np.mean(validation_losses[key])
            d['%s/validation median' % key] = np.median(validation_losses[key])
            d['%s/validation std' % key] = np.std(validation_losses[key])

        print('Validation results:')
        pprint.pprint(d)
        if self.args.wandb:
            wandb.log(d, step=self.global_step)

    def visualize(self, mode='train', val_idx=0):
        self.visualizer.run(mode=mode, val_idx=val_idx)

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

         
        if hasattr(self, 'global_points') and self.global_points is not None:
            reconstructed_vertices = to_numpy(self.global_points[0]) * to_meter_scale_factor
            mesh_2 = trimesh.Trimesh(vertices=reconstructed_vertices, faces=faces, process=False)
            scene.add_geometry(mesh_2, node_name="reconstructed")


        out_fname = os.path.join(out_sequence_dir, f'{mode}_mesh_{self.global_step}_{idx:04d}.ply')
        scene.export(out_fname.replace('.ply','.glb'))


def run(config_fname=''):
    parser = TrainOptions()
    args = parser.parse(config_filename=config_fname)
    parser.print_options()

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if args.wandb:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))

        wandb.init(project='tempeh_final', config=vars(args), name=args.experiment_id)  # NEW
        wandb.run.log_code("trainer/")  # NEW

    # set trainer
    trainer = Trainer(args, device)
    trainer.initialize()
    if args.evaluate:
        trainer.validate()
    else:
        trainer.run()

if __name__ == '__main__':
    run()
    print('Done')
