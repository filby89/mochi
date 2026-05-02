import os
import wandb
import torch
import numpy as np
from torch.autograd import Variable
from utils.utils import print_memory, to_numpy, get_time_string, add_labels_to_images
from utils.mesh_renderer import render_mesh, dist_to_rgb
from utils.point_to_point_loss import PointToPointLoss
from utils.point_to_surface_loss import PointToSurfaceLoss, compute_s2m_distance
from utils.edge_loss import EdgeLoss
from utils.mesh_helper import MeshHelper, pointmap_to_rgb, depth_to_pointmap_robust
from trainer.base_trainer import BaseTrainer
from option_handler.train_options_global import TrainOptions
from models.FLAME.FLAME import FLAME
import math
import cv2
import pprint
import trimesh
from tqdm import tqdm
from trainer.utils import get_dense_vertex_colors, random_pixel_mask, draw_dense_points
from utils.losses import calculate_map_loss, calculate_gradient_map_loss, _points2surface_metric
import torch.nn.functional as F
from models.FLAME.pliks_flame_2 import PliksFlameSolver, build_R_world_from_segments, world_to_relative_rotations


class Trainer(BaseTrainer):

    def __init__(self, args, device):
        super().__init__(args)
        self.args = args
        self.device = device

        self.no_jaw = True


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
        print(sum(self.dense_landmarks_weights_mask > 0.0), 'vertices with dense landmarks supervision')


        # landmarks mediapipe
        # left eyebrow: 17,10,14,12,18,11,16,15,19,13
        # right eyebrow: 3,9,6,5,8,1,2,4,5,7
        # nose: 52,53,54,55,56,57,58,59,60,61,62,63,64

        mp_indices_for_loss = np.arange(0,105).tolist()
        indices_to_ignore = [17,10,14,12,18,11,16,15,19,13,3,9,6,5,8,1,2,4,7,
                    52,53,54,55,56,57,58,59,60,61,62,63,64]

        self.mp_indices_for_loss = [x for x in mp_indices_for_loss if x not in indices_to_ignore]





    def register_model(self):
        import models.model_aligner.prototypes.model_global_stage as models
        # import models.model_aligner.prototypes.model_global_stage_dino as models
        model = models.Model(args=self.args)
        # model.initialize(init_method='normal')
        self.model = model.to(self.device)

        self.base_model = models.Model(self.args).to(self.device)
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
        from datasets.face_align_dataset_mpi_grid import FaceAlignDatasetMPI as DatasetCls

        common_kwargs = dict(
            dataset_root_dir=self.args.dataset_directory,
            image_dir=self.args.image_directory,
            calibration_dir=self.args.calibration_directory,
            scan_dir=self.args.scan_directory,
            return_full_scan="meshes" in self.args.scan_directory or 'decimated' in self.args.scan_directory,
            registration_root_dir=self.args.processed_directory,
            image_resize_factor=self.args.image_resize_factor,
            mesh_sampler=None,
            scan_vertex_count=self.args.scan_vertex_count,
            brightness_sigma=self.args.brightness_sigma,
            image_file_ext=self.args.image_file_ext,
            dense_landmarks_dir=self.args.dense_landmarks_dir,
            dense_semantic_landmarks_dir=self.args.dense_semantic_landmarks_dir,
            normals_dir=self.args.normals_image_directory,
            to_meters=self.args.to_meters,
            depths_dir=self.args.depths_image_directory,
            output_image_height=None,
            output_image_width=None,
        )

        self.dataset_train = DatasetCls(
            data_list_fname=self.args.train_data_list_fname,
            **common_kwargs
        )

        self.dataset_val = DatasetCls(
            data_list_fname=self.args.val_data_list_fname,
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

        views = np.arange(number_views)

        # positions in the subsampled 'views' array
        self.rotated_views = [i for i, v in enumerate(views) if v in self.rotated_views_global]
        self.non_rotated_views = [i for i, v in enumerate(views) if v not in self.rotated_views_global]

        if len(views) > 8:
            self.visualization_view_ids = np.random.choice(views, size=8, replace=False)
        else:
            self.visualization_view_ids = views

        # self.visualization_view_ids = views
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

            rotated_views_for_pointmaps = [6, 7]

            self.pointmaps_gt = depth_to_pointmap_robust(
                depth = self.depth_maps_gt.squeeze(-1),              # (B,V,H,W)
                K     = self.inputs['camera_intrinsics'],
                extr  = self.inputs['camera_extrinsics'], rotated_views=rotated_views_for_pointmaps).permute(0,1,4,2,3)  # (B,V,H,W,3)

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
                'camera_extrinsics': data['color_camera_extrinsics'],
                'camera_distortions': data['color_camera_distortions'],
            }



    def forward(self):

        # self.global_points = self.target_vertices
        # return

        random_grid = True if self.model.training else False

        if self.args.enable_local:
            # coarse_results_cache_dir = os.path.join(self.args.pretrained_path)
            # coarse_results_cache_dir = os.path.join(os.path.dirname(os.path.dirname(pth_path)), "global_regs")

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

            self.pliks_out['V_fit'] = self.pliks_out['V_fit']
            if not self.args.to_meters:
                self.pliks_out['V_fit'] = self.pliks_out['V_fit'] * self.unit_factor  # convert back to mm if needed


    def compute_vertex_losses(self, vertices, target_vertices=None, suffix=""):
        losses = {}

        registrations_are_here = False
        if target_vertices is None:
            registrations_are_here = True
            # print('Using self.target_vertices for vertex losses computation')
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
        
        # assert that all our losses are 0 if registrations_are_here is True
        if registrations_are_here:
            for key in losses:
                assert (losses[key] == 0.0), f"Loss {key} is not zero despite using registrations: {losses[key].item()}"

        if self.args.weight_points2surface > 0.0:
            points2surface_loss = self.args.weight_points2surface * self._points2surface_loss_for_vertices(vertices)
            losses['points2surface_loss' + suffix] = points2surface_loss


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

        rotated_views_for_pointmaps = [6, 7]

        depth_maps_pred = pred['depth_images'].view(bs, num_views, height, width)
        pointmaps_pred = (
            depth_to_pointmap_robust(
                depth=depth_maps_pred,
                K=self.inputs['camera_intrinsics'],
                extr=self.inputs['camera_extrinsics'],
                rotated_views=rotated_views_for_pointmaps
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
            smirk_losses, smirk_predictions = self.compute_smirk_loss(normal_maps_pred_11, visibility_mask, suffix)
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
        inverse_visibility_mask = 1.0 - visibility_mask_flat.float()
        
        gt_images = self.inputs['images'].view(bs * num_views, c, height, width) 
        # denormalize 
        gt_images = gt_images * self.dataset_train.std.to(self.device).view(1, c, 1, 1) + self.dataset_train.mean.to(self.device).view(1, c, 1, 1)

        masked_images, _ = random_pixel_mask(gt_images, visibility_mask=inverse_visibility_mask)
        
        pred_maps = torch.cat([masked_images, normal_maps_pred.view(bs * num_views, c, height, width)], dim=1)

        if self.args.image_resize_factor == 2:
            pred_maps = torch.nn.functional.pad(pred_maps, (8,0,10,0), mode='constant', value=0)

        pred_fused_images = self.fuse_generator(pred_maps)

        if self.args.image_resize_factor == 2:
            pred_fused_images = pred_fused_images[:, :, 10:, 8:]

        pred_fused_images = pred_fused_images.view(bs, num_views, c, height, width)

        gt_maps = torch.cat([masked_images, self.normal_maps_gt.view(bs * num_views, c, height, width)], dim=1)

        if self.args.image_resize_factor == 2:
            gt_maps = torch.nn.functional.pad(gt_maps, (8,0,10,0), mode='constant', value=0)

        gt_fused_images = self.fuse_generator(gt_maps)

        if self.args.image_resize_factor == 2:
            gt_fused_images = gt_fused_images[:, :, 10:, 8:]

        gt_fused_images = gt_fused_images.view(bs, num_views, c, height, width)

        gt_images = gt_images.view(bs, num_views, c, height, width)

        smirk_loss_l1 = torch.nn.functional.l1_loss(pred_fused_images[:,self.non_rotated_views], gt_images[:,self.non_rotated_views]) * 100
        smirk_loss_vgg = self.vgg_loss(pred_fused_images[:,self.non_rotated_views], gt_images[:,self.non_rotated_views], needs_normalization=True)
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


        


            diff_sq = (pred_pts - gt_pts).pow(2).sum(-1)  # (B,V,N)



        # if "mediapipe" in suffix_lower:
        #     print(diff_sq)
        #     print(diff)
        """
        # diff_sq: (B, V, N)
        diff = diff_sq.sqrt()  # Euclidean distance in pixels or normalized coords

        # threshold τ (in same units as diff)
        tau = 1.8 # getattr(self.args, 'landmarks_hinge_thresh', 0.0)
        # print(diff.shape, 'diff', diff_sq.shape, 'dsq')
        # print(diff)
        # hinge: 0 if diff <= τ, (diff - τ) otherwise
        hinge = torch.clamp(diff - tau, min=0.0)
        # print(hinge.shape)

        # choose your “shape” of the penalty:
        #  - L1-like:      loss_point = hinge
        #  - L2-like:      loss_point = hinge ** 2
        #  - smooth-ish:   loss_point = hinge ** 2 / 2, etc.
        loss_point = hinge ** 2     # (B, V, N)

        loss_point = loss_point[loss_point > 0] 

        # if no elements, then 0
        if loss_point.numel() == 0:
            loss_point = torch.tensor(0.0, device=vertices.device)

        per_point_loss = loss_point
        per_view_loss  = per_point_loss.mean(-1)  # (B, V)
        # print(per_view_loss.shape, 'pp')
        """
        # DENSE_LANDMARKS_WEIGHTS = {
        #     'face': 1.0,
        #     'ears': 1.0,
        #     'eyeballs': 1.0,
        #     'eye_region': 1.0,
        #     'lips': 1.0,
        #     'neck': 1.0,
        #     'nostrils': 1.0,
        #     'scalp': 1.0, # changed from 1.0 to 10.0
        #     'boundary': 1.0,
        #     'left_eyeball': 1.0,
        #     'right_eyeball': 1.0,
        #     'right_ear': 1.0,
        #     'right_eye_region': 1.0,
        #     'left_ear': 1.0,
        #     'left_eye_region': 1.0,
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


        # if not getattr(self.args, 'dense_landmarks_weights_mask'

        # print(self.args.dense_mask_weights)
        # print(diff_sq.shape, self.dense_landmarks_weights_mask.shape)

        # DENSE_LANDMARKS_WEIGHTS = {
        #     'face': 1.0,
        #     'ears': 1.0,
        #     'eyeballs': 1.0,
        #     'eye_region': 1.0,
        #     'lips': 5.0,
        #     'neck': 1.0,
        #     'nostrils': 1.0,
        #     'scalp': 1.0, # changed from 1.0 to 10.0
        #     'boundary': 1.0,
        #     'left_eyeball': 1.0,
        #     'right_eyeball': 1.0,
        #     'right_ear': 1.0,
        #     'right_eye_region': 1.0,
        #     'left_ear': 1.0,
        #     'left_eye_region': 1.0,
        # }

        # self.dense_landmarks_weights_mask = torch.zeros((5023,), dtype=torch.float32).to(self.device)
        # for key in self.vertex_masks_tempeh.keys():
        #     if 'vertex_count' in key:
        #         continue
        #     for idx in self.vertex_masks_tempeh[key].tolist():
        #         self.dense_landmarks_weights_mask[idx] = DENSE_LANDMARKS_WEIGHTS.get(key, 0.0)
        # self.dense_landmarks_weights_mask = self.dense_landmarks_weights_mask.unsqueeze(0).unsqueeze(0)  # (1,1,V)
        # print(sum(self.dense_landmarks_weights_mask > 0.0), 'vertices with dense landmarks supervision')


        # if "dense" in suffix_lower and getattr(self, 'dense_landmarks_weights_mask', None) is not None:
        #     print('Applying dense landmarks weights mask')
        #     weights = self.dense_landmarks_weights_mask.to(diff_sq.device)
        #     diff_sq = diff_sq * weights

        # for key in self.vertex_masks_tempeh.keys():
            # print(diff_sq.shape)
            # raise
            # print(diff_sq[:,0,self.vertex_masks_tempeh[key]].mean())


        # # # -------- NEW: hard upweighting for big errors --------
        # thr_sq   = getattr(self.args, 'dense_landmarks_err_thresh_sq 1.0)     # e.g., 3.0**2
        # hi_w     = getattr(self.args, 'dense_landmarks_high_weight', 100.0)     # e.g., 100.0

        # # # elementwise multiplier: 100 if diff_sq > thr_sq, else 1
        # mult = torch.where(
        #     diff_sq > thr_sq,
        #     torch.as_tensor(hi_w, device=diff_sq.device, dtype=diff_sq.dtype),
        #     torch.as_tensor(1.0, device=diff_sq.device, dtype=diff_sq.dtype)
        # )

        # diff_sq = diff_sq * mult

        # print(diff_sq.shape)
        # # get the mean of nonzero
        # mean_nonzero = diff_sq[diff_sq > 0].mean()
        # # print('Landmarks loss - mean nonzero:', mean_nonzero.item())

        per_point_loss = diff_sq

        per_view_loss = per_point_loss.mean(-1)  # (B,V)

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
        else:
            global_vertex_losses = self.compute_vertex_losses(self.global_points, suffix="")
        
        all_losses.update(global_vertex_losses)

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
            

        landmarks_loss_dense_out, global_landmarks_projected_dense_out = self.compute_landmarks_loss(
            self.global_points,
            suffix="_dense",
            gt_landmarks=self.landmarks_dense_uv if self.args.to_meters else self.landmarks_dense,
            gt_mask=self.landmarks_dense_mask
        )
        all_losses['landmarks_loss_dense_out'] = landmarks_loss_dense_out
        self.global_landmarks_projected_dense_out = global_landmarks_projected_dense_out

        landmarks_loss_dense_mediapipe_out, global_landmarks_projected_dense_mediapipe_out = self.compute_landmarks_loss(
            self.global_points,
            suffix="_mediapipe",
            gt_landmarks=self.landmarks_dense_mediapipe_uv if self.args.to_meters else self.landmarks_dense_mediapipe,
            # gt_mask=self.landmarks_dense_mediapipe_mask
        )
        all_losses['landmarks_loss_dense_mediapipe_out'] = landmarks_loss_dense_mediapipe_out
        self.global_landmarks_projected_dense_mediapipe_out = global_landmarks_projected_dense_mediapipe_out

        
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

            landmarks_loss_dense_mediapipe, global_landmarks_projected_dense_mediapipe = self.compute_landmarks_loss(
                self.global_points,
                suffix="_mediapipe",
                gt_landmarks=self.landmarks_dense_mediapipe_uv if self.args.to_meters else self.landmarks_dense_mediapipe,
                # gt_mask=self.landmarks_dense_mediapipe_mask
            )
            all_losses['landmarks_loss_dense_mediapipe'] = landmarks_loss_dense_mediapipe
            self.global_landmarks_projected_dense_mediapipe = global_landmarks_projected_dense_mediapipe

            

        # ---------- END NEW ----------

        # Calculate total loss
        # total_loss = sum([loss for loss in all_losses.values() if isinstance(loss, torch.Tensor)])
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
        self.depth_maps_loss = all_losses.get('depth_maps_loss', 0.0)
        self.point_maps_loss = all_losses.get('point_maps_loss', 0.0)
        self.smirk_loss = all_losses.get('smirk_loss', 0.0)
        self.smirk_loss_l1 = all_losses.get('smirk_loss_l1', 0.0)
        self.smirk_loss_vgg = all_losses.get('smirk_loss_vgg', 0.0)
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
            'Smirk loss': self.smirk_loss if self.args.enable_smirk_loss else 0.0,
            'Smirk loss L1': self.smirk_loss_l1 if self.args.enable_smirk_loss else 0.0,
            'Smirk loss VGG': self.smirk_loss_vgg if self.args.enable_smirk_loss else 0.0,
            'Identity loss': all_losses.get('identity_loss', 0.0) if getattr(self.args, 'enable_arcface_loss', False) else 0.0,
            'Landmarks loss (dense)': all_losses.get('landmarks_loss_dense', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'Landmarks loss (dense) out': all_losses.get('landmarks_loss_dense_out', 0.0) if getattr(self, 'landmarks_dense', None) is not None and self.args.weight_dense_landmarks > 0.0 else 0.0,
            'β regularizer (PLIKS)': all_losses.get('beta_regularizer_pliks', 0.0) if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'ψ regularizer (PLIKS)': all_losses.get('exp_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Pose regularizer (PLIKS)': all_losses.get('pose_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            't regularizer (PLIKS)': all_losses.get('t_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Vertices edge regularizer (PLIKS)': all_losses.get('vertices_regularizer_pliks_edge', 0.0)  if hasattr(self, 'pliks_out') and self.pliks_out is not None else 0.0,
            'Normals grad loss': self.normals_grad_loss if hasattr(self, 'normals_grad_loss') else 0.0,
            'Landmarks loss (dense mediapipe)': all_losses.get('landmarks_loss_dense_mediapipe', 0.0),
            'Landmarks loss (dense mediapipe) out': all_losses.get('landmarks_loss_dense_mediapipe_out', 0.0) # if getattr(self, 'landmarks_dense_mediapipe', None) is not None and self.args.weight_dense_landmarks > 0.0

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

        if (self.global_step % self.args.visualize_frequency == 0):# and (self.global_step > 0):
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
        if mode == 'train':
            key = "_augmented"
        else:
            key = "_augmented" # timo had validation with augmented images as well
        
        with torch.no_grad():
            for idx in range(len(self.inputs['images']))[:1]:
                # if we have base model
                if self.args.enable_local:
                    self.global_points_base = self.coarse_results['vertices']
                    reconstructed_vertices_base = to_numpy(self.global_points_base[idx])


                    if hasattr(self, 'base_local_model'):
                        self.base_local_model.eval()
                        self.base_local_results = self.base_local_model(self.inputs['images'], self.inputs['camera_intrinsics'], self.inputs['camera_extrinsics'], 
                                                            camera_distortions=self.inputs['camera_distortions'], camera_centers=self.camera_centers, global_points=self.global_points_base, random_grid=False)
                        self.global_points_base_local = self.base_local_results[-1]
                        reconstructed_vertices_base_local = to_numpy(self.global_points_base_local[idx])

                else:
                    if hasattr(self, 'base_model'):
                        self.base_model.eval()
                        self.base_results = self.base_model(self.inputs['images'], self.inputs['camera_intrinsics'], self.inputs['camera_extrinsics'], 
                                                            camera_distortions=self.inputs['camera_distortions'])
                        self.global_points_base = self.base_results['vertices']
                        reconstructed_vertices_base = to_numpy(self.global_points_base[idx])

                target_vertices = to_numpy(self.target_vertices[idx])
                reconstructed_vertices = to_numpy(self.global_points[idx])
                faces_pred = to_numpy(self.faces)
                faces_target = self.data.get('f_registration', None)
                if faces_target is not None:
                    if torch.is_tensor(faces_target):
                        faces_target = to_numpy(faces_target[idx] if faces_target.ndim == 3 else faces_target)
                    elif isinstance(faces_target, list):
                        faces_target = to_numpy(faces_target[idx])
                if faces_target is None:
                    faces_target = faces_pred

                # vertex_distance = np.linalg.norm(target_vertices-reconstructed_vertices, axis=-1)


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

                if hasattr(self, 'global_points_base_local'):
                    # Base local model scan colors
                    vertex_colors_scan_base_local = None
                    predicted_vertices_base_local = self.global_points_base_local[idx].unsqueeze(0).float()
                    vertex_colors_scan_base_local = compute_s2m_distance(scan_vertices, predicted_vertices_base_local, predicted_faces, masks=self.flame_masks_triangles)['full']
                    vertex_colors_scan_base_local = dist_to_rgb(vertex_colors_scan_base_local.detach().cpu().numpy(), min_dist=0.0, max_dist=3.0)

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

                        pred_fused_image = (255*pred_fused_image).astype(np.uint8)
                        gt_fused_image = (255*gt_fused_image).astype(np.uint8)
                        masked_image = (255*masked_image).astype(np.uint8)

                        # transpose them !
                        pred_fused_image = pred_fused_image.transpose(1, 2, 0)
                        gt_fused_image = gt_fused_image.transpose(1, 2, 0)
                        masked_image = masked_image.transpose(1, 2, 0)

                        # FLAME SMIRK predictions

                    input_image = (255*input_image).astype(np.uint8)

                    camera_args = {
                        'camera_intrinsics': camera_intrinsics,
                        'camera_extrinsics': camera_extrinsics,
                        'radial_distortion': radial_distortion,
                        'frustum': {'near': 0.01, 'far': 3000.0},
                        'image_size': input_image.shape[:2]
                    }

                    target_rendering = render_mesh(vertices=target_vertices, faces=faces_target, vertex_colors=None, **camera_args)
                    
                    # Only render main model if available
                    reconstruction_rendering = None
                    if reconstructed_vertices is not None:
                        reconstruction_rendering = render_mesh(vertices=reconstructed_vertices, faces=faces_pred, vertex_colors=None, needs_projection=True, **camera_args)


                    # Only render error if we have main model predictions
                    target_error_rendering = None
                    # print(scan_vertices.shape, vertex_colors_scan.shape,self.data['f_scan'][idx].shape)
                    if vertex_colors_scan is not None:
                        target_error_rendering = render_mesh(vertices=scan_vertices, faces=self.data['f_scan'][idx], vertex_colors=vertex_colors_scan, **camera_args)
                    
                    base_error_rendering = None
                    if hasattr(self, 'base_model') and vertex_colors_scan_base is not None:
                        base_error_rendering = render_mesh(vertices=scan_vertices, faces=self.data['f_scan'][idx], vertex_colors=vertex_colors_scan_base, **camera_args)
                    
                    base_reconstruction_rendering = None
                    if hasattr(self, 'base_model'):
                        base_reconstruction_rendering = render_mesh(vertices=reconstructed_vertices_base, faces=faces_pred, vertex_colors=None, needs_projection=True, **camera_args)

                    base_local_reconstruction_rendering = None
                    base_local_error_rendering = None
                    if hasattr(self, 'base_local_model'):
                        base_local_reconstruction_rendering = render_mesh(vertices=reconstructed_vertices_base_local, faces=faces_pred, vertex_colors=None, needs_projection=True, **camera_args)
                        if vertex_colors_scan_base_local is not None:
                            base_local_error_rendering = render_mesh(vertices=scan_vertices, faces=self.data['f_scan'][idx], vertex_colors=vertex_colors_scan_base_local, **camera_args)

                    pliks_reconstruction_rendering = None
                    if reconstructed_vertices_pliks is not None:
                        pliks_reconstruction_rendering = render_mesh(vertices=reconstructed_vertices_pliks, faces=faces_pred, vertex_colors=None, needs_projection=True, **camera_args)

                    pliks_flame_reconstruction_rendering = None
                    if reconstructed_vertices_pliks_flame is not None:
                        pliks_flame_reconstruction_rendering = render_mesh(vertices=reconstructed_vertices_pliks_flame, faces=faces_pred, vertex_colors=None, needs_projection=True, **camera_args)

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
                        landmarks_dense_fan = to_numpy(self.landmarks_dense_fan[idx][relative_view_id])
                        for i in range(landmarks_dense_fan.shape[0]):
                            vis_image = cv2.circle(vis_image, (int(landmarks_dense_fan[i][0]), int(landmarks_dense_fan[i][1])), 2, (0, 0, 255), -1)

                        landmarks_dense_mediapipe = to_numpy(self.landmarks_dense_mediapipe[idx][relative_view_id])
                        for i in range(landmarks_dense_mediapipe.shape[0]):
                            vis_image = cv2.circle(vis_image, (int(landmarks_dense_mediapipe[i][0]), int(landmarks_dense_mediapipe[i][1])), 2, (255, 0, 255), -1)

                        projected_landmarks_mediapipe = None
                        if hasattr(self, 'global_landmarks_projected_dense_mediapipe_out') and self.global_landmarks_projected_dense_mediapipe_out is not None:
                            projected_landmarks_mediapipe = to_numpy(self.global_landmarks_projected_dense_mediapipe_out[idx][relative_view_id])
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
                        # dense_pred_out_image = None

                        if getattr(self, 'landmarks_dense', None) is not None:
                            dense_gt = to_numpy(self.landmarks_dense[idx][relative_view_id])
                            view_mask_val = None
                            if getattr(self, 'landmarks_dense_mask', None) is not None:
                                view_mask_val = float(to_numpy(self.landmarks_dense_mask[idx][relative_view_id]).item())
                            
                            # print(view_mask_val)
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


                    # Pair up images and labels
                    pairs = [
                        (vis_image, "Input"),
                        *([(dense_gt_image, "Dense GT")] if dense_gt_image is not None else []),
                        *([(dense_pred_image, "Dense Pred")] if dense_pred_image is not None else []),
                        *([(dense_pred_out_image, "Dense Pred Out")] if dense_pred_out_image is not None else []),
                        (target_scan, "Target Scan"),
                        (target_rendering, "Target"),
                        *([(base_reconstruction_rendering, "Base Recon")] if base_reconstruction_rendering is not None else []),
                        *([(base_local_reconstruction_rendering, "Base Local Recon")] if base_local_reconstruction_rendering is not None else []),
                        *([(reconstruction_rendering, "Recon")] if reconstruction_rendering is not None else []),
                        *([(flame_reconstruction_rendering, "FLAME Recon")] if flame_reconstruction_rendering is not None else []),
                        *([(target_error_rendering, "Error")] if target_error_rendering is not None else []),
                        *([(base_error_rendering, "Base Error")] if base_error_rendering is not None else []),
                        *([(base_local_error_rendering, "Base Local Error")] if base_local_error_rendering is not None else []),
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
                    if mode == 'val':
                        cv2.imwrite(f'{self.directory_output}/{mode}_images/view_id_{view_id:02d}_{self.global_step}_{idx}_{val_idx}.jpg', cv2.cvtColor(visualization.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    elif mode == 'warm':
                        os.makedirs(os.path.join(self.directory_output, mode + '_images'), exist_ok=True)
                        cv2.imwrite(f'{self.directory_output}/{mode}_images/view_id_{view_id:02d}_{self.global_step}_{idx}_{val_idx}.jpg', cv2.cvtColor(visualization.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(f'{self.directory_output}/{mode}_images/view_id_{view_id:02d}_{self.global_step}_{idx}.jpg', cv2.cvtColor(visualization.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if self.args.wandb and idx == 0 and (self.global_step % 500 == 0):
                        wandb.log({f'{mode.capitalize()}/view_id_{view_id:02d}': wandb.Image(visualization.transpose(1, 2, 0))}, step=self.global_step)

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
        scene.export(out_fname.replace('.ply','.glb'))


# -----------------------------------------------------------------------------

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
