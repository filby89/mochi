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

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """ A standard Transformer-style attention block """
    def __init__(self, embed_dim, num_heads, ff_dim=None):
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if ff_dim is None:
            ff_dim = embed_dim * 4 # A common practice

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        print(f"AttentionBlock forward: query shape {query.shape}, key_value shape {key_value.shape}")
        """
        Args:
        - query: (N, 1, F) -> The context for attention
        - key_value: (N, V, F) -> The features from each view to be attended over
        
        Returns:
        - out: (N, 1, F) -> The fused feature
        """
        # Multi-Head Attention
        attn_output, _ = self.mha(query, key_value, key_value) # query, key, value
        
        # Add & Norm
        x = self.norm1(query + attn_output)
        
        # Feed-Forward Network
        ffn_output = self.ffn(x)
        
        # Add & Norm
        out = self.norm2(x + ffn_output)
        
        return out
# -----------------------------------------------------------------------------

class VolumetricFeatureSampler(nn.Module):

    def __init__(self, padding_mode='zeros', feature_fusion='', feature_dim=0):
        super(VolumetricFeatureSampler, self).__init__()

        """ define a volumetric feature sampler
        """
        self.padding_mode = padding_mode

        # note: the two operations will cumulate the feature and squared feature, independently among the feature channels
        # later in self.compute_stats(), we convert the results into means and standard deviations.
        # currently these are hardcoded operations, and all tensors in this class with some dimension "2F" are also due to this reason
        self.fuse_ops = [
            lambda feat, base: feat if base is None else feat + base,
            lambda feat, base: feat.pow(2.0) if base is None else feat.pow(2.0) + base,
        ]
        self.fuse_ops_num = len(self.fuse_ops)

        if feature_fusion in [None, '', 'baseline_mean_var']:
            self.compute_feature_volume = self.baseline_mean_var        
        elif feature_fusion == 'mean_var':
            self.compute_feature_volume = self.mean_var
        elif feature_fusion == 'attention':
            feature_dim = 8
            num_attention_blocks = 4
            num_attention_heads = 8
            self.compute_feature_volume = self.attention_fusion
            self.attention_blocks = nn.ModuleList([
                AttentionBlock(embed_dim=feature_dim, num_heads=num_attention_heads) 
                for _ in range(num_attention_blocks)
            ])
            print(f"Initialized Volumetric Sampler with {num_attention_blocks} attention blocks.")
        else:
            raise RuntimeError( "Unrecognizable feature_fusion: %s" % ( feature_fusion ) )
                
    @staticmethod
    def project(points, camera_intrinsics, camera_extrinsics, camera_distortions, height, width, eps=1e-7):
        """ project the grid points into images (full perspective projection)
        Args:
        - points: (B, N, 3)
        - camera_intrinsics: (B, 3, 4)
        - camera_extrinsics: (B, 3, 3)
        - camera_distortions: (B, 2)
        - height: int 
        - width: int

        Return:
        - u_coord: (B, N), value range in [-1, 1]
        - v_coord: (B, N), value range in [-1, 1]
        """

        device = points.device
        batch_size, num_points, _ = points.shape
        points = points.transpose(1, 2).contiguous()

        ones = torch.ones(batch_size, 1, num_points).to(device)
        points_homogeneous = torch.cat((points, ones), axis=-2) # (batch_size, 4, num_points)

        # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
        points_image = camera_extrinsics.bmm(points_homogeneous) # (batch_size, 3, num_points)

        # Transformation from 3D camera coordinate system to the undistorted image plane 
        mask = (points_image.abs() < eps)
        mask[:,:2,:] = False
        points_image[mask] = 1.0 # Avoid division by zero

        points_image_x = points_image[:,0,:] / points_image[:,2,:]
        points_image_y = points_image[:,1,:] / points_image[:,2,:]

        # Transformation from undistorted image plane to distorted image coordinates
        K1, K2 = camera_distortions[:,0], camera_distortions[:,1]       # (batch_size)
        r2 = points_image_x**2 + points_image_y**2            # (batch_size, num_points)
        r4 = r2**2
        radial_distortion_factor = (1 + K1[:, None]*r2 + K2[:, None]*r4)  # (batch_size, num_points)

        points_image_x = points_image_x*radial_distortion_factor
        points_image_y = points_image_y*radial_distortion_factor
        points_image_z = torch.ones_like(points_image[:,2,:])
        points_image = torch.cat((points_image_x[:, None, :], points_image_y[:, None, :], points_image_z[:, None, :]), dim=1)

        # Transformation from distorted image coordinates to the final image coordinates with the camera intrinsics
        points_image = camera_intrinsics.bmm(points_image)      # (batch_size, 3, num_points)

        # Convert the range to [-1, 1]
        u_coord = 2. * points_image[:, 0, :] / (width - 1.) - 1.
        v_coord = 2. * points_image[:, 1, :] / (height - 1.) - 1.
        return u_coord, v_coord

    @staticmethod
    def compute_stats(feat_grid, view_num, split_dim=2):
        """ convert the feature grid from 1st- and 2nd-order moments to actual average and std
        Args:
        - feat_grid: (B, G, 2F, GH, GW, GD)
        - view_num: int
        - split_dim: int, which dim to split, currently should always set to 2 (corresp. to the dim of "2F")

        Return:
        - feat_grid: (B, G, 2F, GH, GW, GD)
        """
        feat_grid /= float(view_num)
        feat_mean, feat_var = feat_grid.chunk(2, dim=split_dim)
        feat_var = feat_var - feat_mean.pow(2)
        feat_grid = torch.cat((feat_mean, feat_var), dim=split_dim)
        return feat_grid


    def attention_fusion(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        """ Fuses multi-view features using a series of attention blocks.
        """
        # Step 1: Projection and Sampling (Identical to the mean_var method)
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous()
        xyzs = xyzs.view(batch_size, 3, -1)
        xyzs = xyzs.transpose(1, 2).contiguous()

        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  

        grid2d_uv = torch.stack((u_coord, v_coord), dim=2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)
        feat2d_uv = feat2d_uv.transpose(1, 2)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, fd, grid_h, grid_w, grid_d)

        # Step 2: Reshape for Attention
        # The goal is to get a tensor of shape (N, V, F) where N is the total number of points
        # and V is the number of views.
        # Current shape: (B, V, G, F, GH, GW, GD)
        feat_permuted = feat2d_uv.permute(0, 2, 4, 5, 6, 1, 3).contiguous() # -> (B, G, GH, GW, GD, V, F)
        
        num_points = batch_size * grid_num * grid_h * grid_w * grid_d
        features_per_point = feat_permuted.view(num_points, num_views, fd) # -> (N, V, F)

        # Step 3: Attention-based Fusion
        # We use the mean feature as the initial 'query' to provide context for the point.
        query = features_per_point.mean(dim=1, keepdim=True) # -> (N, 1, F)
        key_value = features_per_point                          # -> (N, V, F)

        # Pass through the stack of attention blocks. The output of one block becomes the query for the next.
        fused_feature = query
        for block in self.attention_blocks:
            fused_feature = block(fused_feature, key_value)

        # The output is of shape (N, 1, F). We remove the middle dimension.
        fused_feature = fused_feature.squeeze(1) # -> (N, F)
        
        # Step 4: Reshape back to volumetric grid
        # Reshape back to (B, G, GH, GW, GD, F)
        feature_volume_reshaped = fused_feature.view(batch_size, grid_num, grid_h, grid_w, grid_d, fd)
        
        # Permute to get the final desired output shape (B, G, F, GH, GW, GD)
        feature_volume = feature_volume_reshaped.permute(0, 1, 5, 2, 3, 4).contiguous()

        # Note: The output feature dimension is F, not 2F, because we are producing a single
        # fused feature vector, not separate mean and variance.
        return feature_volume

    def mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  

        # sample
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2)    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)
        # print(feat2d_uv.shape)
        # voxels = torch.cat((feat2d_uv.mean(dim=1), feat2d_uv.std(dim=1, unbiased=False)), dim=2).contiguous()   # (B, G, 2F, GH, GW, GD)
        feature_volume = torch.cat((feat2d_uv.mean(dim=1), feat2d_uv.var(dim=1, unbiased=True)), dim=2).contiguous()   # (B, G, 2F, GH, GW, GD)
        return feature_volume


    def forward(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        """ the full forward process
        Args:
        - feature_maps: tensor in (B, V, F, H, W), batched feature map
        - camera_intrinsics: tensor in (B, V, 3, 3), batched intrisic matrices
        - camera_extrinsics: tensor in (B, V, 3, 4), batched rigid transformation (camera poses)
        - camera_distortions: tensor in (B, V, 2), batched radial camera distortions
        - grids: (B, G, 3, GH, GW, GD), batched grid coordinates

        B: batch size; F: feature dim; V: view num;
        G: grid num (global stage = 1, local stage = input vertex number)
        GH, GW, GD: shape of the grid, currently all equal to grid dim D

        Return:
        - voxels: tensor in (B, G, 2F, GH, GW, GD), sampled volumetric features
        """

        feature_volume = self.compute_feature_volume(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids)
        return feature_volume

