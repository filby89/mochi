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
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.input_ch = 3

        self.descriptor_dim = args.descriptor_dim
        self.number_points = min(args.number_sample_points)

        self.global_voxel_dim = args.global_voxel_dim
        self.global_voxel_inc = args.global_voxel_inc
        self.global_origin = args.global_origin
        self.norm = args.norm

        self.module_names = ['feature_net', 'sparse_point_net']

        from models.model_aligner import FeatureNet2D
        self.feature_net = FeatureNet2D(
            input_ch=self.input_ch, output_ch=self.descriptor_dim)

        from models.model_aligner import GlobalSparsePointNetTransformer
        self.sparse_point_net = GlobalSparsePointNetTransformer(
            input_ch=self.descriptor_dim,
            number_points=self.number_points,
            global_architecture='v2v2',
            global_voxel_dim=self.global_voxel_dim,
            global_voxel_inc=self.global_voxel_inc,
            global_origin=self.global_origin,
            norm=self.norm,
            global_feature_fusion='mean_var',
            transformer_type='rigid',
            num_transformer_channels=args.global_spatial_transformer_dim,
            num_transformer_levels=args.global_transformer_levels,
            transformer_scale_factor=args.global_transformer_scale_factor,
            transformer_use_pooling=args.global_transformer_use_pooling,
        )

    def print_setting(self):
        pass

    def forward(self, images, camera_intrinsics, camera_extrinsics, camera_distortions=None,
                random_grid=True):

        batch_size, num_images, num_channels, image_height, image_width = images.shape
        assert num_channels == self.input_ch, f"unmatched input image channel: {num_channels} (expected: {self.input_ch})"

        images = images.view(-1, num_channels, image_height, image_width)
        feature_maps = self.feature_net(images)
        feature_maps = feature_maps.view(batch_size, num_images, -1, image_height, image_width)

        sparse_point_net_output = self.sparse_point_net(
            feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions,
            random_grid=random_grid, return_features=False)

        global_points = sparse_point_net_output['pts_global']

        output = {
            'vertices': global_points,
            'vertices_uncertainty': torch.zeros_like(global_points[..., :1]),
        }

        return output
