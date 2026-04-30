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

import torch.nn as nn
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, input_ch, output_ch, **kwargs):
        super(Model, self).__init__()

        self.input_ch = input_ch
        self.output_ch = output_ch

        self.module_names = ['model']

        import modules.resnet_dilated as resnet_dilated
        self.model = resnet_dilated.Resnet34_8s_skip2(
            input_ch=input_ch, num_classes=self.output_ch, pretrained=True)

    def print_setting(self):
        print("-"*40)
        print(f"name: feature_net_2d")
        print(f"\t- input_ch: {self.input_ch}")
        print(f"\t- output_ch: {self.output_ch}")

    def forward(self, x):
        '''compute 2d feature maps given images
        Args:
            x: tensor in (B, C, H', W')
        Returns:
            x: tensor in (B, F, H, W)
        '''
        bs, ic, ih, iw = x.shape
        assert ic == self.input_ch, f"unmatched input image channel {ic}, expected {self.input_ch}"
        return self.model(x)
