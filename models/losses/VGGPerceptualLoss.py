import os

import torch
import torch.nn.functional as F
import torchvision


# class VGGPerceptualLoss(torch.nn.Module):
#     def __init__(self):
#         super(VGGPerceptualLoss, self).__init__()
#         blocks = [torchvision.models.vgg16(weights='DEFAULT').features[:4].eval(),
#                   torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval(),
#                   torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval(),
#                   torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval()]
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
#         self.blocks = torch.nn.ModuleList(blocks)

#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     def forward(self, x, y, needs_normalization=False):
#         # x = x * 0.5 + 0.5
#         # y = y * 0.5 + 0.5
#         # print(x.mean(), x.min(), x.max())
#         if len(x.shape) == 5:
#             x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
#             y = y.reshape(-1, y.shape[2], y.shape[3], y.shape[4])
#         if needs_normalization:
#             x = (x - self.mean) / self.std
#             y = (y - self.mean) / self.std

#         # x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
#         # y = F.interpolate(y, mode='bilinear', size=(224, 224), align_corners=False)
#         perceptual_loss = 0.0
#         style_loss = 0.0

#         for i, block in enumerate(self.blocks):
#             x = block(x)
#             y = block(y)
#             print(x.shape,y.shape)
#             perceptual_loss += torch.nn.functional.l1_loss(x, y)

#             # b, ch, h, w = x.shape
#             # act_x = x.reshape(x.shape[0], x.shape[1], -1)
#             # act_y = y.reshape(y.shape[0], y.shape[1], -1)
#             # gram_x = act_x @ act_x.permute(0, 2, 1) / (ch * h * w)
#             # gram_y = act_y @ act_y.permute(0, 2, 1) / (ch * h * w)
#             # style_loss += torch.nn.functional.l1_loss(gram_x, gram_y)

#         return perceptual_loss#, style_loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        blocks = [torchvision.models.vgg16(weights='DEFAULT').features[:4].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y, needs_normalization=False, mask: torch.Tensor = None, return_map: bool = False):
        """
        x, y:  (B, 3, H, W) or (B*V, 3, H, W)
        mask:  (B, 1, H, W) or (B*V, 1, H, W), 1=keep, 0=ignore
        return_map: if True, returns (scalar_loss, map[ B, H, W ])
        """
        if len(x.shape) == 5:
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            y = y.reshape(-1, y.shape[2], y.shape[3], y.shape[4])
            if mask is not None and len(mask.shape) == 5:
                mask = mask.reshape(-1, mask.shape[2], mask.shape[3], mask.shape[4])

        if needs_normalization:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        B, _, H0, W0 = x.shape
        perceptual_loss = 0.0
        maps = []  # collect layer maps, upsampled to (H0,W0)

        m = mask
        for block in self.blocks:
            x = block(x)
            y = block(y)

            # per-layer absolute diff
            diff = torch.abs(x - y)  # (B, C, h, w)

            if m is not None:
                m_ds = torch.nn.functional.interpolate(m, size=diff.shape[-2:], mode='bilinear', align_corners=False)  # (B,1,h,w)
                diff = diff * m_ds  # broadcast on channel

            # scalar add: L1 over channels and pixels, normalized
            if m is not None:
                num = diff.sum()
                den = (m_ds.sum() * diff.shape[1]).clamp_min(1e-8)
                perceptual_loss = perceptual_loss + (num / den)
            else:
                perceptual_loss = perceptual_loss + torch.nn.functional.l1_loss(x, y)

            if return_map:
                # make a per-pixel map at this layer, average over channels
                layer_map = diff.mean(dim=1, keepdim=True)                    # (B,1,h,w)
                layer_map_up = torch.nn.functional.interpolate(layer_map, size=(H0, W0), mode='bilinear', align_corners=False)  # (B,1,H0,W0)
                maps.append(layer_map_up)

        if return_map:
            vgg_map = torch.stack(maps, dim=0).mean(dim=0).squeeze(1)  # (B, H0, W0)
            return perceptual_loss, vgg_map

        return perceptual_loss
