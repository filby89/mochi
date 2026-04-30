# utils/spatial_grad.py
import torch
import torch.nn.functional as F

def spatial_grad(x: torch.Tensor):
    """
    x: (B, H, W) or (B, C, H, W)
    returns gx, gy with same shape as x (zeros padded on last row / col)
    """
    if x.dim() == 4:               # (B,C,H,W) – treat each channel independently
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = F.pad(gx, (0, 0, 0, 1))
        gy = F.pad(gy, (0, 1, 0, 0))
    else:                           # (B,H,W)
        gx = x[:, :, 1:] - x[:, :, :-1]
        gy = x[:, 1:, :] - x[:, :-1, :]
        gx = F.pad(gx, (0, 0, 0, 1))
        gy = F.pad(gy, (0, 1, 0, 0))
    return gx, gy
