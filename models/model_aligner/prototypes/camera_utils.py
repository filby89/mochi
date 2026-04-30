import einops as ein
import numpy as np
import torch
import torch.nn.functional as F

# camera_utils.py
import torch
import einops as ein
from typing import Optional

def patch_center_pixels(H: int, W: int, patch_size: int, device=None):
    """Return (Py, Px, 2) integer pixel centers (x,y) for non-overlapping patches."""
    Py = H // patch_size
    Px = W // patch_size
    # centers: (col + 0.5)*ps, (row + 0.5)*ps
    xs = torch.arange(Px, device=device).float() * patch_size + (patch_size * 0.5)
    ys = torch.arange(Py, device=device).float() * patch_size + (patch_size * 0.5)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing="xy")  # (Px, Py)
    # (Py, Px, 2) with (x,y)
    centers = torch.stack([x_grid.T, y_grid.T], dim=-1)
    return centers  # (Py, Px, 2)

def camera_rays_at_points(
    Ks: torch.Tensor,                      # (B*V, 3, 3) or (B, V, 3, 3)
    poses_c2w: Optional[torch.Tensor],     # (B*V, 4, 4) or (B, V, 4, 4) or None
    points_xy: torch.Tensor,               # (Py, Px, 2) pixel coords (x,y)
    normalize_to_unit_sphere: bool = True,
):
    """
    Rays for given pixel (x,y) points. If poses_c2w is None → camera frame (origin=(0,0,0)).
    Returns:
      ray_o: (N, P, 3)
      ray_d: (N, P, 3)
    where N = B*V and P = Py*Px
    """
    if Ks.dim() == 4:  # (B, V, 3, 3)
        B, V = Ks.shape[:2]
        N = B * V
        Ks = Ks.reshape(N, 3, 3)
        poses = None if poses_c2w is None else poses_c2w.reshape(N, 4, 4)
    else:
        N = Ks.shape[0]
        poses = poses_c2w

    device = Ks.device
    Py, Px = points_xy.shape[:2]
    P = Py * Px

    # unpack intrinsics
    fx = Ks[:, 0, 0].view(N, 1)
    fy = Ks[:, 1, 1].view(N, 1)
    cx = Ks[:, 0, 2].view(N, 1)
    cy = Ks[:, 1, 2].view(N, 1)

    # (P,) pixel coords
    x = points_xy[..., 0].reshape(P).to(device)
    y = points_xy[..., 1].reshape(P).to(device)

    # broadcast to N
    xx = (x.unsqueeze(0) - cx) / fx  # (N, P)
    yy = (y.unsqueeze(0) - cy) / fy  # (N, P)

    dirs_cam = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (N, P, 3)
    if normalize_to_unit_sphere:
        dirs_cam = dirs_cam / (dirs_cam.norm(dim=-1, keepdim=True) + 1e-8)

    if poses is None:
        # camera frame
        ray_o = torch.zeros((N, P, 3), device=device)
        ray_d = dirs_cam
    else:
        # rotate by R, translate by t
        R = poses[:, :3, :3]        # (N, 3, 3)
        t = poses[:, :3, 3]         # (N, 3)
        ray_d = ein.einsum(dirs_cam, R, "n p c, n c k -> n p k")
        # camera center in world = t (for c2w)
        ray_o = t[:, None, :].expand(N, P, 3)

        if normalize_to_unit_sphere:
            ray_d = ray_d / (ray_d.norm(dim=-1, keepdim=True) + 1e-8)

    return ray_o, ray_d  # (N, P, 3), (N, P, 3)

def rays_to_plucker(ray_o: torch.Tensor, ray_d: torch.Tensor):
    """
    Plücker line from ray: (d, m = o × d). Not normalized (you may L2-normalize d, scale m accordingly if desired).
    Inputs: (N, P, 3)
    Returns: plucker (N, P, 6) = [d, m]
    """
    d = ray_d
    m = torch.cross(ray_o, ray_d, dim=-1)
    return torch.cat([d, m], dim=-1)  # (N, P, 6)


def get_rays_in_camera_frame(intrinsics, height, width, normalize_to_unit_sphere=True):
    """
    Convert camera intrinsics to a raymap (ray origins + directions) in camera frame.
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool

    Returns:
        - ray_origins: (HxWx3 or BxHxWx3) tensor
        - ray_directions: (HxWx3 or BxHxWx3) tensor
    """
    # Add batch dimension if not present
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size = intrinsics.shape[0]
    device = intrinsics.device

    # Compute rays in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    ray_origins = torch.zeros((batch_size, height, width, 3), device=device)
    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    ray_directions = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)

    # Normalize ray directions to unit sphere if required (else rays will lie on unit plane)
    if normalize_to_unit_sphere:
        ray_directions = ray_directions / torch.norm(
            ray_directions, dim=-1, keepdim=True
        )

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        ray_origins = ray_origins.squeeze(0)
        ray_directions = ray_directions.squeeze(0)

    return ray_origins, ray_directions



def get_rays_in_world_frame(
    intrinsics, height, width, normalize_to_unit_sphere, camera_pose=None
):
    """
    Convert camera intrinsics & camera_pose (if provided) to a raymap (ray origins + directions) in camera or world frame (if camera_pose is provided).
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool
        - camera_pose: 4x4 or Bx4x4 torch tensor

    Returns:
        - ray_origins: (HxWx3 or BxHxWx3) tensor
        - ray_directions: (HxWx3 or BxHxWx3) tensor
    """
    # Get rays in camera frame
    ray_origins, ray_directions = get_rays_in_camera_frame(
        intrinsics, height, width, normalize_to_unit_sphere
    )

    if camera_pose is not None:
        # Add batch dimension if not present
        if camera_pose.dim() == 2:
            camera_pose = camera_pose.unsqueeze(0)
            ray_origins = ray_origins.unsqueeze(0)
            ray_directions = ray_directions.unsqueeze(0)
            squeeze_batch_dim = True
        else:
            squeeze_batch_dim = False

        # Convert rays from camera frame to world frame
        ray_origins_homo = torch.cat(
            [ray_origins, torch.ones_like(ray_origins[..., :1])], dim=-1
        )
        ray_directions_homo = torch.cat(
            [ray_directions, torch.zeros_like(ray_directions[..., :1])], dim=-1
        )
        ray_origins_world = ein.einsum(
            camera_pose, ray_origins_homo, "b i k, b h w k -> b h w i"
        )
        ray_directions_world = ein.einsum(
            camera_pose, ray_directions_homo, "b i k, b h w k -> b h w i"
        )
        ray_origins_world = ray_origins_world[..., :3]
        ray_directions_world = ray_directions_world[..., :3]

        # Remove batch dimension if it was added
        if squeeze_batch_dim:
            ray_origins_world = ray_origins_world.squeeze(0)
            ray_directions_world = ray_directions_world.squeeze(0)
    else:
        ray_origins_world = ray_origins
        ray_directions_world = ray_directions

    return ray_origins_world, ray_directions_world
