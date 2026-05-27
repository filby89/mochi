# utils/synth_views_meshhelper.py
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from utils.mesh_helper import depth_to_pointmap_robust, pointmap_to_rgb, MeshHelper

# ----------------- small helpers -----------------
def _normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def _look_at(eye, target, up):
    z = _normalize(eye - target)              # forward
    x = _normalize(torch.cross(up, z))        # right
    y = torch.cross(z, x)                     # up
    R = torch.stack([x, y, z], dim=0)         # (3,3)
    t = -R @ eye.view(3, 1)                   # (3,1)
    w2c = torch.eye(4, device=eye.device, dtype=eye.dtype)
    w2c[:3, :3] = R
    w2c[:3, 3]  = t.view(3)
    return w2c

def _cam_center_from_w2c(w2c):
    # inverse transform of origin: C = -R^T t
    R = w2c[:3,:3]; t = w2c[:3,3]
    return (-R.T @ t).view(3)

def _cam_up_from_w2c(w2c):
    # +Y axis of camera in world = R^T * [0,1,0]
    R = w2c[:3,:3]
    return _normalize(R.T[:,1])

def _compute_mesh_center(vertices):
    return vertices.mean(dim=0)

def _compute_normals_from_points(points, valid_mask):
    """
    points: (H,W,3) in camera space. valid_mask: (H,W) bool.
    returns (H,W,3) normals in camera space, masked->0 where invalid.
    """
    H, W, _ = points.shape
    # pad for central differences
    P = F.pad(points.permute(2,0,1), (1,1,1,1), mode='replicate') # (3,H+2,W+2)
    Px = (P[:,:,2:] - P[:,:,:-2]) * 0.5   # (3,H+2,W)
    Py = (P[:,2:,:] - P[:,:-2,:]) * 0.5   # (3,H,W+2)
    # crop back to (3,H,W)
    Px = Px[:,1:-1,:]
    Py = Py[:,:,1:-1]
    # normal = cross(dx, dy) (right-hand). use camera coords.
    Nx = Px.permute(1,2,0)  # (H,W,3)
    Ny = Py.permute(1,2,0)
    N  = torch.cross(Nx, Ny, dim=-1)
    N  = F.normalize(N, dim=-1)
    N[~valid_mask] = 0.0
    return N

# ----------------- camera sampling -----------------
def sample_virtual_cams_like_rig(Ks_existing, w2c_existing, n=4, res=(400,300), target=None,
                                 radial_scale=(0.8,1.2), jitter_std=0.05, distortions_existing=None):
    """
    Ks_existing:   (N,3,3) torch
    w2c_existing:  (N,4,4) torch
    distortions_existing: (N,D) torch or None
    returns: list of dicts [{'K':(3,3),'w2c':(4,4),'W':int,'H':int}, ...]
    """
    device = Ks_existing.device; dtype = Ks_existing.dtype
    N = Ks_existing.shape[0]

    centers = torch.stack([_cam_center_from_w2c(w2c_existing[i]) for i in range(N)], 0)
    ups     = torch.stack([_cam_up_from_w2c(w2c_existing[i])     for i in range(N)], 0)

    centroid   = centers.mean(0)
    avg_radius = (centers - centroid).norm(dim=-1).mean().clamp(min=1e-6)
    up_ref     = _normalize(ups.mean(0))

    W, H = int(res[0]), int(res[1])

    if distortions_existing is not None:
        distortions_existing = distortions_existing.to(device=device, dtype=dtype)
        if distortions_existing.dim() == 3 and distortions_existing.shape[0] == 1:
            distortions_existing = distortions_existing[0]
        if distortions_existing.dim() == 1:
            distortions_existing = distortions_existing.unsqueeze(0)

    cams = []
    for _ in range(n):
        a, b = np.random.choice(N, 2, replace=False)
        dirv = _normalize(0.5*(centers[a]-centroid) + 0.5*(centers[b]-centroid))
        scale = torch.empty((), device=device, dtype=dtype).uniform_(*radial_scale)
        eye = centroid + dirv * (avg_radius * scale)
        eye = eye + torch.randn(3, device=device, dtype=dtype) * (jitter_std * avg_radius)
        up  = _normalize(0.7*up_ref + 0.3*_cam_up_from_w2c(w2c_existing[a]))

        w2c = _look_at(eye, target, up)

        # pick a realistic K from existing rig
        kidx = np.random.randint(0, N)
        cam_dict = {'K': Ks_existing[kidx].clone(), 'w2c': w2c, 'W': W, 'H': H}
        if distortions_existing is not None and kidx < distortions_existing.shape[0]:
            cam_dict['distortion'] = distortions_existing[kidx].clone()
        cams.append(cam_dict)
    return cams

# ----------------- render using YOUR MeshHelper -----------------
def render_depth_with_meshhelper(
    mesh_helper: MeshHelper,
    vertices: torch.Tensor,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    distortions: Optional[torch.Tensor] = None,
    depth_size: tuple = (300, 300),
) -> Dict[str, torch.Tensor]:
    """
    Render normals and depth for a single mesh given a set of cameras.

    Args:
        mesh_helper: MeshHelper instance configured for the mesh topology.
        vertices: (N, 3) or (1, N, 3) tensor of vertex positions (world space).
        Ks: (V, 3, 3) intrinsics for the virtual cameras.
        w2cs: (V, 4, 4) world-to-camera matrices.
        distortions: (V, 3) radial distortion coefficients or None.
        depth_size: (H, W) output resolution.

    Returns:
        dict with keys:
            'depth':   (V, H, W) tensor of depth values.
            'normals': (V, H, W, 3) tensor of surface normals.
            'mask':    (V, H, W) boolean validity mask.
    """
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)

    device = vertices.device
    dtype = vertices.dtype
    height, width = depth_size
    num_views = Ks.shape[0]

    Ks_batch = Ks.to(device=device, dtype=dtype).unsqueeze(0)  # (1, V, 3, 3)
    extr_batch = w2cs[:, :3, :].to(device=device, dtype=dtype).unsqueeze(0)  # (1, V, 3, 4)
    distortions = None
    if distortions is None:
        radial = torch.zeros((1, num_views, 2), device=device, dtype=dtype)
    else:
        distortions = distortions.to(device=device, dtype=dtype)
        if distortions.dim() == 1:
            distortions = distortions.unsqueeze(0)
        if distortions.shape[0] != num_views:
            distortions = distortions.view(num_views, -1)
        if distortions.shape[-1] < 3:
            pad = torch.zeros((num_views, 3 - distortions.shape[-1]), device=device, dtype=dtype)
            distortions = torch.cat([distortions, pad], dim=-1)
        elif distortions.shape[-1] > 3:
            distortions = distortions[:, :3]
        radial = distortions.unsqueeze(0)

    render_out = mesh_helper.render_normals_and_depth(
        vertices,
        Ks_batch,
        extr_batch,
        radial_distortions=radial,
        depth_rendering_height=height,
        depth_rendering_width=width,
        return_depth=True,
    )

    depth = render_out['depth_images'].view(vertices.shape[0], num_views, height, width)[0]
    normals = render_out['normal_images'].view(vertices.shape[0], num_views, height, width, 3)[0]
    mask = (depth > 0) & torch.isfinite(depth)

    return {'depth': depth, 'normals': normals, 'mask': mask}

# ----------------- depth->pointmap, normals, losses -----------------
def depth_batch_to_pointmaps(K_list, depth_batch):
    V = depth_batch.shape[0]
    p = [depth_to_pointmap_robust(K_list[v], depth_batch[v]) for v in range(V)]
    return torch.stack(p, 0)  # (V,H,W,3)

def loss_points_L1(p_pred, p_gt, mask):
    # p_*: (V,H,W,3), mask: (V,H,W)
    diff = (p_pred - p_gt).abs().sum(-1)
    return diff[mask].mean() if mask.any() else diff.mean()*0.0

def loss_normals_cos(n_pred, n_gt, mask):
    n_pred = F.normalize(n_pred, dim=-1)
    n_gt   = F.normalize(n_gt, dim=-1)
    cos = (n_pred * n_gt).sum(-1).clamp(-1,1)
    loss = 1.0 - cos
    return loss[mask].mean() if mask.any() else loss.mean()*0.0

# ----------------- main entry -----------------
def synth_views_compare_with_meshhelper(
    mesh_helper_pred: MeshHelper,
    pred_vertices: torch.Tensor,        # (B, Nv_pred, 3)
    scan_vertices: torch.Tensor,        # (B, Nv_scan, 3) or list
    Ks_existing: torch.Tensor,          # (B, V_existing, 3, 3)
    w2c_existing: torch.Tensor,         # (B, V_existing, 3, 4)
    n_virtual=4,
    depth_size=(300, 300),
    lam_points=1.0,
    lam_normals=0.2,
    lam_depth=1.0,
    distortions_existing: Optional[torch.Tensor] = None,
    scan_faces: Optional[torch.Tensor] = None,
):
    """
    Render virtual views for both predicted and scan meshes and compute simple
    point / normal / depth discrepancies.

    Returns:
        dict with:
            'loss_point', 'loss_normal', 'loss_depth' (averaged over batch),
            'pointmaps_pred', 'pointmaps_scan',
            'normals_pred', 'normals_scan',
            'depth_pred', 'depth_scan',
            'mask', 'Ks_virtual', 'w2c_virtual'
    """
    if scan_vertices is None:
        return None

    device = pred_vertices.device
    dtype = pred_vertices.dtype
    batch_size = pred_vertices.shape[0]

    def _get_sample(tensor_like, idx):
        if tensor_like is None:
            return None
        if isinstance(tensor_like, (list, tuple)):
            sample = tensor_like[idx]
            return sample.to(device=device, dtype=dtype)
        tensor = tensor_like
        if tensor.dim() >= 3 and tensor.shape[0] == batch_size:
            return tensor[idx].to(device=device, dtype=dtype)
        return tensor.to(device=device, dtype=dtype)

    pointmaps_pred_list: List[torch.Tensor] = []
    pointmaps_scan_list: List[torch.Tensor] = []
    normals_pred_list: List[torch.Tensor] = []
    normals_scan_list: List[torch.Tensor] = []
    depth_pred_list: List[torch.Tensor] = []
    depth_scan_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    Ks_virtual_list: List[torch.Tensor] = []
    w2c_virtual_list: List[torch.Tensor] = []
    loss_point_terms: List[torch.Tensor] = []
    loss_normal_terms: List[torch.Tensor] = []
    loss_depth_terms: List[torch.Tensor] = []

    for b_idx in range(batch_size):
        pred_b = pred_vertices[b_idx]
        scan_b = _get_sample(scan_vertices, b_idx)
        if scan_b is None:
            continue

        Ks_b = Ks_existing[b_idx]
        extr_b = w2c_existing[b_idx]
        if extr_b.shape[-1] == 4 and extr_b.shape[-2] == 3:
            w2c_existing_full = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(extr_b.shape[0], 1, 1)
            w2c_existing_full[:, :3, :] = extr_b
        else:
            w2c_existing_full = extr_b

        distortions_b = None
        if distortions_existing is not None:
            distortions_b = distortions_existing[b_idx]
            if distortions_b.dim() == 3 and distortions_b.shape[0] == 1:
                distortions_b = distortions_b[0]
            if distortions_b.dim() == 1:
                distortions_b = distortions_b.unsqueeze(0)

        target = _compute_mesh_center(pred_b)
        cams = sample_virtual_cams_like_rig(
            Ks_existing=Ks_b,
            w2c_existing=w2c_existing_full,
            n=n_virtual,
            res=(depth_size[1], depth_size[0]),
            target=target,
            distortions_existing=distortions_b,
        )

        Ks_v = torch.stack([c['K'] for c in cams], dim=0).to(device=device, dtype=dtype)
        w2c_v = torch.stack([c['w2c'] for c in cams], dim=0).to(device=device, dtype=dtype)

        if any('distortion' in c for c in cams):
            distortions_list = []
            for c in cams:
                d = c.get('distortion', None)
                if d is None:
                    d_tensor = torch.zeros(3, device=device, dtype=dtype)
                else:
                    d_tensor = d.to(device=device, dtype=dtype).view(-1)
                    if d_tensor.shape[0] < 3:
                        pad = torch.zeros(3 - d_tensor.shape[0], device=device, dtype=dtype)
                        d_tensor = torch.cat([d_tensor, pad], dim=0)
                    elif d_tensor.shape[0] > 3:
                        d_tensor = d_tensor[:3]
                distortions_list.append(d_tensor)
            distortions_v = torch.stack(distortions_list, dim=0)
        else:
            distortions_v = torch.zeros((len(cams), 3), device=device, dtype=dtype)

        Ks_virtual_list.append(Ks_v)
        w2c_virtual_list.append(w2c_v)

        pred_render = render_depth_with_meshhelper(
            mesh_helper=mesh_helper_pred,
            vertices=pred_b,
            Ks=Ks_v,
            w2cs=w2c_v,
            distortions=distortions_v,
            depth_size=depth_size,
        )

        scan_faces_b = _get_sample(scan_faces, b_idx)
        mesh_helper_scan = MeshHelper(num_vertices=scan_b.shape[0], faces=scan_faces_b.long().cpu())

        scan_render = render_depth_with_meshhelper(
            mesh_helper=mesh_helper_scan,
            vertices=scan_b,
            Ks=Ks_v,
            w2cs=w2c_v,
            distortions=distortions_v,
            depth_size=depth_size,
        )

        depth_pred = pred_render['depth']
        depth_scan = scan_render['depth']
        normals_pred = pred_render['normals']
        normals_scan = scan_render['normals']
        mask = pred_render['mask'] & scan_render['mask']

        Ks_batch = Ks_v.unsqueeze(0)
        extr_batch = w2c_v[:, :3, :].unsqueeze(0)
        depth_pred_batch = depth_pred.unsqueeze(0)
        depth_scan_batch = depth_scan.unsqueeze(0)

        pointmaps_pred = depth_to_pointmap_robust(
            depth_pred_batch,
            Ks_batch,
            extr_batch,
            rotated_views=[],
        )[0]
        pointmaps_scan = depth_to_pointmap_robust(
            depth_scan_batch,
            Ks_batch,
            extr_batch,
            rotated_views=[],
        )[0]

        loss_point = loss_points_L1(pointmaps_pred, pointmaps_scan, mask)
        loss_normal = loss_normals_cos(normals_pred, normals_scan, mask)
        depth_diff = (depth_pred - depth_scan).abs()
        loss_depth = depth_diff[mask].mean() if mask.any() else depth_diff.mean() * 0.0

        pointmaps_pred_list.append(pointmaps_pred)
        pointmaps_scan_list.append(pointmaps_scan)
        normals_pred_list.append(normals_pred)
        normals_scan_list.append(normals_scan)
        depth_pred_list.append(depth_pred)
        depth_scan_list.append(depth_scan)
        mask_list.append(mask)
        loss_point_terms.append(loss_point)
        loss_normal_terms.append(loss_normal)
        loss_depth_terms.append(loss_depth)

    if not pointmaps_pred_list:
        return None

    pointmaps_pred_tensor = torch.stack(pointmaps_pred_list, dim=0)
    pointmaps_scan_tensor = torch.stack(pointmaps_scan_list, dim=0)
    normals_pred_tensor = torch.stack(normals_pred_list, dim=0)
    normals_scan_tensor = torch.stack(normals_scan_list, dim=0)
    depth_pred_tensor = torch.stack(depth_pred_list, dim=0)
    depth_scan_tensor = torch.stack(depth_scan_list, dim=0)
    mask_tensor = torch.stack(mask_list, dim=0)
    Ks_virtual_tensor = torch.stack(Ks_virtual_list, dim=0)
    w2c_virtual_tensor = torch.stack(w2c_virtual_list, dim=0)

    raw_loss_point = torch.stack(loss_point_terms).mean()
    raw_loss_normal = torch.stack(loss_normal_terms).mean()
    raw_loss_depth = torch.stack(loss_depth_terms).mean()

    loss_point_mean = raw_loss_point * lam_points
    loss_normal_mean = raw_loss_normal * lam_normals
    loss_depth_mean = raw_loss_depth * lam_depth

    return {
        'loss_point': loss_point_mean,
        'loss_normal': loss_normal_mean,
        'loss_depth': loss_depth_mean,
        'loss_point_raw': raw_loss_point,
        'loss_normal_raw': raw_loss_normal,
        'loss_depth_raw': raw_loss_depth,
        'pointmaps_pred': pointmaps_pred_tensor,
        'pointmaps_scan': pointmaps_scan_tensor,
        'normals_pred': normals_pred_tensor,
        'normals_scan': normals_scan_tensor,
        'depth_pred': depth_pred_tensor,
        'depth_scan': depth_scan_tensor,
        'mask': mask_tensor,
        'Ks_virtual': Ks_virtual_tensor,
        'w2c_virtual': w2c_virtual_tensor,
    }

# ----------------- visualization -----------------
def visualize_synth(res: Dict[str, torch.Tensor], batch_index: int = 0, max_views: int = 4, figsize=(14, 8)):
    if res is None:
        return

    pointmaps_scan = res['pointmaps_scan'][batch_index]
    pointmaps_pred = res['pointmaps_pred'][batch_index]
    normals_scan = res['normals_scan'][batch_index]
    normals_pred = res['normals_pred'][batch_index]
    num_views = min(max_views, pointmaps_scan.shape[0])

    def _np(x):
        return x.detach().float().cpu().numpy()

    scan_rgb = [_np(pointmap_to_rgb(pointmaps_scan[v])).clip(0, 1) for v in range(num_views)]
    pred_rgb = [_np(pointmap_to_rgb(pointmaps_pred[v])).clip(0, 1) for v in range(num_views)]

    normals_scan_vis = (_np(normals_scan[:num_views]) * 0.5 + 0.5).clip(0, 1)
    normals_pred_vis = (_np(normals_pred[:num_views]) * 0.5 + 0.5).clip(0, 1)
    n_diff = (normals_scan[:num_views] - normals_pred[:num_views]).abs().mean(-1, keepdim=True).repeat(1, 1, 1, 3)
    n_diff_vis = _np(n_diff).clip(0, 1)

    fig, ax = plt.subplots(num_views, 4, figsize=figsize)
    if num_views == 1:
        ax = np.array([ax])

    for v in range(num_views):
        ax[v, 0].imshow(normals_scan_vis[v])
        ax[v, 0].set_title("scan normals")
        ax[v, 0].axis('off')

        ax[v, 1].imshow(normals_pred_vis[v])
        ax[v, 1].set_title("pred normals")
        ax[v, 1].axis('off')

        ax[v, 2].imshow(n_diff_vis[v])
        ax[v, 2].set_title("|Δn|")
        ax[v, 2].axis('off')

        ax[v, 3].imshow(np.hstack([scan_rgb[v], pred_rgb[v]]))
        ax[v, 3].set_title("points scan|pred")
        ax[v, 3].axis('off')

    raw_lp = res.get('loss_point_raw')
    raw_ln = res.get('loss_normal_raw')
    raw_ld = res.get('loss_depth_raw')
    title = "Virtual view comparison"
    if raw_lp is not None and raw_ln is not None and raw_ld is not None:
        title = f"Lp={raw_lp.item():.4f}, Ln={raw_ln.item():.4f}, Ld={raw_ld.item():.4f}"
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
