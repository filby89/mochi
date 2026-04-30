import torch
from utils.point_to_surface_loss import PointToSurfaceLoss, compute_s2m_distance, GMO, batch_chamfer_distance
from utils.utils import print_memory, to_numpy, get_time_string, add_labels_to_images
import torch.nn.functional as F
import kornia

def geodesic_normal_loss_p3d(
    n_pred: torch.Tensor,         # (B, V, 3, H, W) or (B*V, 3, H, W)
    n_gt:   torch.Tensor,         # same shape
    mask:   torch.Tensor = None,  # (B, V, H, W) or (B*V, H, W) or None
    reduction: str = "mean",      # "mean" | "sum" | "none"
    eps: float = 1e-8,
    return_per_pixel: bool = True
):
    """
    Geodesic angle on S^2 between predicted and GT normals:
      angle = atan2(||n_pred x n_gt||, n_pred · n_gt)
    Returns loss scalar (or map) and, optionally, the per-pixel angle map (radians).
    """

    # Flatten (B,V,...) -> (BV,...)
    def _flat(x):
        if x.dim() == 5:
            B, V = x.shape[:2]
            return x.view(B*V, *x.shape[2:])
        return x

    n_pred = _flat(n_pred)  # (BV, 3, H, W)
    n_gt   = _flat(n_gt)

    if mask is not None:
        mask = _flat(mask)  # (BV, H, W)

    # Unit-length (in case upstream renorm was skipped)
    n_pred = F.normalize(n_pred, dim=1, eps=eps)
    n_gt   = F.normalize(n_gt,   dim=1, eps=eps)

    # Dot and cross
    dot = (n_pred * n_gt).sum(dim=1).clamp(-1.0, 1.0)   # (BV, H, W)
    cross = torch.cross(n_pred, n_gt, dim=1)            # (BV, 3, H, W)
    cross_norm = cross.norm(dim=1)                      # (BV, H, W)

    # Geodesic angle (radians)
    angle = torch.atan2(cross_norm, dot)                # (BV, H, W)

    if mask is not None:
        angle = angle * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
    else:
        denom = angle.numel()

    if reduction == "mean":
        loss = angle.sum() / denom
    elif reduction == "sum":
        loss = angle.sum()
    else:  # "none"
        loss = angle

    if return_per_pixel:
        return loss, angle
    return loss



def calculate_map_loss(pred, gt, mask=None, robust=False, gmo_sigma=2):
    loss = torch.sqrt((pred - gt).pow(2).sum(dim=1, keepdim=True) + 1e-12)

    if robust:
        loss = GMO(loss, sigma=gmo_sigma)  # Apply Geman-McClure robustifier

    if mask is not None:
        mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
        loss = loss * mask

    masked_mean = loss.mean()
        
    return masked_mean, loss.detach()
 
def calculate_gradient_map_loss(pred, gt, mask=None, robust=False, gmo_sigma=2):
    """
    L_grad = || ∇(pred) - ∇(gt) ||_2  (per pixel, summed over channels and xy)
    pred, gt: (B, C, H, W) in [0,1]
    Returns:
        mean_loss (scalar),
        per_pixel_loss (B, H, W),
        grad_mag_pred (B, H, W),
        grad_mag_gt   (B, H, W)
    """
    # Sobel gradients with Kornia (installed & already imported)
    grad_op = kornia.filters.SpatialGradient(mode='sobel', order=1, normalized=True)

    g_pred = grad_op(pred)  # (B, C, 2, H, W)  (dx, dy)
    g_gt   = grad_op(gt)    # (B, C, 2, H, W)

    # per-pixel L2 over (C,2)
    diff = g_pred - g_gt
    per_pixel = torch.sqrt(diff.pow(2).sum(dim=2).sum(dim=1) + 1e-12)  # (B, H, W)

    if robust:
        per_pixel = GMO(per_pixel.unsqueeze(1), sigma=gmo_sigma).squeeze(1)  # robustify (optional)

    if mask is not None:
        # mask: (B, H, W) or (B,1,H,W)
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        per_pixel = per_pixel * mask

    mean_loss = per_pixel.mean()

    # gradient magnitudes for visualization
    grad_mag_pred = torch.sqrt(g_pred.pow(2).sum(dim=2).sum(dim=1) + 1e-12)  # (B,H,W)
    grad_mag_gt   = torch.sqrt(g_gt.pow(2).sum(dim=2).sum(dim=1) + 1e-12)    # (B,H,W)

    return mean_loss, per_pixel.detach(), grad_mag_pred.detach(), grad_mag_gt.detach()




def _points2surface_metric(points_to_use, v_scan, v_registration, f_reg_global, flame_masks_triangles):
    losses = {}
    device = points_to_use.device
    for i in range(len(v_scan)):
        # print(len(v_scan[i].shape), points_to_use.shape, len(v_registration))
        scan_vertices = v_scan[i].to(device).unsqueeze(0)
        predicted_vertices = points_to_use[i].unsqueeze(0).float()
        predicted_faces = f_reg_global[0].to(device)
        registration_vertices = v_registration[i].to(device).float().unsqueeze(0) # this is wrong !!!!!

        with torch.no_grad():
            distances_dict = compute_s2m_distance(scan_vertices, predicted_vertices, predicted_faces, 
                                                masks=flame_masks_triangles,
                                                registration_vertices_for_regions=registration_vertices,
                                                exclude_regions=['boundary'])
            for key in distances_dict:
                if key not in losses:
                    losses[key] = []
                losses[key].extend(to_numpy(distances_dict[key]).tolist())

    return losses
