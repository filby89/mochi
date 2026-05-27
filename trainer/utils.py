import torch
import kornia
from utils.point_to_surface_loss import PointToSurfaceLoss, compute_s2m_distance, GMO
import numpy as np
import cv2
from typing import Optional
from utils.utils import print_memory, to_numpy, get_time_string, add_labels_to_images

def to_mm(x):
    return x*1000.0

def to_m(x):
    return x/1000.0

def pixels_to_uv(landmarks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Map pixel-coord landmarks to normalized UV in [-1, 1]. Operates on (..., 0) and (..., 1)."""
    out = landmarks.clone()
    out[..., 0] = out[..., 0] / (width - 1) * 2.0 - 1.0
    out[..., 1] = out[..., 1] / (height - 1) * 2.0 - 1.0
    return out

def random_pixel_mask(x: torch.Tensor, keep_prob=0.01, visibility_mask: torch.Tensor = None):
    B, C, H, W = x.shape

    # Create a single-channel mask with Bernoulli sampling
    mask = (torch.rand((B, 1, H, W), device=x.device) < keep_prob).type_as(x)
    if visibility_mask is not None:
        kernel = torch.ones((15, 15), device=x.device)
        visibility_mask = visibility_mask.unsqueeze(1)
        for _ in range(1):
            visibility_mask = kornia.morphology.erosion(visibility_mask, kernel)

        mask = mask.bool() + visibility_mask.bool()  # Combine with visibility mask
        # dilate
        mask = mask.float()  # Convert to float for multiplication

    # Apply mask to all channels
    masked_x = x * mask
    return masked_x, mask

def calculate_map_loss(pred, gt, mask=None, robust=False, gmo_sigma=2):
    loss = torch.sqrt((pred - gt).pow(2).sum(dim=1, keepdim=True) + 1e-12)

    if robust:
        loss = GMO(loss, sigma=gmo_sigma)  # Apply Geman-McClure robustifier

    if mask is not None:
        mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
        loss = loss * mask

    masked_mean = loss.mean()
        
    return masked_mean, loss.detach()



def get_dense_vertex_colors(n_verts, flame_masks):
    dense_palette_bgr = [
        (192, 192, 192),  # silver
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (128, 0, 255),    # purple-ish
        (0, 128, 255),    # orange-ish
        (255, 128, 0),    # teal-ish
        (128, 255, 0),    # lime-ish
        (0, 255, 128),    # spring green-ish
        (255, 0, 128),    # pink-ish
        (128, 128, 0),    # olive-ish
        (128, 0, 128),    # purple
        (0, 128, 128),    # teal
        (128, 128, 128),  # gray
        (0, 0, 0),        # black
        (255, 255, 255),  # white
    ]
    dense_vis_regions = ['neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'forehead',
                                'lips', 'nose', 'left_ear']
    colors = np.full((n_verts, 3), 200, dtype=np.uint8)
    for ridx, name in enumerate(dense_vis_regions):
        idx_list = flame_masks.get(name, [])

        idx_arr = np.asarray(idx_list, dtype=np.int64)
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < n_verts)]

        color = np.array(dense_palette_bgr[ridx % len(dense_palette_bgr)], dtype=np.uint8)
        colors[idx_arr] = color
    return colors


def draw_dense_points(bgr_image: np.ndarray,
                        pts: np.ndarray,
                        colors: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        radius: int = 1) -> np.ndarray:
    h, w = bgr_image.shape[:2]
    num_pts = min(len(pts), len(colors))
    mask_flat = None
    if mask is not None:
        mask_flat = np.asarray(mask).reshape(-1)
    for i in range(num_pts):
        if mask_flat is not None and i < mask_flat.size and mask_flat[i] <= 0:
            continue
        x = int(round(float(pts[i][0])))
        y = int(round(float(pts[i][1])))
        if 0 <= x < w and 0 <= y < h:
            color = colors[i]
            if isinstance(color, np.ndarray):
                color_tuple = tuple(int(c) for c in color.tolist())
            else:
                color_tuple = tuple(int(c) for c in color)
            cv2.circle(bgr_image, (x, y), radius, color_tuple, -1, lineType=cv2.LINE_AA)
    return bgr_image




# def _points2surface_metric(v_scan, points_to_use, faces, registration, flame_masks_triangles, to_millimeters=True):
#     losses = {}
#     for i in range(len(v_scan)):
#         scan_vertices = v_scan[i].to(points_to_use.device).unsqueeze(0)
#         predicted_vertices = points_to_use[i].unsqueeze(0).float()
#         predicted_faces = faces.to(points_to_use.device) #.unsqueeze(0)
#         registration_vertices = registration[i].to(points_to_use.device).float().unsqueeze(0)

#         if to_millimeters:
#             scan_vertices = to_mm(scan_vertices)
#             predicted_vertices = to_mm(predicted_vertices)
#             registration_vertices = to_mm(registration_vertices)

#         # print(scan_vertices.shape, predicted_vertices.shape, predicted_faces.shape, registration_vertices.shape)
#         with torch.no_grad():
#             distances_dict = compute_s2m_distance(scan_vertices, predicted_vertices, predicted_faces, 
#                                                 masks=flame_masks_triangles,
#                                                 registration_vertices_for_regions=registration_vertices,
#                                                 exclude_regions=['boundary'])
#             for key in distances_dict:
#                 if key not in losses:
#                     losses[key] = []
#                 losses[key].extend(to_numpy(distances_dict[key]).tolist())

#     return losses
