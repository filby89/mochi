# pliks_flame.py
# Minimal PLIKS-style linearization & inverse solve for FLAME
# Author: you + ChatGPT

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from utils.mesh_renderer import render_mesh, dist_to_rgb, render_mesh_pixels
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely

import imageio
from psbody.mesh import Mesh
import cv2
import pyrender
import trimesh
import matplotlib as mpl
from pytorch3d.ops import corresponding_points_alignment

# =========================
# Helpers: rotations → angles
# =========================

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


def world_to_relative_rotations(R_world: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    R_world: [B,J,3,3] world rotation per joint (root in world)
    parents: [J] long, -1 for root
    Returns R_rel: [B,J,3,3] relative to parent (R_rel[j] = R_parent^T R_world[j])
    """
    B, J = R_world.shape[:2]
    R_rel = R_world.clone()
    for j in range(J):
        p = int(parents[j].item())
        if p >= 0:
            R_rel[:, j] = torch.matmul(R_world[:, p].transpose(1,2), R_world[:, j])
    return R_rel

def mat_to_axis_angle(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    R: [B,J,3,3] -> axis-angle [B,J,3]
    """
    B, J = R.shape[:2]
    tr = R[...,0,0] + R[...,1,1] + R[...,2,2]
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # [B,J]

    wx = R[...,2,1] - R[...,1,2]
    wy = R[...,0,2] - R[...,2,0]
    wz = R[...,1,0] - R[...,0,1]
    w = torch.stack([wx, wy, wz], dim=-1)  # [B,J,3]

    s = 2.0 * torch.sin(theta).unsqueeze(-1) + eps
    axis = w / s
    aa = axis * theta.unsqueeze(-1)
    return aa  # [B,J,3]

def build_R_world_from_segments(Rk: torch.Tensor,
                                seg_list: torch.Tensor,
                                J: int) -> torch.Tensor:
    """
    Fill a [B,J,3,3] tensor with identity, then copy per-segment rotations
    for joints listed in seg_list (those present in the segmentation).
    Rk: [B,K,3,3]
    seg_list: [K] long, joint ids (0..J-1) that the segments correspond to
    """
    B, K = Rk.shape[:2]
    device = Rk.device
    R_world = torch.eye(3, device=device).view(1,1,3,3).repeat(B, J, 1, 1)
    for si in range(K):
        j = int(seg_list[si].item())
        if 0 <= j < J:
            R_world[:, j] = Rk[:, si]
    return R_world


def render_mesh_helper(mesh, t_center=np.zeros(3), rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    # mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center

    texture_rendering = tex_img is not None and hasattr(mesh, 'vt') and hasattr(mesh, 'ft')
    if texture_rendering:
        intensity = 0.5
        tex = pyrender.Texture(source=tex_img, source_channels='RGB') 
        material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex)

        # Workaround as pyrender requires number of vertices and uv coordinates to be the same
        temp_filename = '%s.obj' % next(tempfile._get_candidate_names())
        mesh.write_obj(temp_filename)
        tri_mesh = trimesh.load(temp_filename, process=False)
        try:
            os.remove(temp_filename)
        except:
            print('Failed deleting temporary file - %s' % temp_filename)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
    elif errors is not None:
        intensity = 0.5
        unit_factor = get_unit_factor('mm')/get_unit_factor(error_unit)
        errors = unit_factor*errors

        norm = mpl.colors.Normalize(vmin=min_dist_in_mm, vmax=max_dist_in_mm)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba_per_v = colormapper.to_rgba(errors)
        rgb_per_v = rgba_per_v[:, 0:3]
    elif v_colors is not None:
        intensity = 0.5
        rgb_per_v = v_colors
    else:
        intensity = 5.0
        rgb_per_v = None

    if not texture_rendering:
        tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.PointLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

# @torch.no_grad()
def max_weight_segments(W: torch.Tensor, locked_joint_ids: Optional[torch.Tensor]=None):
    # W: [V, J+1]; locked_joint_ids: 1D long tensor of joint ids to exclude
    # NOTE:
    # On some newer GPUs (e.g. B200) older PyTorch/CUDA builds may fail on
    # CUDA advanced indexing in this one-time preprocessing step with:
    # "no kernel image is available for execution on the device".
    # Compute segment assignment on CPU, then move result back.
    device = W.device
    W_cpu = W.detach().cpu()
    if locked_joint_ids is None or locked_joint_ids.numel() == 0:
        seg_idx = W_cpu.argmax(dim=1).long()
        return seg_idx.to(device)
    locked_cpu = locked_joint_ids.detach().long().cpu()
    Wm = W_cpu.clone()
    Wm[:, locked_cpu] = -1e9
    seg_idx = Wm.argmax(dim=1).long()
    return seg_idx.to(device)

def build_linear_system(
    V_templ, B, seg_idx, Rk, V_pred, t=None,
    seg_list=None, v_mask=None, v_wts=None
):
    """
    Build A x = b for x = [beta, t], with:
       R_s(i) B_i @ beta + t  ≈  X_pred_i - R_s(i) X0_i
    Returns:
       A: [B, 3*V_used, NB+3]
       b: [B, 3*V_used]
    """
    device = V_templ.device
    V, _, NB = B.shape
    if seg_list is None:
        seg_list = torch.unique(seg_idx)
    K = seg_list.numel()

    if v_mask is None:
        v_mask = torch.ones(V, dtype=torch.bool, device=device)[None, :].expand(Rk.shape[0], -1)
    elif v_mask.dim() == 1:
        v_mask = v_mask[None, :].expand(Rk.shape[0], -1)
    if v_wts is None:
        v_wts = torch.ones_like(v_mask, dtype=V_templ.dtype)

    max_jid = int(seg_idx.max().item())
    lut = torch.full((max_jid+1,), -1, dtype=torch.long, device=device)
    lut[seg_list] = torch.arange(K, device=device, dtype=torch.long)
    seg_pos = lut[seg_idx]  # [V]

    B_resh = B.view(V, 3, NB)     # [V,3,NB]
    X0 = V_templ                  # [V,3]

    A_batch, b_batch = [], []
    Bsz = Rk.shape[0]
    for b in range(Bsz):
        m = v_mask[b]                            # [V]
        w = v_wts[b][m][:, None]                 # [V_used,1]
        pos = seg_pos[m]                         # [V_used]
        R_sel = Rk[b, pos]                       # [V_used,3,3]

        Bi = B_resh[m]                           # [V_used,3,NB]
        RiBi = torch.matmul(R_sel, Bi)           # [V_used,3,NB]
        A_beta = RiBi.reshape(-1, NB)            # [3*V_used, NB]

        A_t = torch.eye(3, device=device).expand(pos.numel(), 3, 3).reshape(-1, 3)  # [3*V_used,3]

        Xi0 = X0[m]                               # [V_used,3]
        Xp  = V_pred[b, m]                        # [V_used,3]
        rhs = (Xp - torch.matmul(R_sel, Xi0.unsqueeze(-1)).squeeze(-1)).reshape(-1)  # [3*V_used]

        A = torch.cat([A_beta, A_t], dim=1)      # [3*V_used, NB+3]

        # weights
        w3 = w.repeat_interleave(3, dim=0).reshape(-1)  # [3*V_used]
        A = A * w3[:, None]
        rhs = rhs * w3

        A_batch.append(A)
        b_batch.append(rhs)

    A_full = torch.stack(A_batch, dim=0)  # [B, 3*V_used, NB+3]
    b_full = torch.stack(b_batch, dim=0)  # [B, 3*V_used]
    return A_full, b_full


# @torch.no_grad()
def estimate_segment_rotations_svd(
    V_src: torch.Tensor,         # [B, V, 3], "source" points (template or template + shape) per batch
    V_tgt: torch.Tensor,         # [B, V, 3], "target" points (predicted vertices)
    seg_idx: torch.Tensor,       # [V] int64 (0..J)
    seg_list: Optional[torch.Tensor]=None,  # [K]
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Per-segment Procrustes (no scale) using SVD.
    Returns:
        Rk: [B, K, 3, 3]
    """
    device = V_src.device
    if seg_list is None:
        seg_list = torch.unique(seg_idx)
    K = seg_list.numel()
    Bsz, V, _ = V_src.shape
    Rk = torch.eye(3, device=device).view(1,1,3,3).repeat(Bsz, K, 1, 1).clone()

    # Build lookup: global joint id -> 0..K-1
    max_jid = int(seg_idx.max().item())
    lut = torch.full((max_jid+1,), -1, dtype=torch.long, device=device)
    lut[seg_list] = torch.arange(K, device=device, dtype=torch.long)
    seg_pos = lut[seg_idx]  # [V]

    for k in range(K):
        mask = (seg_pos == k)  # [V]
        if mask.sum() < 3:
            # too few points, keep identity
            continue
        X = V_src[:, mask, :]  # [B, Nk, 3]
        Y = V_tgt[:, mask, :]  # [B, Nk, 3]
        # Center
        Xc = X - X.mean(dim=1, keepdim=True)
        Yc = Y - Y.mean(dim=1, keepdim=True)
        # Covariance
        H = torch.matmul(Xc.transpose(1,2), Yc)  # [B, 3, 3]
        U, S, Vt = torch.linalg.svd(H)           # Vt is V^T
        R = torch.matmul(Vt.transpose(1,2), U.transpose(1,2))  # [B,3,3]
        # Handle reflection
        det = torch.det(R)
        fix = (det < 0).float().view(-1,1,1)
        if fix.any():
            Vt_fix = Vt.clone()
            Vt_fix[:, 2, :] *= -1.0
            R = torch.matmul(Vt_fix.transpose(1,2), U.transpose(1,2))
        Rk[:, k] = R
        
    return Rk

def estimate_segment_rotations_p3d(V_src, V_tgt, seg_idx, seg_list=None):
    """
    V_src, V_tgt: [B, V, 3]
    seg_idx: [V]  (vertex -> joint)
    seg_list: [K] joints to estimate (subset of unique(seg_idx))
    Returns: Rk [B, K, 3, 3]  (left-multiply convention)
    """
    device = V_src.device
    if seg_list is None:
        seg_list = torch.unique(seg_idx)
    B, V, _ = V_src.shape
    K = seg_list.numel()

    Rk = torch.eye(3, device=device).view(1,1,3,3).repeat(B, K, 1, 1).clone()

    for k_idx, j in enumerate(seg_list.tolist()):
        mask = (seg_idx == j)
        if mask.sum() < 3:
            continue  # too few points → keep identity

        P = V_src[:, mask, :]  # [B, Nk, 3]
        Q = V_tgt[:, mask, :]  # [B, Nk, 3]

        # estimate scale=False (we only want rotation + translation)
        R_row, T, s = corresponding_points_alignment(P, Q, estimate_scale=False)
        # R_row maps row vectors: s*(P @ R_row^T)+T ≈ Q
        # Convert to left-multiply convention used elsewhere (R @ x):
        R_left = R_row.transpose(-1, -2)  # [B,3,3]

        Rk[:, k_idx] = R_left

    return Rk

def apply_segments_transform(
    V_src: torch.Tensor,     # [B,V,3]
    Rk: torch.Tensor,        # [B,K,3,3]
    seg_idx: torch.Tensor,   # [V]
    seg_list: Optional[torch.Tensor]=None,
    t: Optional[torch.Tensor]=None   # [B,3]
) -> torch.Tensor:
    """
    Produces per-vertex transformed points: R_{s(i)} X_i + t
    """
    device = V_src.device
    if seg_list is None:
        seg_list = torch.unique(seg_idx)
    K = seg_list.numel()

    # lut: global joint id -> 0..K-1
    max_jid = int(seg_idx.max().item())
    lut = torch.full((max_jid+1,), -1, dtype=torch.long, device=device)
    lut[seg_list] = torch.arange(K, device=device, dtype=torch.long)
    seg_pos = lut[seg_idx]  # [V]

    Bsz, V, _ = V_src.shape
    R_sel = Rk[:, seg_pos]                    # [B,V,3,3]
    out = torch.matmul(R_sel, V_src.unsqueeze(-1)).squeeze(-1)  # [B,V,3]
    if t is not None:
        out = out + t[:, None, :]
    return out

class PliksFlameSolver(nn.Module):
    """
    PLIKS-style inverse for FLAME.
    - Ignores pose blendshapes for the linear step.
    - Segments mesh by argmax LBS weight.
    - Estimates per-segment rotations by SVD.
    - Solves for [beta; t] via batched least squares.
    - Optionally iterate once more.

    Expects FLAME buffers:
      v_template [V,3]
      shapedirs  [V,3,NB]
      lbs_weights[V,J+1]
    """
    def __init__(self, flame, seg_list: Optional[torch.Tensor]=None, locked_joint_ids: Optional[torch.Tensor]=None):
        super().__init__()
        self.register_buffer("v_template", flame.v_template.clone())            # [V,3]
        self.register_buffer("B_all", flame.shapedirs.clone())                  # [V,3,NB]
        self.register_buffer("W", flame.lbs_weights.clone())                    # [V,J+1]
        self.seg_idx = max_weight_segments(self.W, locked_joint_ids)                              # [V]
        # print(self.seg_idx.shape)
        # seg_list = torch.Tensor(np.array([0,1,2,3,4])).long()
        if seg_list is None:
            seg_list = torch.unique(self.seg_idx.detach().cpu()).to(self.seg_idx.device)
        self.register_buffer("seg_list", seg_list.clone())
        # print(self.seg_list, self.seg_idx)

        if locked_joint_ids is None:
            locked_joint_ids = torch.empty(0, dtype=torch.long)
        self.register_buffer("locked_joint_ids", locked_joint_ids.clone())

        self.NB = self.B_all.shape[-1]
        self.V = self.v_template.shape[0]
        self.K = self.seg_list.numel()

    # @torch.no_grad()
    def initial_rotations(self, V_pred: torch.Tensor, beta: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Compute initial segment rotations using ARE/SVD.
        Args:
            V_pred: [B,V,3] predicted (target) vertices
            beta:   [B,NB] optional, to warp template by shape before rotation estimation
        """
        Bsz = V_pred.shape[0]
        if beta is None:
            V_src = self.v_template[None, :, :].expand(Bsz, -1, -1)  # [B,V,3]
        else:
            # shape-only displacement (no pose blends)
            disp = torch.einsum('bL, v c L -> b v c', beta, self.B_all)  # [B,V,3]
            V_src = self.v_template[None, :, :] + disp

        Rk = estimate_segment_rotations_svd(V_src, V_pred, self.seg_idx, self.seg_list)
        # Rk = estimate_segment_rotations_p3d(V_src, V_pred, self.seg_idx, self.seg_list)
        # print(Rk.shape)

        # for locked_joint_id in self.locked_joint_ids.tolist():
            # Rk[:, locked_joint_id] = torch.eye(3, device=Rk.device).unsqueeze(0)

        if self.locked_joint_ids.numel() > 0:
            # print('asdf')
            # map global joint ids -> local k in seg_list
            for j in self.locked_joint_ids.tolist():
                k = (self.seg_list == j).nonzero(as_tuple=False)
                if k.numel() > 0:
                    k = k.item()
                    Rk[:, k] = torch.eye(3, device=Rk.device).unsqueeze(0)
        # print(Rk.shape)
        # print(Rk[:,2])
        return Rk

    # def linear_solve(self, V_pred, Rk, v_mask=None, v_wts=None, damping: float = 1e-6):
    #     """
    #     Differentiable solve for x = [beta; t] using normal equations:
    #         (A^T A + λI) x = A^T b
    #     """
    #     A, b = build_linear_system(
    #         self.v_template, self.B_all, self.seg_idx, Rk, V_pred,
    #         seg_list=self.seg_list, v_mask=v_mask, v_wts=v_wts
    #     )  # A: [B, M, NB+3], b: [B, M]

    #     AT = A.transpose(1, 2)                          # [B, NB+3, M]
    #     ATA = AT @ A                                     # [B, NB+3, NB+3]
    #     ATb = AT @ b.unsqueeze(-1)                       # [B, NB+3, 1]

    #     # Levenberg–Marquardt style damping to keep things well-posed
    #     Bsz, D, _ = ATA.shape
    #     I = torch.eye(D, device=A.device).unsqueeze(0).expand(Bsz, -1, -1)
    #     ATA_damped = ATA + damping * I

    #     x = torch.linalg.solve(ATA_damped, ATb).squeeze(-1)  # [B, NB+3]
    #     beta = x[:, :self.NB]
    #     t    = x[:, self.NB:]
    #     return beta, t

    def linear_solve(
        self, V_pred, Rk, v_mask=None, v_wts=None,
        method: str = "qr", damping: float = 1e-6
    ):
        """
        Solve x = [beta; t] from A x ≈ b.
        method ∈ {"qr","ne","lstsq"}:
          - "qr":    batched reduced-QR, solve R x = Q^T b  (most stable; no AᵀA)
          - "ne":    (AᵀA + λI) x = Aᵀ b    (fast but squares condition #)
          - "lstsq": torch.linalg.lstsq (QR/SVD backend, robust fallback)
        """
        A, b = build_linear_system(
            self.v_template, self.B_all, self.seg_idx, Rk, V_pred,
            seg_list=self.seg_list, v_mask=v_mask, v_wts=v_wts
        )  # A: [B, M, D], b: [B, M]
        Bsz, M, D = A.shape

        if method == "qr":
            Q, R = torch.linalg.qr(A, mode="reduced")           # Q:[B,M,K], R:[B,K,D], K=min(M,D)=D here
            y    = Q.transpose(-2, -1) @ b.unsqueeze(-1)        # [B, D, 1]
            # Prefer triangular solve if available; else generic solve is fine
            try:
                x = torch.linalg.solve(R, y).squeeze(-1)        # [B, D]
            except RuntimeError:
                # rare rank issues → gentle Tikhonov on R
                I = torch.eye(R.shape[-1], device=R.device).unsqueeze(0).expand(Bsz, -1, -1)
                x = torch.linalg.solve(R.transpose(-2,-1) @ R + 1e-8*I,
                                        R.transpose(-2,-1) @ y).squeeze(-1)

        elif method == "ne":
            AT  = A.transpose(1, 2)                              # [B,D,M]
            ATA = AT @ A                                         # [B,D,D]
            ATb = AT @ b.unsqueeze(-1)                           # [B,D,1]
            I   = torch.eye(D, device=A.device).unsqueeze(0).expand(Bsz, -1, -1)
            x   = torch.linalg.solve(ATA + damping*I, ATb).squeeze(-1)

        else:  # "lstsq"
            x = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)

        beta = x[:, :self.NB]
        t    = x[:, self.NB:]
        return beta, t


    # def forward(
    #     self,
    #     V_pred: torch.Tensor,                   # [B,V,3] predicted vertices in FLAME topology (or sampled accordingly)
    #     iters: int = 1,
    #     v_mask: Optional[torch.Tensor] = None,  # [B,V] or [V]
    #     v_wts: Optional[torch.Tensor]  = None   # [B,V]
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Returns:
    #         beta: [B,NB]         (shape+expr concatenated)
    #         t:    [B,3]
    #         Rk:   [B,K,3,3]
    #         V_fit:[B,V,3]        reconstructed vertices = R_s (X0 + B beta) + t
    #     """
    #     # 1) initial rotations from template (no shape)
    #     Rk = self.initial_rotations(V_pred, beta=None)

    #     beta = None
    #     t = None
    #     for _ in range(max(1, iters)):
    #         # 2) linear least-squares for [beta; t] with fixed Rk
    #         beta, t = self.linear_solve(V_pred, Rk, v_mask=v_mask, v_wts=v_wts)

    #         # 3) (optional) re-estimate rotations with current shape
    #         if iters > 1:
    #             Rk = self.initial_rotations(V_pred, beta=beta)

    #     # 4) reconstruct vertices with final params
    #     disp = torch.einsum('bL, v c L -> b v c', beta, self.B_all)  # [B,V,3]
    #     V_src = self.v_template[None, :, :] + disp
    #     V_fit = apply_segments_transform(V_src, Rk, self.seg_idx, self.seg_list, t=t)  # [B,V,3]

    #     return dict(beta=beta, t=t, Rk=Rk, V_fit=V_fit)
    def forward(
        self,
        V_pred: torch.Tensor,                   # [B,V,3]
        iters: int = 1,
        v_mask: Optional[torch.Tensor] = None,  # [B,V] or [V]
        v_wts: Optional[torch.Tensor]  = None,  # [B,V]
        lsq_method: str = "qr",                 # "qr" | "ne" | "lstsq"
        estimate_root: bool = False             # add rigid R_root,t_root via Procrustes
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          beta,t,Rk,V_fit
          (+) R_root, t_root, V_fit_root  if estimate_root=True
        """
        # 1) initial rotations from template (no shape)
        Rk = self.initial_rotations(V_pred, beta=None)

        beta = None
        t = None
        for _ in range(max(1, iters)):
            # 2) linear least-squares for [beta; t] with fixed Rk
            beta, t = self.linear_solve(V_pred, Rk, v_mask=v_mask, v_wts=v_wts, method=lsq_method)

            # 3) (optional) re-estimate rotations with current shape
            if iters > 1:
                Rk = self.initial_rotations(V_pred, beta=beta)
            
            
        # 4) reconstruct vertices with segment rotations + translation
        disp  = torch.einsum('bL, v c L -> b v c', beta, self.B_all)  # [B,V,3]
        V_src = self.v_template[None, :, :] + disp
        X_no_root = apply_segments_transform(V_src, Rk, self.seg_idx, self.seg_list, t=t)  # [B,V,3]
        
        out = dict(beta=beta, t=t, Rk=Rk, V_fit=X_no_root)

        if estimate_root:
            # Align (no scale): R_root X + t_root ≈ V_pred
            R_root, t_root = rigid_align_no_scale(X_no_root, V_pred)     # [B,3,3], [B,3]
            V_fit_root = (R_root[:, None] @ X_no_root.unsqueeze(-1)).squeeze(-1) + t_root[:, None, :]
            out.update(R_root=R_root, t_root=t_root, V_fit_root=V_fit_root)

        return out

    def compute_regularizers(
        self,
        pliks_out: dict,
        global_points: torch.Tensor,
        flame,
        *,
        weight_shape: float = 0.0,
        weight_expression: float = 0.0,
        weight_pose: float = 0.0,
        weight_vertices: float = 0.0,
        weight_vertices_edge: float = 0.0,
        edge_loss_fn=None,
        no_jaw: bool = False,
        to_meters: bool = False,
        unit_factor: float = 1000.0,
    ):
        """PLIKS regularization losses + FLAME-fitted vertices aligned to global_points.

        Returns (losses_dict, V_flame_mm) where V_flame_mm is in the same unit as global_points.
        """
        from pytorch3d.transforms import matrix_to_axis_angle

        beta_all = pliks_out['beta']
        n_id, n_exp = flame.n_shape, flame.n_exp
        beta_id  = beta_all[:, :n_id]
        beta_exp = beta_all[:, n_id:n_id + n_exp]

        Rk = pliks_out['Rk']
        J  = flame.J_regressor.shape[0]

        R_world = build_R_world_from_segments(Rk, self.seg_list, J)
        R_rel   = world_to_relative_rotations(R_world, flame.parents)
        aa_all  = matrix_to_axis_angle(R_rel)

        NECK_ID, JAW_ID, LEYE_ID, REYE_ID = 1, 2, 3, 4
        pose_params = torch.zeros(beta_all.shape[0], 3, device=beta_all.device)
        neck_pose_params = aa_all[:, NECK_ID]
        jaw_params = torch.zeros_like(aa_all[:, JAW_ID]) if no_jaw else aa_all[:, JAW_ID]
        eye_pose_params = torch.cat([aa_all[:, LEYE_ID], aa_all[:, REYE_ID]], dim=-1)

        out = flame.forward({
            'shape_params':      beta_id,
            'expression_params': beta_exp,
            'pose_params':       pose_params,
            'neck_pose_params':  neck_pose_params,
            'jaw_params':        jaw_params,
            'eye_pose_params':   eye_pose_params,
        })
        V_flame = out['vertices']

        with torch.cuda.amp.autocast(enabled=False):
            gp = global_points.float() if to_meters else global_points.float() / unit_factor
            R_root, t_root = rigid_align_no_scale(V_flame.float(), gp)

        V_flame_mm = (R_root[:, None] @ V_flame.unsqueeze(-1)).squeeze(-1) + t_root[:, None, :]
        if not to_meters:
            V_flame_mm = V_flame_mm * unit_factor

        losses = {
            'beta_regularizer_pliks':     (beta_id  ** 2).mean() * weight_shape,
            'exp_regularizer_pliks':      (beta_exp ** 2).mean() * weight_expression,
            'pose_regularizer_pliks':     aa_all.pow(2).sum(dim=(1, 2)).mean() * weight_pose,
            'vertices_regularizer_pliks': F.mse_loss(V_flame_mm, global_points) * weight_vertices,
        }
        if edge_loss_fn is not None:
            losses['vertices_regularizer_pliks_edge'] = edge_loss_fn(V_flame_mm, global_points) * weight_vertices_edge

        return losses, V_flame_mm


@torch.no_grad()
def demo_pliks_with_render_and_pose(
    device='cuda',
    img_size=(800, 800),
    out_path='./results/pliks_side_by_side.png',
    out_path_pose='./results/pliks_pose_consistent.png',
    n_shape=300, n_expr=100,
    shape_scale=1.2, expr_scale=1,
    JOINT_MAP=None,  # {'neck':1,'jaw':?,'leye':?,'reye':?}  <-- fill these indices once
):
    """
    - Random FLAME sample (shape+expr; pose=0)
    - Invert with PLIKS (beta,t,Rk), render GT vs recon with error colors
    - Optionally: convert Rk (world) -> relative joint angles; re-forward FLAME incl. pose blends; render again
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_path_pose), exist_ok=True)

    # NO_JAW = False
    from models.FLAME.FLAME import FLAME

    # 1) FLAME & solver
    flame_w_jaw = FLAME(flame_model_path='assets/FLAME2020/generic_model.pkl', no_jaw=False).to(device)
    flame = FLAME(flame_model_path='assets/FLAME2023/flame2023_no_jaw.pkl', no_jaw=True).to(device)

    NO_JAW = True

    fl = flame_w_jaw if not NO_JAW else flame
    # solver = PliksFlameSolver(fl).to(device)  
    solver = PliksFlameSolver(fl, locked_joint_ids=torch.Tensor([2]).long()).to(device)

    faces = fl.faces_tensor.detach().cpu().numpy()
    V = fl.v_template.shape[0]
    J = fl.J_regressor.shape[0]  # # of joints used by LBS

    # 2) Random params (pose zeroed for clarity)
    B = 1
    # seed
    torch.manual_seed(5555)
    shape_params = torch.randn(B, n_shape, device=device) #* shape_scale
    expr_params  = torch.randn(B, n_expr,  device=device) #* expr_scale
    pose_params  = torch.zeros(B, 3,      device=device)
    pose_params[0,0] = 0.0
    pose_params[0,1] = 0.0
    pose_params[0,2] = -0.0
    neck_params  = torch.zeros(B, 3,      device=device)
    neck_params[0,0] = -0.0
    neck_params[0,1] = 0.0
    neck_params[0,2] = 0.0
    jaw_params   = torch.zeros(B, 3,      device=device)
    jaw_params  [0,0] = 0.6
    jaw_params  [0,1] = 0.2
    eye_params   = torch.zeros(B, 6,      device=device)
    trans        = torch.zeros(B, 3,      device=device)
    trans[0,0] = -0.05
    trans[0,1] = +0.02
    trans[0,2] = -0.15


    params = {
        'shape_params':      shape_params,
        'expression_params': expr_params,
        'pose_params':       pose_params,
        'neck_pose_params':  neck_params,
        'jaw_params':        jaw_params,
        'eye_pose_params':   eye_params,
    }

    # 3) Forward FLAME -> GT vertices (pose=0)
    out = flame_w_jaw.forward(params)
    V_gt = out['vertices'] + trans[:, None, :]  # [B,V,3]
    # V_gt = torch.randn_like(V_gt) * 0.001 + V_gt  # add tiny noise to avoid degenerate t
    # V_gt = torch.randn_like(V_gt) * 0.001 + V_gt  # add tiny noise to avoid degenerate t

    # 4) Invert with PLIKS
    inv = solver(V_pred=V_gt, iters=1, lsq_method="ne")
    beta_hat, t_hat, Rk, V_fit = inv['beta'], inv['t'], inv['Rk'], inv['V_fit']
    # print(Rk[0,2],'Rk of locked joint (should be identity)')
    # 5) Error colors
    err = torch.norm((V_fit - V_gt), dim=-1)[0].detach().cpu().numpy()
    col = dist_to_rgb(err, min_dist=0.0, max_dist=np.percentile(err, 99.0))

    # 6) Render GT vs recon (shape/expression recovered; pose ignored by design)
    V_gt_np  = V_gt[0].detach().cpu().numpy()
    V_fit_np = V_fit[0].detach().cpu().numpy()

    # img_left  = render_mesh(V_gt_np, faces, vertex_colors=None, image_size=img_size, needs_projection=False)
    # img_right = render_mesh(V_fit_np, faces, vertex_colors=col,   image_size=img_size, needs_projection=False)
    img_left  = render_mesh_helper(
        Mesh(V_gt_np, faces), t_center=np.array([0,0,0]), rot=np.array([0,0,0]),
        tex_img=None, v_colors=None, errors=None,
        error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0
    )
    img_right = render_mesh_helper(
        Mesh(V_fit_np, faces), t_center=np.array([0,0,0]), rot=np.array([0,0,0]),
        tex_img=None, v_colors=None, errors=None,
        error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0
    )

    stacked = np.hstack([img_left, img_right])
    imageio.imwrite(out_path, stacked)

    # 7) OPTIONAL: pose-consistent re-forward including pose blendshapes
    # Build R_world[J] from segment rotations, then -> relative -> axis-angle
    R_world = build_R_world_from_segments(Rk, solver.seg_list, J)               # [B,J,3,3]
    # print(R_world[0,2],'R_world of locked joint (should be identity)')
    R_rel   = world_to_relative_rotations(R_world, fl.parents)               # [B,J,3,3]
    # print(flame.parents)
    # print(R_rel[0,1],'R_rel of root joint (should be R_world)')
    # print(R_rel[0,2],'R_world of locked joint (should be identity)')
    aa_all  = mat_to_axis_angle(R_rel)                                          # [B,J,3]
    

    # If you give JOINT_MAP once, we’ll pack angles into FLAME’s fields.
    # From your code we know: neck index is 1 (NECK_IDX=1).
    # Fill JAW/LEYE/REYE when you know them.
    # If JOINT_MAP is None or incomplete, we default to zeros (=pose off).
    if JOINT_MAP is None:
        if NO_JAW:
            JOINT_MAP = {
                'neck': 1,
                'leye': 2,
                'reye': 3,
            }
        else:
            JOINT_MAP = {
                'neck': 1,
                'jaw':  2,
                'leye': 3,
                'reye': 4,
            }
    

    # Compose a param dict for a "pose-blend aware" forward using the recovered shape (beta_hat)
    # Split beta_hat back to (shape, expr)
    num_shape = n_shape
    num_expr  = n_expr
    beta_shape = beta_hat[:, :num_shape]
    beta_expr  = beta_hat[:, num_shape:num_shape+num_expr]

    # Pack recovered relative angles if indices are provided, otherwise zeros
    neck_idx = JOINT_MAP.get('neck', 1)
    jaw_idx  = JOINT_MAP.get('jaw',  None)
    # jaw_idx = None
    leye_idx = JOINT_MAP.get('leye', None)
    reye_idx = JOINT_MAP.get('reye', None)
    
    pose_params_rec  = torch.zeros(B, 3, device=device)

    # pose_params_rec  = aa_all[:, 0]               # always root
    neck_pose_rec    = aa_all[:, neck_idx] if neck_idx is not None else torch.zeros(B,3,device=device)
    # print(Rk[:,2])
    pri
    jaw_pose_rec     = aa_all[:, jaw_idx] if jaw_idx  is not None else torch.zeros(B,3,device=device)
    # print(neck_pose_rec, jaw_pose_rec)
    # eyes: 6D (left then right)
    print(jaw_pose_rec, 'jaw')

    if leye_idx is not None and reye_idx is not None:
        eye_pose_rec = torch.cat([aa_all[:, leye_idx], aa_all[:, reye_idx]], dim=-1)  # [B,6]
    else:
        eye_pose_rec = torch.zeros(B, 6, device=device)

    params_recon_pose = {
        'shape_params':      beta_shape,
        'expression_params': beta_expr,
        'pose_params':       pose_params_rec,
        'neck_pose_params':  neck_pose_rec,
        'jaw_params':        jaw_pose_rec,
        'eye_pose_params':   eye_pose_rec,
    }

    out_pose = fl.forward(params_recon_pose)                # includes pose blendshapes
    V_pose  = out_pose['vertices'] # + t_hat[:, None, :]         # add recovered translation

    # Align FLAME output to your target with a single rigid transform
    R_root, t_root = rigid_align_no_scale(V_pose, V_gt)               # <-- external alignment
    V_pose_aligned = (R_root[:,None] @ V_pose.unsqueeze(-1)).squeeze(-1) + t_root[:,None,:]
    V_pose_np = V_pose_aligned[0].detach().cpu().numpy()

    # Color error wrt ground truth
    # err_pose = torch.norm((V_pose - V_gt), dim=-1)[0].detach().cpu().numpy()
    # col_pose = dist_to_rgb(err_pose, min_dist=0.0, max_dist=np.percentile(err_pose, 99.0))

    # img_left2  = render_mesh(V_gt_np,  faces, vertex_colors=None,     image_size=img_size, needs_projection=False)
    # img_right2 = render_mesh(V_pose_np, faces, vertex_colors=col_pose, image_size=img_size, needs_projection=False)
    # img_left2  = render_mesh_helper(
    #     Mesh(V_gt_np, faces), t_center=np.array([0,0,0]), rot=np.array([0,0,0]),
    #     tex_img=None, v_colors=None, errors=None,
    #     error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0
    # )
    img_right2 = render_mesh_helper(
        Mesh(V_pose_np, faces), t_center=np.array([0,0,0]), rot=np.array([0,0,0]),
        tex_img=None, v_colors=None, errors=None,
        error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0
    )

    stacked2 = np.hstack([img_left, img_right, img_right2])
    imageio.imwrite(out_path, stacked2)

    # Diagnostics
    print(f"[PLIKS] beta_hat norm: {beta_hat.norm(dim=1).item():.3f},  t_hat: {t_hat[0].tolist()}")
    print(f"Beta original norm: {(torch.cat([shape_params, expr_params],dim=1)).norm(dim=1).item():.3f}")

    # print(f"[PLIKS] mean |V_fit - V_gt|:  {err.mean():.6f}  (max {err.max():.6f})  -> saved {out_path}")
    # print(f"[POSE ] mean |V_pose - V_gt|: {err_pose.mean():.6f}  (max {err_pose.max():.6f}) -> saved {out_path_pose}")
    # print(f"Segments present (joint ids): {solver.seg_list.tolist()}")
    # print("TIP: set JOINT_MAP={'neck':1,'jaw':<id>,'leye':<id>,'reye':<id>} once you confirm indices.")


import torch
import torch.nn as nn

def optimize_vertices_to_canonical(
    device='cuda',
    steps=500,
    lr=1e-2,
    n_shape=300, n_expr=100,
    lamb_beta=1.0,
    lamb_expr=1.0,
    lamb_pose=0.1,
    lamb_t=0.1,
    lamb_fit=0.5,
    lamb_temp=0.1,
    log_every=50,
):
    """
    Start from random vertices; minimize norms of inferred beta, expr, pose, t.
    Requires PLIKS to be differentiable (no @torch.no_grad() in SVD/lstsq).
    """

    # --- Build FLAME & solver ---
    flame  = FLAME(flame_model_path='assets/FLAME2023/flame2023_no_jaw.pkl', no_jaw=True).to(device)
    solver = PliksFlameSolver(flame).to(device)

    faces = flame.faces_tensor.detach().cpu().numpy()
    V0    = flame.v_template.detach().to(device)             # (V,3)
    J     = flame.J_regressor.shape[0]

    # --- Learnable vertices V ---
    B = 1
    V_param = nn.Parameter(
         torch.randn(1, V0.shape[0], 3, device=device) # small random init around template
    )

    V_param = nn.Parameter(
        V0[None] + 0.02 * torch.randn(1, V0.shape[0], 3, device=device) # small random init around template
    )


    opt = torch.optim.Adam([V_param], lr=lr)

    history = []
    for it in range(1, steps+1):
        opt.zero_grad(set_to_none=True)

        # ----- Invert with differentiable PLIKS -----
        # inv = solver(V_pred=V_param, iters=2)                        # beta,t,Rk,V_fit


        inv = solver(V_pred=V_param, iters=5, lsq_method="qr", estimate_root=False)
        # inv = solver(V_pred=V_param, iters=2, lsq_method="ne")

        beta_hat = inv['beta']                                       # [B, NB=shape+expr]
        t_hat    = inv['t']                                          # [B, 3]
        Rk       = inv['Rk']                                         # [B, K, 3, 3]
        V_fit = inv.get('V_fit_root', inv['V_fit'])

        # split beta -> (shape, expr)
        beta_shape = beta_hat[..., :n_shape]
        beta_expr  = beta_hat[..., n_shape:n_shape+n_expr]

        # recover pose angles (relative) from segment rotations
        R_world = build_R_world_from_segments(Rk, solver.seg_list, J)   # [B,J,3,3]
        R_rel   = world_to_relative_rotations(R_world, flame.parents)   # [B,J,3,3]
        aa_all  = mat_to_axis_angle(R_rel)                               # [B,J,3]

        # ----- Loss -----
        loss_beta  = beta_shape.pow(2).sum(dim=1).mean()
        loss_expr  = beta_expr.pow(2).sum(dim=1).mean()
        loss_pose  = aa_all.pow(2).sum(dim=(1,2)).mean()
        loss_t     = t_hat.pow(2).sum(dim=1).mean()
        loss_fit   = (V_param - V_fit).pow(2).sum(dim=(1,2)).mean()
        loss_temp  = (V_param - V0[None]).pow(2).sum(dim=(1,2)).mean()

        loss = (lamb_beta * loss_beta
               +lamb_expr * loss_expr
               +lamb_pose * loss_pose
               +lamb_t    * loss_t
               +lamb_fit  * loss_fit
               +lamb_temp * loss_temp)

        loss.backward()
        opt.step()

        if it % log_every == 0 or it == 1 or it == steps:
            with torch.no_grad():
                mean_err = (V_param - V_fit).norm(dim=-1).mean().item()
                history.append(dict(
                    it=it,
                    loss=float(loss),
                    beta=float(loss_beta),
                    expr=float(loss_expr),
                    pose=float(loss_pose),
                    t=float(loss_t),
                    fit=float(loss_fit),
                    temp=float(loss_temp),
                    mean_err=mean_err,
                ))
                print(f"[{it:04d}] total={loss:.4e} | "
                      f"β={loss_beta:.3e} expr={loss_expr:.3e} pose={loss_pose:.3e} t={loss_t:.3e} | "
                      f"fit={loss_fit:.3e} temp={loss_temp:.3e} | "
                      f"‖V-V_fit‖mean={mean_err:.4e}")

    # return optimized vertices and last inversion
    return V_param.detach(), inv, history, faces


if __name__ == "__main__":
    demo_pliks_with_render_and_pose(device='cuda', img_size=(800, 800), out_path='./results/pliks_side_by_side.png')
    # V_opt, inv, hist, faces = optimize_vertices_to_canonical(steps=300, lr=5e-3)
    # V_fit = inv['V_fit'][0].detach().cpu().numpy()
    # V_opt_np = V_opt[0].detach().cpu().numpy()

    # img_left  = render_mesh_helper(Mesh(V_opt_np, faces))
    # img_right = render_mesh_helper(Mesh(V_fit, faces))
    # import imageio, os
    # os.makedirs("./results", exist_ok=True)
    # imageio.imwrite("./results/canon_vs_fit.png", np.hstack([img_left, img_right]))
    # print("Saved ./results/canon_vs_fit.png")
