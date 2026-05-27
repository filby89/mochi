"""Per-step visualization helpers for the global trainer and refiner."""

import math
import os

import cv2
import numpy as np
import torch
import wandb
from pytorch3d.transforms import axis_angle_to_matrix

from trainer.utils import draw_dense_points, get_dense_vertex_colors
from utils.mesh_helper import pointmap_to_rgb
from utils.mesh_renderer import dist_to_rgb, render_mesh
from utils.point_to_surface_loss import compute_s2m_distance
from utils.utils import add_labels_to_images, to_numpy


def _depth_to_rgb(depth_np):
    d = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-12)
    d = (255 * d).astype(np.uint8)
    return np.stack((d, d, d), axis=-1)


def _loss_heatmap(loss_per_pixel_np, max_dist=3.0):
    h, w = loss_per_pixel_np.shape
    return dist_to_rgb(loss_per_pixel_np.reshape(-1), min_dist=0.0, max_dist=max_dist).reshape(h, w, 3)


def _render_or_none(vertices, faces, vertex_colors, camera_args, needs_projection=False):
    if vertices is None:
        return None
    return render_mesh(
        vertices=vertices, faces=faces, vertex_colors=vertex_colors,
        needs_projection=needs_projection, **camera_args,
    )


class GlobalTrainerVisualizer:
    """Owns the per-step visualization for the global trainer.

    Reads state directly off the trainer reference rather than copying.
    """

    def __init__(self, trainer):
        self.t = trainer

    def run(self, mode='train', val_idx=0):
        key = "_augmented" if mode == "train" else ""
        with torch.no_grad():
            for idx in [0]:  # only sample 0
                batch = self._batch_state(idx)
                for view_id in self.t.visualization_view_ids:
                    input_image, camera_args = self._load_view(idx, view_id, mode, key)
                    vis_image, vis_image_pred, dense_panels = self._landmark_overlays(input_image, idx, view_id)
                    mesh_panels = self._mesh_panels(batch, camera_args)
                    diff_panels = self._diff_panels(idx, view_id) if self.t.args.enable_diff_rendering else {}

                    # Panel order matches the original layout (private repo).
                    panels = {'Input': vis_image}
                    panels.update(dense_panels)
                    for k in ('Target Scan', 'Target', 'Base Recon', 'Base Local Recon', 'Recon',
                              'Error', 'Base Error', 'Base Local Error'):
                        panels[k] = mesh_panels.get(k)
                    panels.update(diff_panels)
                    for k in ('PLIKS Recon', 'PLIKS Error'):
                        panels[k] = mesh_panels.get(k)
                    panels['Input Image with Landmarks'] = vis_image
                    panels['Input Image with Predicted Landmarks'] = vis_image_pred
                    self._save_grid(panels, mode, view_id, idx, val_idx)

    # --- per-batch state ---------------------------------------------------

    def _batch_state(self, idx):
        """Run base/base_local models, gather reconstructions and s2m vertex colors."""
        t = self.t

        # Keep these as torch tensors so we can pass them straight to compute_s2m_distance
        # without round-tripping through numpy.
        recon_base_t = None
        recon_base_local_t = None
        if t.args.enable_local:
            t.global_points_base = t.coarse_results['vertices']
            recon_base_t = t.global_points_base[idx]
            if hasattr(t, 'base_local_model'):
                t.base_local_model.eval()
                t.base_local_results = t.base_local_model(
                    t.inputs['images'], t.inputs['camera_intrinsics'], t.inputs['camera_extrinsics'],
                    camera_distortions=t.inputs['camera_distortions'],
                    camera_centers=t.camera_centers,
                    global_points=t.global_points_base, random_grid=False,
                )
                t.global_points_base_local = t.base_local_results[-1]
                recon_base_local_t = t.global_points_base_local[idx]
        elif hasattr(t, 'base_model'):
            t.base_model.eval()
            t.base_results = t.base_model(
                t.inputs['images'], t.inputs['camera_intrinsics'], t.inputs['camera_extrinsics'],
                camera_distortions=t.inputs['camera_distortions'],
            )
            t.global_points_base = t.base_results['vertices']
            recon_base_t = t.global_points_base[idx]

        recon_main_t = t.global_points[idx]
        recon_pliks_t = t.pliks_out['V_fit'][idx] if hasattr(t, 'pliks_out') and t.pliks_out is not None else None

        faces_pred = to_numpy(t.faces)
        faces_target = t.data.get('f_registration', None)
        if faces_target is not None:
            if torch.is_tensor(faces_target):
                faces_target = to_numpy(faces_target[idx] if faces_target.ndim == 3 else faces_target)
            elif isinstance(faces_target, list):
                faces_target = to_numpy(faces_target[idx])
        if faces_target is None:
            faces_target = faces_pred

        # s2m vertex colors — match the original: predicted_faces as a float Tensor.
        # (Kaolin's index_vertices_by_faces is dtype-sensitive here.)
        scan_v_dev = t.data['v_scan'][idx].to(t.device).unsqueeze(0)
        predicted_faces = torch.Tensor(t.faces).to(t.device)

        def colors(vertices_t):
            if vertices_t is None:
                return None
            v = vertices_t.unsqueeze(0).float()
            dist = compute_s2m_distance(scan_v_dev, v, predicted_faces, masks=t.flame_masks_triangles)['full']
            return dist_to_rgb(dist.detach().cpu().numpy(), min_dist=0.0, max_dist=3.0)

        return {
            'target_vertices': to_numpy(t.target_vertices[idx]),
            'recon_main':       to_numpy(recon_main_t),
            'recon_base':       to_numpy(recon_base_t) if recon_base_t is not None else None,
            'recon_base_local': to_numpy(recon_base_local_t) if recon_base_local_t is not None else None,
            'recon_pliks':      to_numpy(recon_pliks_t) if recon_pliks_t is not None else None,
            'faces_pred':       faces_pred,
            'faces_target':     faces_target,
            'scan_vertices':    t.data['v_scan'][idx],
            'scan_faces':       t.data['f_scan'][idx] if 'f_scan' in t.data else None,
            'colors_main':       colors(recon_main_t),
            'colors_base':       colors(recon_base_t),
            'colors_base_local': colors(recon_base_local_t),
            'colors_pliks':      colors(recon_pliks_t),
        }

    # --- per-view loaders and panels --------------------------------------

    def _load_view(self, idx, view_id, mode, key):
        t = self.t
        input_image = to_numpy(t.data[f'color_images{key}'][idx][view_id].permute(1, 2, 0))
        ds = t.dataset_train if mode == 'train' else t.dataset_val
        input_image = (255 * ds.denormalize_image(input_image)).astype(np.uint8)

        camera_args = {
            'camera_intrinsics': to_numpy(t.data[f'color_camera_intrinsics{key}'][idx][view_id]),
            'camera_extrinsics': to_numpy(t.data['color_camera_extrinsics'][idx][view_id]),
            'radial_distortion': to_numpy(t.data['color_camera_distortions'][idx][view_id]),
            'frustum': {'near': 0.01, 'far': 3000.0},
            'image_size': input_image.shape[:2],
        }
        return input_image, camera_args

    def _mesh_panels(self, b, camera_args):
        scan_args = (b['scan_vertices'], b['scan_faces'])
        return {
            'Target Scan': render_mesh(vertices=b['scan_vertices'], faces=b['scan_faces'], vertex_colors=None, **camera_args),
            'Target': render_mesh(vertices=b['target_vertices'], faces=b['faces_target'], vertex_colors=None, **camera_args),
            'Base Recon': _render_or_none(b['recon_base'], b['faces_pred'], None, camera_args, needs_projection=True),
            'Base Local Recon': _render_or_none(b['recon_base_local'], b['faces_pred'], None, camera_args, needs_projection=True),
            'Recon': _render_or_none(b['recon_main'], b['faces_pred'], None, camera_args, needs_projection=True),
            'Error': _render_or_none(*scan_args, b['colors_main'], camera_args, needs_projection=True) if b['colors_main'] is not None else None,
            'Base Error': _render_or_none(*scan_args, b['colors_base'], camera_args, needs_projection=True) if b['colors_base'] is not None else None,
            'Base Local Error': _render_or_none(*scan_args, b['colors_base_local'], camera_args, needs_projection=True) if b['colors_base_local'] is not None else None,
            'PLIKS Recon': _render_or_none(b['recon_pliks'], b['faces_pred'], None, camera_args, needs_projection=True),
            'PLIKS Error': _render_or_none(*scan_args, b['colors_pliks'], camera_args, needs_projection=True) if b['colors_pliks'] is not None else None,
        }

    def _diff_panels(self, idx, view_id):
        """Diff-rendering panels, interleaved Pred → GT → Loss per modality (matches original layout)."""
        t = self.t
        if not (hasattr(t, 'normal_maps_pred') and t.normal_maps_pred is not None):
            return {}

        # Compute GT first so its pointmap min/max can be shared with the predicted one.
        gt_normals = (255 * to_numpy(t.normal_maps_gt_01[idx][view_id].permute(1, 2, 0))).astype(np.uint8)
        depth_gt = _depth_to_rgb(t.depth_maps_gt[idx][view_id].cpu().numpy().squeeze(-1))
        point_maps_gt, _min, _max = pointmap_to_rgb(t.pointmaps_gt[idx][view_id].cpu().numpy())

        pred_normals = np.clip((255 * to_numpy(t.normal_maps_pred[idx][view_id].permute(1, 2, 0))).astype(np.uint8), 0, 255)
        depth_vis = _depth_to_rgb(t.depth_maps_pred[idx][view_id].cpu().numpy())
        point_maps_pred, _, _ = pointmap_to_rgb(t.pointmaps_pred[idx][view_id].cpu().numpy(), _min, _max)

        panels = {
            'Pred Normals': pred_normals,
            'GT Normals':   gt_normals,
            'Normals Loss': _loss_heatmap(t.normals_loss_per_pixel[idx][view_id].cpu().numpy()),
            'Pred Depth':   depth_vis,
            'GT Depth':     depth_gt,
        }
        if hasattr(t, 'depth_maps_loss_per_pixel'):
            panels['Depth Loss'] = _loss_heatmap(t.depth_maps_loss_per_pixel[idx][view_id].cpu().numpy())
        panels['Pred PointMap'] = point_maps_pred
        panels['GT PointMap']   = point_maps_gt
        panels['PointMap Loss'] = _loss_heatmap(t.point_maps_loss_per_pixel[idx][view_id].cpu().numpy())
        return panels

    def _landmark_overlays(self, input_image, idx, view_id):
        """Build the mediapipe-dot overlays and dense-landmark panels.

        Returns:
            (vis_image, vis_image_pred, dense_panels_dict)
            - vis_image: input with GT mediapipe dots (also reused as the 'Input' panel)
            - vis_image_pred: input with predicted mediapipe dots
            - dense_panels_dict: ordered 'Dense GT'/'Dense Pred'/'Dense Pred Out' panels
        """
        t = self.t
        vis_image = input_image.copy()
        vis_image_pred = input_image.copy()

        for x, y, *_ in to_numpy(t.landmarks_dense_mediapipe[idx][view_id]):
            cv2.circle(vis_image, (int(x), int(y)), 2, (255, 0, 255), -1)

        if getattr(t, 'global_landmarks_projected_dense_mediapipe_out', None) is not None:
            for x, y, *_ in to_numpy(t.global_landmarks_projected_dense_mediapipe_out[idx][view_id]):
                cv2.circle(vis_image_pred, (int(x), int(y)), 2, (255, 0, 0), -1)

        dense_panels = {}
        dense_gt = to_numpy(t.landmarks_dense[idx][view_id]) if getattr(t, 'landmarks_dense', None) is not None else None
        view_mask = (
            float(to_numpy(t.landmarks_dense_mask[idx][view_id]).item())
            if getattr(t, 'landmarks_dense_mask', None) is not None else None
        )
        if dense_gt is not None and dense_gt.shape[0] > 0 and (view_mask is None or view_mask > 0):
            colors = get_dense_vertex_colors(dense_gt.shape[0], t.flame_masks)
            gt_bgr = cv2.cvtColor(input_image.copy(), cv2.COLOR_RGB2BGR)
            dense_panels['Dense GT'] = cv2.cvtColor(draw_dense_points(gt_bgr.copy(), dense_gt, colors), cv2.COLOR_BGR2RGB)
            if getattr(t, 'global_landmarks_projected_dense', None) is not None:
                pred = to_numpy(t.global_landmarks_projected_dense[idx][view_id])
                dense_panels['Dense Pred'] = cv2.cvtColor(draw_dense_points(gt_bgr.copy(), pred, colors), cv2.COLOR_BGR2RGB)
            if getattr(t, 'global_landmarks_projected_dense_out', None) is not None:
                pred = to_numpy(t.global_landmarks_projected_dense_out[idx][view_id])
                dense_panels['Dense Pred Out'] = cv2.cvtColor(draw_dense_points(gt_bgr.copy(), pred, colors), cv2.COLOR_BGR2RGB)

        return vis_image, vis_image_pred, dense_panels

    # --- save -------------------------------------------------------------

    def _save_grid(self, panels, mode, view_id, idx, val_idx):
        t = self.t
        pairs = [(img, label) for label, img in panels.items() if img is not None]
        images, labels = map(list, zip(*pairs))
        labeled_images = add_labels_to_images(images, labels)

        nrows = 4
        ncols = math.ceil(len(labeled_images) / nrows)
        h, w, c = labeled_images[0].shape
        while len(labeled_images) < nrows * ncols:
            labeled_images.append(np.full((h, w, c), 255, dtype=np.uint8))

        rows = [np.hstack(labeled_images[i * ncols:(i + 1) * ncols]) for i in range(nrows)]
        visualization = np.vstack(rows)

        out_dir = os.path.join(t.directory_output, mode + '_images')
        os.makedirs(out_dir, exist_ok=True)
        suffix = f'_{idx}_{val_idx}' if mode in ('val', 'warm') else f'_{idx}'
        out_path = os.path.join(out_dir, f'view_id_{view_id:02d}_{t.global_step}{suffix}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        if t.args.wandb and idx == 0 and (t.global_step % 500 == 0):
            wandb.log({f'{mode.capitalize()}/view_id_{view_id:02d}': wandb.Image(visualization)}, step=t.global_step)


class RefinerVisualizer:
    """Refiner-specific visualizations (front-row mesh comparisons + a color grid)."""

    def __init__(self, trainer):
        self.t = trainer

    def multi_front(self, vertices_list, faces_list, labels, out_path=None, sample_idx=0,
                    dataset_idx=None, image_size=(800, 800), max_err_mm=3.0):
        """Render a row of (gray mesh | scan error) tiles for each (vertices, faces) pair."""
        t = self.t
        assert len(vertices_list) == len(faces_list), "vertices_list and faces_list must have same length."

        # Canonical front-on camera derived from the current batch's intrinsics
        K = t.inputs['camera_intrinsics'][0, 0].detach().cpu().numpy().copy()
        s = 3.0
        K[0, 0] *= s; K[1, 1] *= s
        K[0, 2] *= s; K[1, 2] *= s

        Extr = t.inputs['camera_extrinsics'][0, 0].detach().cpu().numpy().copy()
        R = axis_angle_to_matrix(torch.tensor([[3.125, 0.0, 0.0]], dtype=torch.float32)).squeeze(0).numpy()
        Extr[:3, :3] = R
        Extr[0, 3] = -50.0
        Extr[1, 3] = -38.0
        Extr[2, 3] = 1350.0

        scan_v = t.data['v_scan'][sample_idx].detach().cpu()
        scan_f = t.data['f_scan'][sample_idx].detach().cpu()
        scan_v_np = scan_v.numpy()
        scan_f_np = scan_f.numpy()

        def put_label(img, txt):
            out = img.copy()
            cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            return out

        tiles = []
        for i, (V_np, F_np) in enumerate(zip(vertices_list, faces_list)):
            gray_img = render_mesh(
                vertices=V_np, faces=F_np, vertex_colors=None,
                camera_extrinsics=Extr, camera_intrinsics=K,
                radial_distortion=None, image_size=image_size, needs_projection=True,
            )

            with torch.no_grad():
                pred_v = torch.as_tensor(V_np, dtype=torch.float32, device=t.device).unsqueeze(0)
                pred_f = torch.as_tensor(F_np, dtype=torch.int64, device=t.device)
                scan_v_t = scan_v.unsqueeze(0).to(t.device)
                dist = compute_s2m_distance(scan_v_t, pred_v, pred_f, masks=t.flame_masks_triangles)['full']
                err_rgb = dist_to_rgb(dist.detach().cpu().numpy(), min_dist=0.0, max_dist=1.0)

            err_img = render_mesh(
                vertices=scan_v_np, faces=scan_f_np, vertex_colors=err_rgb,
                camera_extrinsics=Extr, camera_intrinsics=K,
                radial_distortion=None, image_size=image_size, needs_projection=True,
            )

            tiles.append(np.hstack([put_label(gray_img, f"Mesh {labels[i]} gray"), put_label(err_img, "Scan error")]))

        if not tiles:
            return

        row = np.hstack(tiles)
        os.makedirs(os.path.join(t.directory_output, 'val_images'), exist_ok=True)
        if out_path is None:
            if dataset_idx is None:
                out_path = os.path.join(t.directory_output, 'val_images', f'front_row_{t.global_step:06d}.jpg')
            else:
                out_path = os.path.join(t.directory_output, 'val_images', f'front_row_idx_{int(dataset_idx):05d}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f'[visualize_multiple_front] Saved: {out_path}')
