"""Blender-backed refiner visualization helpers.

This module is imported only when --visualization-renderer blender is selected.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.transforms import axis_angle_to_matrix

from utils.mesh_renderer import dist_to_rgb, render_mesh
from utils.point_to_surface_loss import compute_s2m_distance


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class RefinerBlenderVisualizer:
    """Refiner front-row visualizer rendered through Blender."""

    def __init__(self, trainer):
        self.t = trainer

    def _default_script_root(self):
        return os.path.join(_repo_root(), "camera_ready_rendering", "rendering_scripts")

    def _resolve_blender_bin(self):
        raw = getattr(self.t.args, "blender_bin", "") or os.environ.get("BLENDER_BIN", "")
        if not raw:
            local_blender = os.path.expanduser("~/my_blender/blender-3.6.5-linux-x64/blender")
            raw = local_blender if os.path.exists(local_blender) else "blender"

        if os.path.sep in raw or raw.startswith("."):
            if not os.path.exists(raw):
                raise FileNotFoundError(f"Blender binary not found: {raw}")
            return raw

        found = shutil.which(raw)
        if found is None:
            raise FileNotFoundError(f"Blender binary not found on PATH: {raw}")
        return found

    def _resolve_render_asset(self, arg_name, default_name):
        raw = getattr(self.t.args, arg_name, "") or os.environ.get(arg_name.upper(), "")
        path = raw or os.path.join(self._default_script_root(), default_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Blender render asset not found: {path}")
        return path

    @staticmethod
    def _mesh_token_from_label(label):
        label = str(label).lower()
        if "scan" in label:
            return "scan.001"
        if "tto" in label:
            return "mochi_tto"
        if "traditional" in label or "registration" in label or "target" in label:
            return "target.001"
        if "tempeh" in label:
            return "mochi_tempeh"
        if "initial refined" in label or "initial coarse" in label or "mochi" in label:
            return "mochi"
        return "mochi"

    @staticmethod
    def _put_label(img, txt):
        out = img.copy()
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _slug(text):
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_")

    def _load_blender_render(self, path, image_size):
        h, w = image_size
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Missing Blender render: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            bgr = img[:, :, :3].astype(np.float32)
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            white = np.full_like(bgr, 255.0)
            bgr = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        else:
            bgr = img[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return rgb

    def _front_camera(self, image_size):
        t = self.t
        k = t.inputs["camera_intrinsics"][0, 0].detach().cpu().numpy().copy()
        s = 3.0
        k[0, 0] *= s
        k[1, 1] *= s
        k[0, 2] *= s
        k[1, 2] *= s

        extr = t.inputs["camera_extrinsics"][0, 0].detach().cpu().numpy().copy()
        r = axis_angle_to_matrix(torch.tensor([[3.125, 0.0, 0.0]], dtype=torch.float32)).squeeze(0).numpy()
        if extr.shape not in ((3, 4), (4, 4)):
            raise ValueError(f"Unexpected camera extrinsics shape: {extr.shape}")
        extr[:3, :3] = r
        extr[0, 3] = -50.0
        extr[1, 3] = -38.0
        extr[2, 3] = 1350.0
        return k, extr

    def _export_scene(self, vertices_list, faces_list, labels, mesh_path):
        scene = trimesh.Scene()
        mesh_names = []
        verts_faces_np = []

        max_abs_vertex = 0.0
        for verts in vertices_list:
            verts_np = _to_numpy(verts)
            if verts_np.size > 0:
                max_abs_vertex = max(max_abs_vertex, float(np.max(np.abs(verts_np))))

        blender_unit_scale = 0.001 if max_abs_vertex > 10.0 else 1.0

        for idx, (verts, faces, label) in enumerate(zip(vertices_list, faces_list, labels)):
            verts_np = _to_numpy(verts)
            faces_np = _to_numpy(faces)
            if verts_np.ndim != 2 or verts_np.shape[1] != 3:
                raise ValueError(f"Expected verts (V,3), got {verts_np.shape} for label={label}")
            if faces_np.ndim != 2 or faces_np.shape[1] != 3:
                raise ValueError(f"Expected faces (F,3), got {faces_np.shape} for label={label}")

            mesh_name = f"{self._mesh_token_from_label(label)}_{idx:02d}"
            mesh = trimesh.Trimesh(
                vertices=(verts_np * blender_unit_scale).astype(np.float32),
                faces=faces_np.astype(np.int64),
                process=False,
            )
            scene.add_geometry(mesh, node_name=mesh_name)
            mesh_names.append(mesh_name)
            verts_faces_np.append((verts_np, faces_np))

        scene.export(mesh_path)
        return mesh_names, verts_faces_np, blender_unit_scale, max_abs_vertex

    def _save_camera_npz(self, path, k, extr, image_size, blender_unit_scale):
        h, w = image_size
        extr_blender = extr.copy()
        extr_blender[0, 3] *= blender_unit_scale
        extr_blender[1, 3] *= blender_unit_scale
        extr_blender[2, 3] *= blender_unit_scale
        np.savez(
            path,
            K=np.expand_dims(k.astype(np.float64), 0),
            E=np.expand_dims(extr_blender.astype(np.float64), 0),
            H=int(h),
            W=int(w),
        )

    def _run_blender(self, mesh_path, work_dir, cam_npz_path, samples=None, extra_args=None,
                     log_prefix="visualize_multiple_front"):
        blender_bin = self._resolve_blender_bin()
        blender_script = self._resolve_render_asset(
            "blender_render_script", "render_all_views_correct_transform.py"
        )
        blender_scene = self._resolve_render_asset("blender_scene", "render_smaller_shadow.blend")
        samples = int(samples if samples is not None else getattr(self.t.args, "blender_samples", 8))
        device = getattr(self.t.args, "blender_cycles_device", "OPTIX")

        cmd = [
            blender_bin,
            "-b", blender_scene,
            "-P", blender_script,
            "--",
            "--args", mesh_path, work_dir, str(samples), cam_npz_path,
        ]
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        env["BLENDER_CYCLES_DEVICE"] = device
        print(f"[{log_prefix}] Running Blender (samples={samples}, device={device})")
        subprocess.run(cmd, check=True, env=env)

    def _save_composite_cameras_and_refs(self, out_dir, base_name, sample_idx, blender_unit_scale):
        t = self.t
        k_all = _to_numpy(t.inputs["camera_intrinsics"][sample_idx])
        e_all = _to_numpy(t.inputs["camera_extrinsics"][sample_idx])
        d_all = _to_numpy(t.inputs["camera_distortions"][sample_idx])
        images = t.inputs["images"][sample_idx]

        h, w = int(images.shape[-2]), int(images.shape[-1])
        k_new_list = []
        e_list = []
        dataset = getattr(t, "dataset_val", None) or getattr(t, "dataset_train", None)

        for view_idx in range(k_all.shape[0]):
            k = k_all[view_idx].astype(np.float64).copy()
            e = e_all[view_idx].astype(np.float64).copy()
            dist = np.asarray(d_all[view_idx], dtype=np.float64).reshape(-1)

            if e.shape not in ((3, 4), (4, 4)):
                raise ValueError(f"Unexpected camera extrinsics shape: {e.shape}")
            e[:3, 3] *= blender_unit_scale

            if dist.shape[0] < 4:
                dist = np.pad(dist, (0, 4 - dist.shape[0]))

            new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), 1, (w, h))

            img_np = _to_numpy(images[view_idx].permute(1, 2, 0))
            if dataset is not None and hasattr(dataset, "denormalize_image"):
                img_np = dataset.denormalize_image(img_np)
            img_np = np.asarray(img_np)
            if img_np.max() <= 1.5:
                img_np = img_np * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            if img_np.ndim == 2:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR)

            map1, map2 = cv2.initUndistortRectifyMap(k, dist, np.eye(3), new_k, (w, h), cv2.CV_16SC2)
            undistorted_img = cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
            ref_path = os.path.join(out_dir, f"{base_name}_ref_view_{view_idx:02d}.png")
            if not cv2.imwrite(ref_path, undistorted_img):
                raise RuntimeError(f"Failed to write reference image: {ref_path}")

            k_new_list.append(new_k)
            e_list.append(e)

        cam_npz_path = os.path.join(out_dir, f"{base_name}_cameras.npz")
        np.savez(
            cam_npz_path,
            K=np.stack(k_new_list).astype(np.float64),
            E=np.stack(e_list).astype(np.float64),
            H=h,
            W=w,
        )
        return cam_npz_path

    def _run_composite(self, out_dir, base_name):
        composite_script = self._resolve_render_asset("blender_composite_script", "make_composite.py")
        layout = getattr(self.t.args, "render_half_sides_overlay_layout", "v2")
        cmd = [sys.executable, composite_script, out_dir, base_name, layout]
        print(f"[half_sides_overlay] Running composite layout={layout}")
        subprocess.run(cmd, check=True)

    def multi_view_half_sides_overlay(self, vertices_list, faces_list, labels, sample_idx=0,
                                      dataset_idx=None):
        t = self.t
        assert len(vertices_list) == len(faces_list), "vertices_list and faces_list must have same length."

        labels = list(labels) if labels is not None else []
        while len(labels) < len(vertices_list):
            labels.append(f"Mesh {len(labels)}")
        labels = labels[:len(vertices_list)]

        out_root = os.path.join(t.directory_output, "val_images", "half_sides_overlay")
        os.makedirs(out_root, exist_ok=True)
        if dataset_idx is None:
            base_name = f"half_sides_{t.global_step:06d}"
        else:
            base_name = f"half_sides_{t.global_step:06d}_idx_{int(dataset_idx):05d}"
        out_dir = os.path.join(out_root, base_name)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        mesh_path = os.path.join(out_dir, f"{base_name}.glb")
        mesh_names, _, unit_scale, max_abs_vertex = self._export_scene(
            vertices_list, faces_list, labels, mesh_path
        )
        print(
            f"[half_sides_overlay] Exported {len(mesh_names)} meshes with unit scale={unit_scale} "
            f"(max_abs_vertex={max_abs_vertex:.3f})"
        )
        cam_npz_path = self._save_composite_cameras_and_refs(
            out_dir, base_name, sample_idx, unit_scale
        )

        samples = int(getattr(t.args, "render_half_sides_overlay_samples", 64))
        self._run_blender(
            mesh_path,
            out_dir,
            cam_npz_path,
            samples=samples,
            extra_args=["--outer-surface-only"],
            log_prefix="half_sides_overlay",
        )

        half_meshes = getattr(t.args, "render_half_sides_overlay_meshes", "mochi")
        extra_hide_path = (
            getattr(t.args, "render_half_sides_overlay_extra_hide_indices_by_view_path", "")
            or os.path.join(self._default_script_root(), "extra_hide_indices_by_view.txt")
        )
        half_args = ["--half-face", "--outer-surface-only"]
        if half_meshes:
            half_args.extend(["--half-face-meshes", half_meshes])
        if extra_hide_path:
            if not os.path.exists(extra_hide_path):
                raise FileNotFoundError(f"Half-side extra-hide file not found: {extra_hide_path}")
            half_args.extend(["--half-face-extra-hide-indices-by-view-path", extra_hide_path])

        self._run_blender(
            mesh_path,
            out_dir,
            cam_npz_path,
            samples=samples,
            extra_args=half_args,
            log_prefix="half_sides_overlay",
        )
        self._run_composite(out_dir, base_name)
        print(f"[half_sides_overlay] Saved composite assets in: {out_dir}")

    def multi_front(self, vertices_list, faces_list, labels, out_path=None, sample_idx=0,
                    dataset_idx=None, image_size=(800, 800), max_err_mm=3.0, show_labels=True):
        t = self.t
        assert len(vertices_list) == len(faces_list), "vertices_list and faces_list must have same length."

        labels = list(labels) if labels is not None else []
        while len(labels) < len(vertices_list):
            labels.append(f"Mesh {len(labels)}")
        labels = labels[:len(vertices_list)]

        out_dir = os.path.join(t.directory_output, "val_images")
        os.makedirs(out_dir, exist_ok=True)
        if out_path is None:
            if dataset_idx is None:
                out_path = os.path.join(out_dir, f"front_row_{t.global_step:06d}.jpg")
            else:
                out_path = os.path.join(out_dir, f"front_row_idx_{int(dataset_idx):05d}.jpg")

        work_dir = tempfile.mkdtemp(prefix=f"front_blender_{t.global_step:06d}_", dir=out_dir)
        keep_workdir = bool(getattr(t.args, "blender_keep_workdir", False))

        try:
            mesh_path = os.path.join(work_dir, "front_row_scene.glb")
            cam_npz_path = os.path.join(work_dir, "front_camera.npz")
            k, extr = self._front_camera(image_size)
            mesh_names, verts_faces_np, unit_scale, max_abs_vertex = self._export_scene(
                vertices_list, faces_list, labels, mesh_path
            )
            print(
                f"[visualize_multiple_front] Blender unit scale={unit_scale} "
                f"(max_abs_vertex={max_abs_vertex:.3f})"
            )
            self._save_camera_npz(cam_npz_path, k, extr, image_size, unit_scale)
            self._run_blender(mesh_path, work_dir, cam_npz_path)

            scan_v = t.data["v_scan"][sample_idx].detach().cpu()
            scan_f = t.data["f_scan"][sample_idx].detach().cpu()
            scan_v_np = scan_v.numpy()
            scan_f_np = scan_f.numpy()

            tiles = []
            for idx, ((verts_np, faces_np), mesh_name) in enumerate(zip(verts_faces_np, mesh_names)):
                gray_path = os.path.join(work_dir, f"renders_{mesh_name}", "view_00.png")
                gray_img = self._load_blender_render(gray_path, image_size)
                label = labels[idx]

                if "scan" in str(label).lower():
                    tile = self._put_label(gray_img, f"Mesh {label} gray") if show_labels else gray_img
                    tiles.append(tile)
                    continue

                with torch.no_grad():
                    pred_v = torch.as_tensor(verts_np, dtype=torch.float32, device=t.device).unsqueeze(0)
                    pred_f = torch.as_tensor(faces_np, dtype=torch.int64, device=t.device)
                    scan_v_t = scan_v.unsqueeze(0).to(t.device)
                    dist = compute_s2m_distance(scan_v_t, pred_v, pred_f, masks=t.flame_masks_triangles)["full"]
                    err_rgb = dist_to_rgb(dist.detach().cpu().numpy(), min_dist=0.0, max_dist=max_err_mm)

                err_img = render_mesh(
                    vertices=scan_v_np, faces=scan_f_np, vertex_colors=err_rgb,
                    camera_extrinsics=extr, camera_intrinsics=k,
                    radial_distortion=None, image_size=image_size, needs_projection=True,
                )

                if show_labels:
                    gray_img = self._put_label(gray_img, f"Mesh {label} gray")
                    err_img = self._put_label(err_img, "Scan error")
                tiles.append(np.hstack([gray_img, err_img]))

            if not tiles:
                raise ValueError("No tiles rendered in visualize_multiple_front.")

            row = np.hstack(tiles)
            cv2.imwrite(out_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
            print(f"[visualize_multiple_front] Saved: {out_path}")
        finally:
            if keep_workdir:
                print(f"[visualize_multiple_front] Kept Blender work dir: {work_dir}")
            else:
                shutil.rmtree(work_dir, ignore_errors=True)
