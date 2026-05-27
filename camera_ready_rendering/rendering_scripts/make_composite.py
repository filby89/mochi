import sys
import os
import cv2
import glob
import numpy as np

def overlay_rgba_on_bgr(bg_bgr, fg_bgra):
    if fg_bgra.ndim != 3:
        raise ValueError("Render image must be HxWxC")
    if fg_bgra.shape[2] == 3:
        return fg_bgra
    if fg_bgra.shape[2] != 4:
        raise ValueError("Render image must have 3 or 4 channels")
    fg_bgr = fg_bgra[:, :, :3].astype("float32")
    alpha = fg_bgra[:, :, 3:4].astype("float32") / 255.0
    bg = bg_bgr.astype("float32")
    out = fg_bgr * alpha + bg * (1.0 - alpha)
    out[out < 0] = 0
    out[out > 255] = 255
    return out.astype("uint8")

def make_focus_grid(overlay_dir, output_dir):
    view_paths = []
    for v in range(6):
        path = os.path.join(overlay_dir, f"view_{v:02d}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing view image for grid: {path}")
        view_paths.append(path)

    views = []
    for path in view_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image for grid: {path}")
        views.append(img)

    left_full = views[0]
    full_h, full_w = left_full.shape[:2]
    x0 = full_w // 5
    x1 = full_w - (full_w // 5)
    left = left_full[:, x0:x1, :]
    if left.shape[1] == 0:
        raise ValueError(f"Invalid crop for grid main view in {overlay_dir}")
    left_h, left_w = left.shape[:2]

    tile_heights = [left_h // 5] * 5
    tile_heights[-1] += left_h - sum(tile_heights)
    tile_widths = []
    for idx in range(1, 6):
        src_h, src_w = views[idx].shape[:2]
        if src_h == 0:
            raise ValueError(f"Invalid tile height for grid view {idx:02d} in {overlay_dir}")
        tile_w = max(1, int(round(tile_heights[idx - 1] * float(src_w) / float(src_h))))
        tile_widths.append(tile_w)
    right_w = max(tile_widths)

    right_col = np.zeros((left_h, right_w, 3), dtype=np.uint8)
    y = 0
    for idx in range(1, 6):
        tile_h = tile_heights[idx - 1]
        tile_w = tile_widths[idx - 1]
        tile = cv2.resize(views[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        x = (right_w - tile_w) // 2
        right_col[y:y + tile_h, x:x + tile_w, :] = tile
        y += tile_h

    grid = np.concatenate([left, right_col], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "grid_view0_plus_1to5.png")
    if not cv2.imwrite(out_path, grid):
        raise RuntimeError(f"Failed to write grid image: {out_path}")
    print(f"Saved grid: {out_path}")

def make_focus_grid_v2(overlay_dir, output_dir):
    required_views = [0, 1, 4, 2, 5]
    view_paths = []
    for v in required_views:
        path = os.path.join(overlay_dir, f"view_{v:02d}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing view image for grid v2: {path}")
        view_paths.append(path)

    views = []
    for path in view_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image for grid v2: {path}")
        views.append(img)

    left_full = views[0]
    full_h, full_w = left_full.shape[:2]
    x0 = full_w // 5
    x1 = full_w - (full_w // 5)
    left = left_full[:, x0:x1, :]
    if left.shape[1] == 0:
        raise ValueError(f"Invalid crop for grid v2 main view in {overlay_dir}")
    left_h, left_w = left.shape[:2]

    top_h = left_h // 2
    bottom_h = left_h - top_h

    right_cols = []
    for top_idx, bottom_idx in ((1, 2), (3, 4)):
        top_src = views[top_idx]
        bottom_src = views[bottom_idx]

        top_w = max(1, int(round(top_h * float(top_src.shape[1]) / float(top_src.shape[0]))))
        bottom_w = max(1, int(round(bottom_h * float(bottom_src.shape[1]) / float(bottom_src.shape[0]))))
        col_w = max(top_w, bottom_w)

        col = np.zeros((left_h, col_w, 3), dtype=np.uint8)
        top_tile = cv2.resize(top_src, (top_w, top_h), interpolation=cv2.INTER_AREA)
        bottom_tile = cv2.resize(bottom_src, (bottom_w, bottom_h), interpolation=cv2.INTER_AREA)

        top_x = (col_w - top_w) // 2
        bottom_x = (col_w - bottom_w) // 2
        col[0:top_h, top_x:top_x + top_w, :] = top_tile
        col[top_h:left_h, bottom_x:bottom_x + bottom_w, :] = bottom_tile
        right_cols.append(col)

    grid = np.concatenate([left] + right_cols, axis=1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "grid_view0_plus_12_45_2x2.png")
    if not cv2.imwrite(out_path, grid):
        raise RuntimeError(f"Failed to write grid image: {out_path}")
    print(f"Saved grid v2: {out_path}")

def make_composite(output_dir, mesh_basename, grid_layout="v1"):
    render_dirs = sorted(glob.glob(os.path.join(output_dir, "renders_*")))
    if not render_dirs:
        raise FileNotFoundError(f"No render directories found in {output_dir} (expected renders_*)")

    methods = []
    for render_dir in render_dirs:
        if not os.path.isdir(render_dir):
            continue
        method_name = os.path.basename(render_dir).replace("renders_", "", 1)
        if not method_name:
            raise ValueError(f"Invalid render directory name: {render_dir}")
        methods.append((method_name, render_dir))

    if not methods:
        raise FileNotFoundError(f"No valid render directories found in {output_dir} (expected renders_*)")

    ref_paths = sorted(glob.glob(os.path.join(output_dir, f"{mesh_basename}_ref_view_*.png")))
    if not ref_paths:
        raise FileNotFoundError(f"No ref images found for {mesh_basename} in {output_dir}")

    for ref_path in ref_paths:
        ref_name = os.path.basename(ref_path)
        view_str = ref_name.replace(f"{mesh_basename}_ref_view_", "").replace(".png", "")
        view_id = int(view_str)

        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref_img is None:
            raise ValueError(f"Failed to read ref image: {ref_path}")
        ref_is_flat = bool(np.max(ref_img) == np.min(ref_img))
        if ref_is_flat:
            print(f"Ref image is flat for view {view_id:02d}; using render-only background for overlays")

        original_dir = os.path.join(output_dir, "overlays_original")
        os.makedirs(original_dir, exist_ok=True)
        out_original = os.path.join(original_dir, f"view_{view_id:02d}.png")
        if not cv2.imwrite(out_original, ref_img):
            raise RuntimeError(f"Failed to write image: {out_original}")

        for method_name, render_dir in methods:
            render_path = os.path.join(render_dir, f"view_{view_id:02d}.png")
            if not os.path.exists(render_path):
                raise FileNotFoundError(f"Missing render image for method '{method_name}': {render_path}")

            render_img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
            if render_img is None:
                raise ValueError(f"Failed to read render image: {render_path}")
            if render_img.shape[:2] != ref_img.shape[:2]:
                render_img = cv2.resize(render_img, (ref_img.shape[1], ref_img.shape[0]))

            if ref_is_flat:
                black_bg = np.zeros_like(ref_img)
                overlay = overlay_rgba_on_bgr(black_bg, render_img)
            else:
                overlay = overlay_rgba_on_bgr(ref_img, render_img)
            overlay_dir = os.path.join(output_dir, f"overlays_{method_name}")
            os.makedirs(overlay_dir, exist_ok=True)
            out_overlay = os.path.join(overlay_dir, f"view_{view_id:02d}.png")
            if not cv2.imwrite(out_overlay, overlay):
                raise RuntimeError(f"Failed to write image: {out_overlay}")

        print(f"Saved view {view_id:02d}: original + {len(methods)} method overlays")

    overlay_dirs = sorted(glob.glob(os.path.join(output_dir, "overlays_*")))
    if not overlay_dirs:
        raise FileNotFoundError(f"No overlay directories found in {output_dir} (expected overlays_*)")

    for overlay_dir in overlay_dirs:
        method_name = os.path.basename(overlay_dir).replace("overlays_", "", 1)
        if not method_name:
            raise ValueError(f"Invalid overlay directory name: {overlay_dir}")
        grid_dir = os.path.join(output_dir, f"grid_{method_name}")
        if grid_layout == "v1":
            make_focus_grid(overlay_dir, grid_dir)
        elif grid_layout == "v2":
            make_focus_grid_v2(overlay_dir, grid_dir)
        else:
            raise ValueError(f"Unknown grid layout: {grid_layout}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python make_composite.py <output_dir> <mesh_basename> [v1|v2]")
    layout = "v1"
    if len(sys.argv) >= 4:
        layout = sys.argv[3]
    make_composite(sys.argv[1], sys.argv[2], layout)
