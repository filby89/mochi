import bpy
import numpy as np
import mathutils
import bmesh
import sys
import os
from bisect import bisect_left

argv = sys.argv
try:
    args_idx = argv.index("--args")
    argv = argv[args_idx + 1:] 
except ValueError:
    print("Error: No arguments passed.")
    sys.exit(1)

WORLD_FIX = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0,-1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)  # -90deg about X (Blender Z-up -> dataset Y-up)

def parse_indices_spec(raw):
    parsed = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if "-" in t:
            bounds = t.split("-", 1)
            start = int(bounds[0].strip())
            end = int(bounds[1].strip())
            if end < start:
                raise ValueError(f"Invalid index range: {t}")
            parsed.extend(list(range(start, end + 1)))
        else:
            parsed.append(int(t))
    return parsed


def parse_args(args):
    if len(args) < 4:
        raise ValueError("Expected at least 4 args: mesh_path, output_dir, samples, cam_npz_path")
    mesh_path = args[0]
    output_base_dir = args[1]
    samples = int(args[2])
    cam_npz_path = args[3]

    opts = {
        "half_face": False,
        "half_face_indices_path": None,
        "half_face_var": "right_half",
        "half_face_meshes": None,
        "half_face_extra_hide_views": None,
        "half_face_extra_hide_indices": None,
        "half_face_extra_hide_indices_path": None,
        "half_face_extra_hide_indices_by_view_path": None,
        "half_face_extra_hide_indices_var": "extra_hide_indices",
        "render_subdir": "renders",
        "outer_surface_only": False,
    }

    i = 4
    while i < len(args):
        arg = args[i]
        if arg == "--":
            break
        if arg == "--half-face":
            opts["half_face"] = True
            i += 1
            continue
        if arg == "--half-face-indices" and i + 1 < len(args):
            opts["half_face_indices_path"] = args[i + 1]
            i += 2
            continue
        if arg == "--half-face-var" and i + 1 < len(args):
            opts["half_face_var"] = args[i + 1]
            i += 2
            continue
        if arg == "--half-face-meshes" and i + 1 < len(args):
            raw = args[i + 1]
            opts["half_face_meshes"] = [s.strip().lower() for s in raw.split(",") if s.strip()]
            i += 2
            continue
        if arg == "--half-face-extra-hide-views" and i + 1 < len(args):
            raw = args[i + 1]
            opts["half_face_extra_hide_views"] = [int(s.strip()) for s in raw.split(",") if s.strip()]
            i += 2
            continue
        if arg == "--half-face-extra-hide-indices" and i + 1 < len(args):
            raw = args[i + 1]
            parsed = parse_indices_spec(raw)
            opts["half_face_extra_hide_indices"] = parsed
            i += 2
            continue
        if arg == "--half-face-extra-hide-indices-path" and i + 1 < len(args):
            opts["half_face_extra_hide_indices_path"] = args[i + 1]
            i += 2
            continue
        if arg == "--half-face-extra-hide-indices-by-view-path" and i + 1 < len(args):
            opts["half_face_extra_hide_indices_by_view_path"] = args[i + 1]
            i += 2
            continue
        if arg == "--half-face-extra-hide-indices-var" and i + 1 < len(args):
            opts["half_face_extra_hide_indices_var"] = args[i + 1]
            i += 2
            continue
        if arg == "--render-subdir" and i + 1 < len(args):
            opts["render_subdir"] = args[i + 1]
            i += 2
            continue
        if arg == "--outer-surface-only":
            opts["outer_surface_only"] = True
            i += 1
            continue
        i += 1

    return mesh_path, output_base_dir, samples, cam_npz_path, opts


mesh_path, output_base_dir, samples, cam_npz_path, opts = parse_args(argv)

def enable_gpus():
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    has_non_cpu = False
    for device in cycles_preferences.devices:
        if device.type == "CPU":
            device.use = False
        else:
            device.use = True
            has_non_cpu = True
    if not has_non_cpu:
        for device in cycles_preferences.devices:
            device.use = True
    cycles_preferences.compute_device_type = os.environ.get("BLENDER_CYCLES_DEVICE", "OPTIX")
    bpy.context.scene.cycles.device = "GPU"

enable_gpus()

def enforce_backface_culling(mat):
    mat.use_backface_culling = True
    if mat.get("backface_culling_wrapped"):
        return
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    if output is None:
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (400, 0)

    if not output.inputs["Surface"].links:
        return
    surf_link = output.inputs["Surface"].links[0]
    orig_socket = surf_link.from_socket

    geom = nodes.new(type="ShaderNodeNewGeometry")
    geom.location = (orig_socket.node.location.x - 300, orig_socket.node.location.y - 200)

    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    transparent.location = (orig_socket.node.location.x - 300, orig_socket.node.location.y - 400)

    mix = nodes.new(type="ShaderNodeMixShader")
    mix.location = (orig_socket.node.location.x + 200, orig_socket.node.location.y)

    links.new(geom.outputs["Backfacing"], mix.inputs["Fac"])
    links.new(orig_socket, mix.inputs[1])
    links.new(transparent.outputs["BSDF"], mix.inputs[2])

    links.remove(surf_link)
    links.new(mix.outputs["Shader"], output.inputs["Surface"])

    mat["backface_culling_wrapped"] = True

# -------------------------------------------------------------
# 1. Load GLB & FIX ROTATION
# -------------------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.import_scene.gltf(filepath=mesh_path)
imported_objects = bpy.context.selected_objects 

# Separate Meshes from Empty/Parent helpers
imported_meshes = [obj for obj in imported_objects if obj.type == 'MESH']

# --- CRITICAL FIX: RESET TRANSFORMS ---
# The GLTF importer often adds a -90 X rotation or creates a hierarchy.
# We must clear this so the mesh vertices align 1:1 with the numpy camera arrays.
print("Resetting object transforms to match raw data...")

# # 1. Clear Parent relationships (keep transformation)
# for obj in imported_objects:
#     obj.select_set(True)
# bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
# bpy.ops.object.select_all(action='DESELECT')

# # 2. Reset Rotation/Location/Scale to Identity
# # This puts the vertices in Blender exactly where they are in your PyTorch tensor.
# for obj in imported_objects:
#     obj.location = (0, 0, 0)
#     obj.rotation_euler = (0, 0, 0)
#     obj.scale = (1, 1, 1)
print("Baking imported transforms into mesh data (so world coords match cameras)...")

# 1) Manually clear parents (operator can be flaky in background mode)
for obj in imported_objects:
    if obj.parent is not None:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw

# 2) Bake transforms into mesh geometry, then reset object transforms
I = mathutils.Matrix.Identity(4)
for obj in imported_objects:
    if obj.type == "MESH" and obj.data is not None:
        obj.data.transform(obj.matrix_world)   # bake into vertices
        obj.data.update()
    obj.matrix_world = I

# -------------------------------------------------------------
# 2. Restore Materials
# -------------------------------------------------------------
mat_assignments = {
    "target.001": "wireframe",
    "scan.001": "scan",
    "reconstructed.001": "wireframe",
    "mochi":  "wireframe",
    "mochi_tto": "wireframe",
    "Mesh": "wireframe"
}

for mesh_obj in imported_meshes:
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.shade_smooth()
    
    mat_name = mat_assignments.get(mesh_obj.name)
    if not mat_name:
        for key in mat_assignments:
            if key in mesh_obj.name:
                mat_name = mat_assignments[key]
                break
    
    if mat_name and mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = mat
        else:
            mesh_obj.data.materials.append(mat)
    if mesh_obj.data:
        for mat in mesh_obj.data.materials:
            if mat is not None:
                enforce_backface_culling(mat)

# -------------------------------------------------------------
# 2.5 Optional: Half-face Mask
# -------------------------------------------------------------
def load_half_face_indices(path, var_name):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    if var_name not in data:
        raise ValueError(f"Variable '{var_name}' not found in {path}")
    return [int(i) for i in data[var_name]]

def load_extra_hide_indices(path, var_name):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    if var_name not in data:
        raise ValueError(f"Variable '{var_name}' not found in {path}")
    return [int(i) for i in data[var_name]]

def load_extra_hide_indices_by_view_txt(path):
    by_view = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Invalid line {line_no} in {path}: expected '<view>: <indices>'")
            view_part, indices_part = stripped.split(":", 1)
            view_id = int(view_part.strip())
            if view_id in by_view:
                raise ValueError(f"Duplicate view {view_id} in {path} at line {line_no}")
            parsed = parse_indices_spec(indices_part.strip())
            by_view[view_id] = sorted(set(int(i) for i in parsed))
    return by_view

def should_mask_mesh(mesh_obj, allowed_names_lower):
    name_lower = mesh_obj.name.lower()
    if allowed_names_lower is not None:
        return any(token in name_lower for token in allowed_names_lower)
    if "scan" in name_lower:
        return False
    return any(token in name_lower for token in ("mochi", "target", "reconstructed", "flame"))

def apply_half_face_mask(mesh_obj, keep_indices):
    if mesh_obj.data.users > 1:
        mesh_obj.data = mesh_obj.data.copy()
    mesh = mesh_obj.data
    max_idx = max(keep_indices)
    if max_idx >= len(mesh.vertices):
        print(f"Skipping half-face mask for {mesh_obj.name}: {len(mesh.vertices)} verts < max idx {max_idx}")
        return False

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    keep = set(keep_indices)
    faces_to_remove = [f for f in bm.faces if any(v.index not in keep for v in f.verts)]
    bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return True

def remap_extra_indices_to_mesh(indices_to_hide, mesh_vert_count, keep_indices):
    if not indices_to_hide:
        return []

    resolved = set()
    raw = [int(i) for i in indices_to_hide]
    for idx in raw:
        if 0 <= idx < mesh_vert_count:
            resolved.add(idx)

    if keep_indices:
        keep_list = [int(i) for i in keep_indices]
        full_to_local = {full_idx: local_idx for local_idx, full_idx in enumerate(keep_list)}
        for idx in raw:
            local_idx = full_to_local.get(idx)
            if local_idx is not None and 0 <= local_idx < mesh_vert_count:
                resolved.add(local_idx)

        max_raw = max(raw)
        keep_set = set(keep_list)
        hidden_sorted = [i for i in range(max_raw + 1) if i not in keep_set]
        for idx in raw:
            local_idx = idx - bisect_left(hidden_sorted, idx)
            if 0 <= local_idx < mesh_vert_count:
                resolved.add(local_idx)

    return sorted(resolved)

def remove_faces_touching_indices(mesh_data, indices_to_hide, mesh_name, keep_indices):
    mapped_indices = remap_extra_indices_to_mesh(indices_to_hide, len(mesh_data.vertices), keep_indices)
    if not mapped_indices:
        print(
            f"Extra hide indices map to nothing on {mesh_name} "
            f"(mesh verts={len(mesh_data.vertices)}). Skipping."
        )
        return False
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    bm.verts.ensure_lookup_table()
    hide = set(mapped_indices)
    faces_to_remove = [f for f in bm.faces if any(v.index in hide for v in f.verts)]
    bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()
    print(
        f"Applied extra hide on {mesh_name}: requested={len(indices_to_hide)} "
        f"mapped={len(mapped_indices)}"
    )
    return True

def build_outer_surface_view_mesh(mesh_obj, cam_obj):
    source_mesh = mesh_obj.data
    if source_mesh is None:
        raise ValueError(f"Mesh data missing for {mesh_obj.name}")

    if len(source_mesh.polygons) == 0:
        return source_mesh.copy()

    verts_world = [mesh_obj.matrix_world @ v.co for v in source_mesh.vertices]
    polygons = [list(poly.vertices) for poly in source_mesh.polygons]
    bvh = mathutils.bvhtree.BVHTree.FromPolygons(verts_world, polygons, all_triangles=False)
    if bvh is None:
        raise RuntimeError(f"Failed to build BVH for {mesh_obj.name}")

    cam_loc = cam_obj.matrix_world.translation.copy()
    normal_mat = mesh_obj.matrix_world.to_3x3()
    eps = 1e-5
    keep_poly_indices = set()

    def sample_is_visible(sample_world, poly_index):
        to_sample = sample_world - cam_loc
        dist = to_sample.length
        if dist <= eps:
            return False
        ray_dir = to_sample / dist
        hit_loc, hit_normal, hit_index, hit_dist = bvh.ray_cast(cam_loc + ray_dir * eps, ray_dir, dist + eps)
        if hit_index is None:
            return False
        if hit_index == poly_index:
            return True
        tol = max(1e-4, dist * 1e-3)
        return hit_dist >= (dist - tol)

    for poly in source_mesh.polygons:
        poly_vert_ids = list(poly.vertices)
        if len(poly_vert_ids) < 3:
            continue

        center_world = mesh_obj.matrix_world @ poly.center
        normal_world = (normal_mat @ poly.normal).normalized()
        if normal_world.dot(cam_loc - center_world) <= 0.0:
            continue

        sample_points = [center_world]

        vert_step = max(1, len(poly_vert_ids) // 6)
        for idx in range(0, len(poly_vert_ids), vert_step):
            sample_points.append(verts_world[poly_vert_ids[idx]])
            if len(sample_points) >= 8:
                break

        v0 = verts_world[poly_vert_ids[0]]
        for idx in range(1, len(poly_vert_ids) - 1):
            v1 = verts_world[poly_vert_ids[idx]]
            v2 = verts_world[poly_vert_ids[idx + 1]]
            sample_points.append((v0 + v1 + v2) / 3.0)
            sample_points.append(v0 * 0.2 + v1 * 0.4 + v2 * 0.4)
            if len(sample_points) >= 20:
                break

        for sample_world in sample_points:
            if sample_is_visible(sample_world, poly.index):
                keep_poly_indices.add(poly.index)
                break

    view_mesh = source_mesh.copy()
    bm = bmesh.new()
    bm.from_mesh(view_mesh)
    bm.faces.ensure_lookup_table()
    faces_to_remove = [face for idx, face in enumerate(bm.faces) if idx not in keep_poly_indices]
    if faces_to_remove:
        bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
    bm.to_mesh(view_mesh)
    bm.free()
    view_mesh.update()
    return view_mesh

render_subdir = opts.get("render_subdir", "renders")
if render_subdir is None:
    render_subdir = "renders"
render_subdir = render_subdir.strip()
if not render_subdir or render_subdir.lower() == "none":
    render_subdir = "renders"

half_face_enabled = opts["half_face"]
half_face_suffix = "_half" if half_face_enabled else ""
outer_surface_only = opts["outer_surface_only"]

if half_face_enabled:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    indices_path = opts["half_face_indices_path"] or os.path.join(repo_root, "utils", "flame_indices.py")
    keep_indices = load_half_face_indices(indices_path, opts["half_face_var"])
    allowed = opts["half_face_meshes"]
    extra_hide_indices_by_view = {}

    extra_hide_views = set(opts["half_face_extra_hide_views"] or [])
    extra_hide_indices = list(opts["half_face_extra_hide_indices"] or [])
    extra_hide_path = opts["half_face_extra_hide_indices_path"]
    if extra_hide_path:
        extra_hide_indices.extend(
            load_extra_hide_indices(extra_hide_path, opts["half_face_extra_hide_indices_var"])
        )
    if extra_hide_indices:
        shared_indices = sorted(set(int(i) for i in extra_hide_indices))
        for view_id in extra_hide_views:
            extra_hide_indices_by_view[int(view_id)] = list(shared_indices)

    extra_hide_by_view_path = opts["half_face_extra_hide_indices_by_view_path"]
    if extra_hide_by_view_path:
        loaded_by_view = load_extra_hide_indices_by_view_txt(extra_hide_by_view_path)
        for view_id, indices in loaded_by_view.items():
            if view_id in extra_hide_indices_by_view:
                merged = extra_hide_indices_by_view[view_id] + indices
                extra_hide_indices_by_view[view_id] = sorted(set(int(i) for i in merged))
            else:
                extra_hide_indices_by_view[view_id] = sorted(set(int(i) for i in indices))
    extra_hide_mesh_data = {}

    for mesh_obj in imported_meshes:
        if not mesh_obj.data:
            continue
        if should_mask_mesh(mesh_obj, allowed):
            applied = apply_half_face_mask(mesh_obj, keep_indices)
            if applied:
                print(f"Applied half-face mask to {mesh_obj.name}")
                if extra_hide_indices_by_view:
                    base_half_data = mesh_obj.data
                    extra_by_view = {}
                    for view_id, view_indices in sorted(extra_hide_indices_by_view.items()):
                        if not view_indices:
                            continue
                        extra_half_data = base_half_data.copy()
                        remove_faces_touching_indices(
                            extra_half_data,
                            view_indices,
                            f"{mesh_obj.name}@view_{view_id:02d}",
                            keep_indices
                        )
                        extra_by_view[int(view_id)] = extra_half_data
                    extra_hide_mesh_data[mesh_obj.name] = {
                        "base": base_half_data,
                        "extra_by_view": extra_by_view
                    }
                    print(
                        f"Prepared extra-hide half mesh for {mesh_obj.name} "
                        f"on views {sorted(extra_by_view.keys())}"
                    )

# -------------------------------------------------------------
# 3. Camera Setup
# -------------------------------------------------------------
def set_camera(cam_obj, K, E, width, height):
    cam = cam_obj.data
    scene = bpy.context.scene

    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")

    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 32.0
    cam.sensor_height = cam.sensor_width * (float(height) / float(width))

    fx = float(K[0, 0])
    cam.lens = fx * cam.sensor_width / float(width)

    cam.clip_start = 0.01
    cam.clip_end = 100.0

    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Correct principal point mapping
    cam.shift_x = (float(width) * 0.5 - cx) / float(width)
    # For sensor_fit='HORIZONTAL', Blender scales shift_y by W/H.
    cam.shift_y = (cy - float(height) * 0.5) / float(width)

    if E.shape == (3, 4):
        E = np.vstack([E, [0, 0, 0, 1]])
    elif E.shape != (4, 4):
        raise ValueError(f"E must be (3,4) or (4,4), got {E.shape}")

    # If E is world->camera (OpenCV), invert to get camera->world
    # E_inv = np.linalg.inv(E)
    E_inv = np.linalg.inv(E @ WORLD_FIX)   # or WORLD_FIX @ E depending on your convention

    # OpenCV cam axes -> Blender cam axes
    cv_to_blender = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64)

    blender_matrix = E_inv @ cv_to_blender

    cam_obj.matrix_world = mathutils.Matrix(blender_matrix.tolist())

# -------------------------------------------------------------
# 4. Render
# -------------------------------------------------------------
if not os.path.exists(cam_npz_path):
    sys.exit(1)

data = np.load(cam_npz_path)
K_all, E_all = data['K'], data['E']
W, H = int(data['W']), int(data['H'])

if "Camera" in bpy.data.objects:
    cam_obj = bpy.data.objects["Camera"]
else:
    bpy.ops.object.camera_add()
    cam_obj = bpy.context.active_object
    cam_obj.name = "Camera"
if cam_obj.constraints:
    cam_obj.constraints.clear()

scene = bpy.context.scene
scene.camera = cam_obj
scene.render.engine = "CYCLES"
scene.render.film_transparent = True
scene.cycles.samples = samples

for mesh_obj in imported_meshes:
    if not mesh_obj.data: continue
    
    for other in imported_meshes:
        other.hide_render = (other != mesh_obj)
        
    safe_mesh_name = mesh_obj.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    mesh_render_dir = os.path.join(output_base_dir, f"{render_subdir}_{safe_mesh_name}{half_face_suffix}")
    os.makedirs(mesh_render_dir, exist_ok=True)
    
    for v in range(K_all.shape[0]):
        if half_face_enabled and mesh_obj.name in extra_hide_mesh_data:
            mesh_hide_data = extra_hide_mesh_data[mesh_obj.name]
            mesh_obj.data = mesh_hide_data["extra_by_view"].get(v, mesh_hide_data["base"])
        current_mesh_data = mesh_obj.data
        set_camera(cam_obj, K_all[v], E_all[v], W, H)
        temp_outer_mesh = None
        if outer_surface_only:
            temp_outer_mesh = build_outer_surface_view_mesh(mesh_obj, cam_obj)
            mesh_obj.data = temp_outer_mesh
        filename = f"view_{v:02d}.png"
        scene.render.filepath = os.path.join(mesh_render_dir, filename)
        
        print(f"Rendering {safe_mesh_name} | View {v:02d}")
        bpy.ops.render.render(write_still=True)
        if temp_outer_mesh is not None:
            mesh_obj.data = current_mesh_data
            bpy.data.meshes.remove(temp_outer_mesh)
