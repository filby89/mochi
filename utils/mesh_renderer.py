"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms 
in the LICENSE file included with this software distribution. 
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

from psbody.mesh import Mesh

import os
# if not os.environ.get( 'PYOPENGL_PLATFORM' ):
    # os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl' 

import cv2
import imageio
import trimesh
import pyrender
import numpy as np

def dist_to_rgb(errors, min_dist=0.0, max_dist=1.0):
    # print(errors.shape)
    import matplotlib as mpl
    import matplotlib.cm as cm
    norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
    cmap = cm.get_cmap(name='jet')
    colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return colormapper.to_rgba(errors)[:, 0:3]





def render_mesh_pixels(vertices_ndc, faces, vertex_colors=None,
                    frustum={'near': 0.01, 'far': 3000.0},
                    image_size=(200, 600)):
    """
    Render a mesh whose vertices are already in NDC space.

    Args:
        vertices_ndc: (N,2) or (N,3) array with:
            x, y in [-1, 1] (NDC). If z provided, any range; it will be normalized for depth.
        faces: (F,3) int array of triangle indices, or None for point cloud.
        vertex_colors: (N,3|4) floats in [0,1] (optional).
        frustum: dict with 'near', 'far' for z range.
        image_size: (H, W) output resolution in pixels.

    Returns:
        color: uint8 image of shape (H, W, 3)
    """
    H, W = image_size
    verts = np.asarray(vertices_ndc, dtype=np.float64).copy()

    # Ensure z present
    if verts.shape[1] == 2:
        z_coords = np.zeros((verts.shape[0],), dtype=np.float64)
        verts = np.column_stack([verts, z_coords])
    else:
        z_coords = verts[:, 2].copy()

    x_ndc = verts[:, 0]
    y_ndc = verts[:, 1]

    # Depth: normalize to [near, 2.0] for visibility (same behavior as before)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_diff = z_max - z_min
    if abs(z_diff) < 1e-7:
        z_diff = 1.0
    z_scale = (2.0 - frustum['near']) / z_diff
    z_vis = z_scale * (z_coords - z_min) + frustum['near']

    verts_render = np.stack([x_ndc, y_ndc, z_vis], axis=-1)

    # Colors
    if vertex_colors is not None and vertex_colors.shape[0] == verts_render.shape[0]:
        rgb_per_v = vertex_colors.astype(np.float64)
    else:
        rgb_per_v = 0.7 * np.ones_like(verts_render)

    # Mesh / points
    if faces is None:
        mesh = pyrender.Mesh.from_points(verts_render, colors=rgb_per_v)
    else:
        tri_mesh = trimesh.Trimesh(vertices=verts_render, faces=faces,
                                   vertex_colors=rgb_per_v, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    # Scene
    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    scene.add(mesh, pose=np.eye(4))

    # Orthographic camera: world coords already in [-1,1] extents
    cam = pyrender.OrthographicCamera(
        xmag=1.0, ymag=1.0,
        znear=frustum['near'], zfar=frustum['far']
    )
    cam_pose = np.eye(4)
    # Keep same "upright" look as your previous renderer
    cam_pose[:3, :3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    scene.add(cam, pose=cam_pose)

    # Simple 4-point lighting around view direction
    angle = np.pi / 8.0
    pos = np.array([0, 0, -2])
    light = pyrender.PointLight(color=np.array([1., 1., 1.]), intensity=0.8)
    lp = np.eye(4)
    lp[:3, 3] = cv2.Rodrigues(np.array([ angle, 0, 0]))[0].dot(pos) + np.array([0, 0, 1]); scene.add(light, pose=lp.copy())
    lp[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos) + np.array([0, 0, 1]); scene.add(light, pose=lp.copy())
    lp[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos) + np.array([0, 0, 1]); scene.add(light, pose=lp.copy())
    lp[:3, 3] = cv2.Rodrigues(np.array([0,  angle, 0]))[0].dot(pos) + np.array([0, 0, 1]); scene.add(light, pose=lp.copy())

    # Render
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    try:
        color, _ = r.render(scene)
    finally:
        r.delete()

    return color.astype(np.uint8)

def render_mesh(vertices, faces, vertex_colors=None, camera_extrinsics=None, camera_intrinsics=None, radial_distortion=None, 
                frustum={'near': 0.01, 'far': 3000.0}, image_size=(800, 800), needs_projection=True):
    '''
    Returns a rendered mesh as uint8 tensor of size (H,W,3)
    '''
    # print('Rendering mesh', vertices.shape, faces.shape)
    if needs_projection:
        vertices = np.copy(vertices)
        if camera_extrinsics is not None:
            num_points, _ = vertices.shape
            ones = np.ones((num_points, 1))        
            points_homogeneous = np.concatenate((vertices, ones), axis=-1)
            vertices = camera_extrinsics.dot(points_homogeneous.T).T
    else:
        # If no projection is needed, we assume the vertices are already in camera coordinates
        vertices = np.copy(vertices)

    z_coords = vertices[:,2].copy()
    z_coords[np.where(np.abs(z_coords) < 1e-7)] = 1.0
    vertices[:,0] = vertices[:,0] / z_coords
    vertices[:,1] = vertices[:,1] / z_coords
    vertices[:,2] = 1.0

    if radial_distortion is not None:
        K1, K2 = radial_distortion[0], radial_distortion[1]
        r2 = vertices[:,0]**2 + vertices[:,1]**2
        r4 = r2**2
        radial_distortion_factor = (1 + K1*r2 + K2*r4)
        vertices[:,0] = vertices[:,0]*radial_distortion_factor
        vertices[:,1] = vertices[:,1]*radial_distortion_factor    

    if camera_intrinsics is None:
        camera_intrinsics = np.array([
                [ 3000,      0,     image_size[0]//2 ],
                [    0,    3000,    image_size[1]//2 ],
                [    0,      0,        1.0 ]])

    vertices = camera_intrinsics.dot(vertices.T).T

    # Normalize the x and y coordinates
    vertices[:,0] = 2*(vertices[:,0] - (image_size[1]/2)) / image_size[0] 
    vertices[:,1] = 2*(vertices[:,1] - (image_size[0]/2)) / image_size[0]

    # Normalize z to be in [z_near, 2.0]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_diff = z_max-z_min
    if z_diff < 1e-7:
        z_diff = 1.0
    z_scale = (2.0-frustum['near'])/z_diff
    vertices[:,2] = z_scale*(z_coords-z_min)+frustum['near'] 

    rgb_per_v = 0.7*np.ones_like(vertices)
    if vertex_colors is not None:
        if vertex_colors.shape[0] == vertices.shape[0]:
            rgb_per_v[:] = vertex_colors

    if faces is None:
        render_mesh = pyrender.Mesh.from_points(vertices, colors=rgb_per_v)
    else:
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    scene.add(render_mesh, pose=np.eye(4))

    camera = pyrender.camera.OrthographicCamera(xmag=1.0, ymag=1.0, znear=frustum['near'], zfar=frustum['far'])
    camera_pose = np.eye(4)
    camera_pose[:3,:3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    scene.add(camera, pose=camera_pose)

    angle = np.pi / 8.0
    pos = np.array([0,0,-2])
    light_color = np.array([1., 1., 1.])
    light = pyrender.PointLight(color=light_color, intensity=0.8)
    
    light_pose = np.eye(4)
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_height=image_size[0], viewport_width=image_size[1])
        color, _ = r.render(scene) #, flags=flags)
    except Exception as e:
        print('pyrender: Failed rendering frame')
        # print(e)
        raise e
        color = np.zeros((image_size[0], image_size[1], 3), dtype='uint8')
    return color
 
def render_textured_mesh(scan_fname, texture_fname, camera_extrinsics=None, camera_intrinsics=None, radial_distortion=None, 
                frustum={'near': 0.01, 'far': 3000.0}, image_size=(800, 800)):
    '''
    Returns a rendered mesh as uint8 tensor of size (H,W,3)
    '''

    tri_mesh = trimesh.load(scan_fname, process=False)
    # tri_mesh.vertices = 0.001*np.array(tri_mesh.vertices)
    texture = imageio.imread(texture_fname)

    vertices = np.copy(tri_mesh.vertices)
    if camera_extrinsics is not None:
        num_points, _ = vertices.shape
        ones = np.ones((num_points, 1))        
        points_homogeneous = np.concatenate((vertices, ones), axis=-1)
        vertices = camera_extrinsics.dot(points_homogeneous.T).T

    z_coords = vertices[:,2].copy()
    z_coords[np.where(np.abs(z_coords) < 1e-7)] = 1.0
    vertices[:,0] = vertices[:,0] / z_coords
    vertices[:,1] = vertices[:,1] / z_coords
    vertices[:,2] = 1.0

    if radial_distortion is not None:
        K1, K2 = radial_distortion[0], radial_distortion[1]
        r2 = vertices[:,0]**2 + vertices[:,1]**2
        r4 = r2**2
        radial_distortion_factor = (1 + K1*r2 + K2*r4)
        vertices[:,0] = vertices[:,0]*radial_distortion_factor
        vertices[:,1] = vertices[:,1]*radial_distortion_factor    

    if camera_intrinsics is None:
        camera_intrinsics = np.array([
                [ 3000,      0,     image_size[0]//2 ],
                [    0,    3000,    image_size[1]//2 ],
                [    0,      0,        1.0 ]])

    vertices = camera_intrinsics.dot(vertices.T).T

    # Normalize the x and y coordinates
    vertices[:,0] = 2*(vertices[:,0] - (image_size[1]/2)) / image_size[0] 
    vertices[:,1] = 2*(vertices[:,1] - (image_size[0]/2)) / image_size[0]

    # Normalize z to be in [z_near, 2.0]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_diff = z_max-z_min
    if z_diff < 1e-7:
        z_diff = 1.0
    z_scale = (2.0-frustum['near'])/z_diff
    vertices[:,2] = z_scale*(z_coords-z_min)+frustum['near'] 

    # material = trimesh.visual.texture.SimpleMaterial(image=texture)
    # visuals = trimesh.visual.TextureVisuals(uv=uv, image=texture, material=material)
    # tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visuals=visuals)
    # tri_mesh = trimesh.Trimesh(vertices=tri_mesh.vertices, faces=tri_mesh.faces, visuals=tri_mesh._visual, process=False)
    tri_mesh.vertices = vertices
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
    scene.add(render_mesh, pose=np.eye(4))

    camera = pyrender.camera.OrthographicCamera(xmag=1.0, ymag=1.0, znear=frustum['near'], zfar=frustum['far'])
    camera_pose = np.eye(4)
    camera_pose[:3,:3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    scene.add(camera, pose=camera_pose)

    angle = np.pi / 8.0
    pos = np.array([0,0,-2])
    light_color = np.array([1., 1., 1.])
    light = pyrender.PointLight(color=light_color, intensity=2.0)
    
    light_pose = np.eye(4)
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_height=image_size[0], viewport_width=image_size[1])
        color, _ = r.render(scene, flags=flags)
    except Exception as e:
        print('pyrender: Failed rendering frame')
        print(e)
        color = np.zeros((image_size[0], image_size[1], 3), dtype='uint8')
    return color
 


def test_mesh_rendering():
    import os
    import numpy as np
    import trimesh
    import cv2
    
    # Define paths
    flame_obj = "data/template/sampling_template.obj"
    mask_path = "data/template/vertex_masks2.npz"
    output_dir = "data/mask_visualizations"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the FLAME mesh
    if not os.path.exists(flame_obj):
        print(f"Error: Mesh file not found at {flame_obj}")
        return
    
    # Load mesh using trimesh (process=False preserves vertex order matching the masks)
    mesh = trimesh.load(flame_obj, process=False)
    vertices = np.array(mesh.vertices)*1000.0
    faces = np.array(mesh.faces)

    # Load masks
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        return
    vertex_masks = dict(np.load(mask_path))
    # convert to dict

    
    # print(vertex_masks['face'].shape)
    # print(vertex_masks.files)
    # print([vertex_masks[x].shape for x in vertex_masks.files if x not in ['boundary','scalp','vertex_count']])
    vertex_masks['total'] = np.concatenate([vertex_masks[x] for x in vertex_masks.keys() if x not in ['boundary','scalp','vertex_count']],axis=0)
    
    # Define Colors [R, G, B] in range 0-1
    # Light Blue for the active mask area
    color_highlight = np.array([1.0, 0.3, 0.2]) 
    # Gray for the rest of the face
    color_base = np.array([0.7, 0.7, 0.7]) 

    # Define Camera Extrinsics
    # FLAME is centered at (0,0,0). We need to move it along Z so it's visible 
    # to the camera (which is at 0,0,0 looking forward).
    # We create a transformation matrix that translates the mesh +0.4 meters in Z.
    # camera_extrinsics = np.eye(4)
    # camera_extrinsics[2, 3] = -3 # Translation Z
    # camera_extrinsics = camera_extrinsics[:3,:4]
    from pytorch3d.transforms import axis_angle_to_matrix
    import torch

    # print(f"Found masks: {vertex_masks.files}")
    # Extr = self.inputs['camera_extrinsics'][0, 0].detach().cpu().numpy().copy()
    camera_extrinsics = np.eye(4)
    # Slight tilt from above; feel free to tweak the 1st component
    R = axis_angle_to_matrix(torch.tensor([[3.125, 0.0, 0.0]], dtype=torch.float32)).squeeze(0).numpy()
    camera_extrinsics[:3, :3] = R
    camera_extrinsics[0, 3] = -50.0
    camera_extrinsics[1, 3] = -38.0
    camera_extrinsics[2, 3] = 1350.0
    camera_extrinsics = camera_extrinsics[:3]

    for key in vertex_masks.keys():
        # 'vertex_count' is a metadata field, not a mask
        if key == 'vertex_count':
            continue
            
        print(f"Rendering mask: {key}")
        
        # Get vertex indices for the current mask
        mask_indices = vertex_masks[key]
        
        # Initialize all vertices with the base gray color
        # Shape: (N_vertices, 3)
        current_colors = np.tile(color_base, (vertices.shape[0], 1))
        
        # Update the specific mask indices to light blue
        # Note: We ensure indices are integers suitable for indexing
        current_colors[mask_indices] = color_highlight
        
        try:
            # Render the mesh
            # We use a standard image size of 800x800
            img = render_mesh(
                vertices, 
                faces, 
                vertex_colors=current_colors, 
                camera_extrinsics=camera_extrinsics, 
                image_size=(800, 800), 
                needs_projection=True
            )
            
            # Save result
            # Note: render_mesh returns RGB, cv2 uses BGR, so we flip channels
            save_path = os.path.join(output_dir, f"{key}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Failed to render {key}: {e}")

    print(f"All renders saved to {output_dir}")

if __name__ == '__main__':
    test_mesh_rendering()
    print('Done')