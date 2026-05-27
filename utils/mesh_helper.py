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

import torch
import kaolin
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MeshHelper:
    def __init__(self, num_vertices, faces):
        super().__init__()

        self.num_vertices = num_vertices
        if isinstance(faces, torch.Tensor):
            self.faces = faces.long()
        else:
            self.faces = torch.from_numpy(faces.astype('int32')).long()
    
        self.vertices_by_faces = MeshHelper.vertex_by_face(num_vertices, faces)

    def face_vertices(self, vertices, faces):
        """ 
        :param vertices: [batch size, number of vertices, 3]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of faces, 3, 3]
        """
        assert (vertices.ndimension() == 3)
        assert (faces.ndimension() == 3)
        assert (vertices.shape[0] == faces.shape[0])
        assert (vertices.shape[2] == 3)
        assert (faces.shape[2] == 3)

        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]
        device = vertices.device
        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices = vertices.reshape((bs * nv, 3))
        # pytorch only supports long and byte tensors for indexing
        return vertices[faces.long()]

    def face_normals(self, vertices):
        """
        Compute the normalized normals for each face in the mesh.
        
        Args:
            vertices: torch.Tensor of shape (B, V, 3)
                Batched vertex positions.
        
        Returns:
            torch.Tensor of shape (B, F, 3)
                Batched, normalized face normals.
        """
        # vertices: (B, V, 3)
        # self.faces: (F, 3) with vertex indices
        faces = self.faces.to(vertices.device)           # (F, 3)
        # Gather the three vertices for each face:
        v0 = vertices[:, faces[:, 0], :]                 # (B, F, 3)
        v1 = vertices[:, faces[:, 1], :]                 # (B, F, 3)
        v2 = vertices[:, faces[:, 2], :]                 # (B, F, 3)
        
        # Compute two edge vectors per face
        edge1 = v1 - v0                                   # (B, F, 3)
        edge2 = v2 - v0                                   # (B, F, 3)
        
        # Cross product gives the (unnormalized) face normal
        normals = torch.cross(edge1, edge2, dim=-1)      # (B, F, 3)
        
        # Normalize to unit length
        normals = F.normalize(normals, p=2, dim=-1)      # (B, F, 3)
        
        return normals


    def vertex_normals(self, vertices):
        '''
        Compute the normalized vertex normals by averaging the normals of all faces adjacent to the vertex.
        Args:
            vertices: (B, L, 3), batched vertices
        Returns:
            vertex_normals: (B, L, 3), batched normalized vertex normals
        '''

        batch_size, num_vertices = vertices.shape[:2]
        device = vertices.device

        if num_vertices != self.num_vertices:
            raise RuntimeError(f"Wrong number of vertices: {num_vertices} != {self.num_vertices}")

        # Compute edges
        faces = self.faces.to(device)
        edges1 = torch.index_select(vertices, dim=1, index=faces[:, 1]) - \
                    torch.index_select(vertices, dim=1, index=faces[:, 0])
        edges2 = torch.index_select(vertices, dim=1, index=faces[:, 2]) -\
                    torch.index_select(vertices, dim=1, index=faces[:, 0])
    
        # Compute face normals
        face_normals = torch.cross(edges1, edges2, dim=-1)

        # Compute vertex normals by averaging the normals of all adjacent faces
        vertices_by_faces = self.vertices_by_faces.to(device)
        vertex_normals = []
        for batch_idx in range(batch_size):
            vertex_normals.append(torch.matmul(vertices_by_faces, face_normals[batch_idx]))
        vertex_normals = torch.stack(vertex_normals).contiguous()
        return torch.nn.functional.normalize(vertex_normals, dim=-1)

    

    def vertex_normals(self, vertices):
        '''
        Compute the normalized vertex normals by averaging the normals of all faces adjacent to the vertex.
        Args:
            vertices: (B, L, 3), batched vertices
        Returns:
            vertex_normals: (B, L, 3), batched normalized vertex normals
        '''

        batch_size, num_vertices = vertices.shape[:2]
        device = vertices.device

        if num_vertices != self.num_vertices:
            raise RuntimeError(f"Wrong number of vertices: {num_vertices} != {self.num_vertices}")

        # Compute edges
        faces = self.faces.to(device)
        edges1 = torch.index_select(vertices, dim=1, index=faces[:, 1]) - \
                    torch.index_select(vertices, dim=1, index=faces[:, 0])
        edges2 = torch.index_select(vertices, dim=1, index=faces[:, 2]) -\
                    torch.index_select(vertices, dim=1, index=faces[:, 0])
    
        # Compute face normals
        face_normals = torch.cross(edges1, edges2, dim=-1)

        # Compute vertex normals by averaging the normals of all adjacent faces
        vertices_by_faces = self.vertices_by_faces.to(device)
        vertex_normals = []
        for batch_idx in range(batch_size):
            vertex_normals.append(torch.matmul(vertices_by_faces, face_normals[batch_idx]))
        vertex_normals = torch.stack(vertex_normals).contiguous()
        return torch.nn.functional.normalize(vertex_normals, dim=-1)

    def vertex_visibility(self, vertices, camera_centers):
        '''
        Args:
            vertices: (B, L, 3), batched vertices
            camera_centers: (B, V, 3), batched camera centers for all V views
        Returns:
            visibilities: (B, V, L) batched binary vertex visibilities, with 0 = invisible, 1 = visible
        '''

        from psbody.mesh.visibility import visibility_compute

        batch_size, num_views, _ = camera_centers.shape
        num_vertices = vertices.shape[1]
        device = vertices.device

        if num_vertices != self.num_vertices:
            raise RuntimeError(f"Wrong number of vertices: {num_vertices} != {self.num_vertices}")

        np_vertices = vertices.detach().cpu().numpy().astype('float64')
        faces = self.faces.detach().cpu().numpy().astype('uint32')

        visibilities = []
        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                camera_center = camera_centers[batch_idx, view_idx].detach().cpu().numpy().astype('float64')              
                tmp_visibility, _ = np.array(visibility_compute(v=np_vertices[batch_idx], f=faces, cams=camera_center.reshape(1,3)))
                visibilities.append(tmp_visibility.squeeze())
        visibilities = np.stack(visibilities, axis=0).astype('float32').reshape((batch_size, num_views, num_vertices))
        return torch.from_numpy(visibilities).to(device)  

    def render_normals(self, vertices, camera_intrinsics, camera_extrinsics, radial_distortions=None, depth_eps=3e-3, eps=1e-7, depth_rendering_height=200, depth_rendering_width=200):
        '''
        Compute the vertex visibility with a thresholded depth rendering check. 

        Args:
            vertices: (B, L, 3), batched vertices
            camera_intrinsics: (B, V, 3, 3), batched camera intrinsics for all V views
            camera_extrinsics: (B, V, 3, 4), batched camera extrinsics for all V views
        Returns:
            visibilities: (B, V, L) batched binary vertex visibilities, with 0 = invisible, 1 = visible
        '''

        device = vertices.device
        batch_size, num_vertices, _ = vertices.shape
        num_faces, _ = self.faces.shape
        _, num_views, _, _ = camera_extrinsics.shape

        faces = self.faces.unsqueeze(0).repeat(batch_size*num_views,1,1).to(device) # (batch_size*num_views, num_faces, 3)
        camera_intrinsics = camera_intrinsics.contiguous().view(-1, 3, 3)   # (batch_size*num_views, 3, 3)
        camera_extrinsics = camera_extrinsics.contiguous().view(-1, 3, 4)   # (batch_size*num_views, 3, 4)

        vertices = vertices.transpose(1, 2) # (batch_size, 3, num_vertices)
        ones = torch.ones(batch_size, 1, num_vertices).to(device)   # (batch_size, 1, num_vertices)
        vertices_homogeneous = torch.cat((vertices, ones), axis=-2) # (batch_size, 4, num_vertices)

        vertices_homogeneous = vertices_homogeneous.unsqueeze(1).repeat(1,num_views,1,1).contiguous()    # (batch_size, num_views, 4, num_vertices)
        vertices_homogeneous = vertices_homogeneous.view(-1, 4, num_vertices)   # (batch_size*num_views, 4, num_vertices)

        # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
        vertices_image = camera_extrinsics.bmm(vertices_homogeneous)    # (batch_size*num_views, 3, num_vertices)
       
        vertices_depth = torch.clone(vertices_image[:,2,:])             # (batch_size*num_views, num_vertices)       
        # min_depth, max_depth = torch.floor(vertices_depth.min(dim=-1).values), torch.ceil(vertices_depth.max(dim=-1).values)
        # vertices_depth = (vertices_depth-min_depth.unsqueeze(-1))/(max_depth.unsqueeze(-1)-min_depth.unsqueeze(-1))

        # Transformation from 3D camera coordinate system to the undistorted image plane 
        mask = (vertices_image.abs() < eps)
        mask[:,:2,:] = False
        vertices_image[mask] = 1.0 # Avoid division by zero

        vertices_image[:,0,:] = vertices_image[:,0,:] / vertices_image[:,2,:]  
        vertices_image[:,1,:] = vertices_image[:,1,:] / vertices_image[:,2,:]  
        vertices_image[:,2,:] = 1.0

        if radial_distortions is not None:
            radial = radial_distortions.to(device)
            if radial.dim() == 2:                  # (B,2)  → broadcast to all views
                radial = radial.unsqueeze(1).expand(batch_size, num_views, 2)
            elif radial.dim() == 3 and radial.shape[1] == 1:    # (B,1,2)
                radial = radial.expand(batch_size, num_views, 2)
            elif radial.dim() != 3 or radial.shape[1] != num_views:
                raise ValueError("radial_distortions must have shape (B,2) or (B,V,2)")
            
            radial = radial.contiguous().view(-1, 2)            # (B*V,2)
            k1 = radial[:, 0].unsqueeze(-1)                     # (B*V,1)
            k2 = radial[:, 1].unsqueeze(-1)                     # (B*V,1)

            x = vertices_image[:, 0, :]                         # (B*V,L)
            y = vertices_image[:, 1, :]
            r2 = x.pow(2) + y.pow(2)
            radial_factor = 1 + k1 * r2 + k2 * r2.pow(2)
            vertices_image[:, 0, :] = x * radial_factor
            vertices_image[:, 1, :] = y * radial_factor

        # -------------------------------------------
        # Transformation from image coordinates to the final image coordinates with the camera intrinsics
        vertices_image = camera_intrinsics.bmm(vertices_image)      # (batch_size*num_views, 3, num_vertices)    
        vertices_image = vertices_image.transpose(1,2)              # (batch_size*num_views, num_vertices, 3)
        vertices_image[:,:,2] = vertices_depth

        # Normalize image coordinates to [-1,1]
        # x_min, x_max = torch.floor(vertices_image[:,:,0].min(dim=1).values), torch.ceil(vertices_image[:,:,0].max(dim=1).values)
        # y_min, y_max = torch.floor(vertices_image[:,:,1].min(dim=1).values), torch.ceil(vertices_image[:,:,1].max(dim=1).values)
        # x_center, y_center = (x_max+x_min)/2.0, (y_max+y_min)/2.0
        # normalizing_scale = 1.0/torch.max(x_max-x_min, y_max-y_min)
        # vertices_image[:,:,0] = 2*normalizing_scale.unsqueeze(-1)*(vertices_image[:,:,0]-x_center.unsqueeze(-1))
        # vertices_image[:,:,1] = 2*normalizing_scale.unsqueeze(-1)*(vertices_image[:,:,1]-y_center.unsqueeze(-1))

        # Normalize image coordinates to [-1,1]
        Bv, Nv, _ = vertices_image.shape
        W, H = depth_rendering_width, depth_rendering_height           # your target res

        cx = camera_intrinsics[:,0,2].unsqueeze(-1)    # (B*V,1)
        cy = camera_intrinsics[:,1,2].unsqueeze(-1)    # (B*V,1)

        # vertices_image[:,:,0] =  2.0 * (vertices_image[:,:,0] - cx) / W     # +X right
        # vertices_image[:,:,1] =  2.0 * (vertices_image[:,:,1] - cy) / H     # +Y *down*

        vertices_image[:,:,0] =  2.0 * (vertices_image[:,:,0]) / (W-1)  -1      # +X right
        vertices_image[:,:,1] = 2.0 * (vertices_image[:,:,1]) / (H-1) - 1      # +Y *down*

        # ---------------------------------------------------------------------------
        # keep Z untouched – it is copied later:
        vertices_image[:,:,2] = vertices_depth
        # ---------------------------------------------------------------------------
        
        # Get vertices per face
        # faces = self.faces.unsqueeze(1).repeat(1,num_views,1,1).contiguous().view(batch_size*num_views, num_faces*3) # (batch_size*num_views, num_faces*3)
        faces = faces.contiguous().view(batch_size*num_views, num_faces*3) # (batch_size*num_views, num_faces*3)
        face_vertices_image = torch.gather(vertices_image, dim=1, index=faces[:,:].unsqueeze(-1).repeat(1,1,3)) # (batch_size*num_views, num_faces*3, 3)
        face_vertices_image = face_vertices_image.contiguous().view(-1, num_faces, 3, 3) # (batch_size*num_views, num_faces, 3, 3)

        # Render inverse depth images
        face_vertices_z = -face_vertices_image[:,:,:,-1]  # (batch_size, num_faces, 3)
        face_vertices_xy = torch.clone(face_vertices_image[:,:,:,:2]) # (batch_size, num_faces, 3, 2)
        face_vertices_xy[:,:,:,1] = -face_vertices_xy[:,:,:,1]
        
        # face = self.face_vertices_normals(vertices.transpose(1, 2)).contiguous().view(batch_size*num_views, num_faces, 3) # (batch_size*num_views, num_faces, 3)
        vertex_normals = self.vertex_normals(vertices.transpose(1, 2))
        face_vertex_normals = self.face_vertices(vertex_normals, self.faces.unsqueeze(0).repeat(batch_size,1,1).to(device)) # (batch_size*num_views, num_faces, 3)

        
        face_features = 1.0-face_vertices_image[:,:,:,-1].unsqueeze(-1) # (batch_size, num_faces, 3, feature_dim)
        # print(face_features.shape)
        face_features = face_vertex_normals.repeat(batch_size*num_views,1,1,1)
        # print(face_features.shape)

        rendered_images, _ = kaolin.render.mesh.rasterize(depth_rendering_height, depth_rendering_width, face_vertices_z, face_vertices_xy, face_features)   

        # rendered_images = rendered_images.contiguous().view(batch_size*num_views, depth_rendering_size, depth_rendering_size)  # (batch_size*num_views, image_size, image_size)
        
        # Sample depth images for every projected vertex
        # img_grid = vertices_image[:,:,:2] # (batch_size*num_views, num_vertices, 2)
        # img_grid = img_grid.unsqueeze(2) # (batch_size*num_views, num_vertices, 1, 2)
        # sampled_image_inv_depths = F.grid_sample(rendered_images.view(-1, 1, depth_rendering_size, depth_rendering_size), img_grid, padding_mode='zeros', align_corners=False)    # (batch_size*num_views, 1, 1, num_vertices)
        # sampled_image_inv_depths = sampled_image_inv_depths.contiguous().view(-1, num_vertices) # (batch_size*num_views, num_vertices)

        # # Compute difference in depth between each sampled, projected vertex and the vertex.
        # # If the sampled depth of the projected point is lower than the vertex depth, it is occluded.
        # vertex_visibility_mask = vertices_depth-depth_eps <= 1.0-sampled_image_inv_depths
        # vertex_visibility = vertex_visibility_mask.float()
        # vertex_visibility = vertex_visibility.contiguous().view(batch_size, num_views, num_vertices)

        return rendered_images

    def depth_vertex_visibility(self, vertices, camera_intrinsics, camera_extrinsics, depth_eps=3e-3, eps=1e-7, depth_rendering_size=200):
        '''
        Compute the vertex visibility with a thresholded depth rendering check. 

        Args:
            vertices: (B, L, 3), batched vertices
            camera_intrinsics: (B, V, 3, 3), batched camera intrinsics for all V views
            camera_extrinsics: (B, V, 3, 4), batched camera extrinsics for all V views
        Returns:
            visibilities: (B, V, L) batched binary vertex visibilities, with 0 = invisible, 1 = visible
        '''

        device = vertices.device
        batch_size, num_vertices, _ = vertices.shape
        num_faces, _ = self.faces.shape
        _, num_views, _, _ = camera_extrinsics.shape

        faces = self.faces.unsqueeze(0).repeat(batch_size*num_views,1,1).to(device) # (batch_size*num_views, num_faces, 3)
        camera_intrinsics = camera_intrinsics.contiguous().view(-1, 3, 3)   # (batch_size*num_views, 3, 3)
        camera_extrinsics = camera_extrinsics.contiguous().view(-1, 3, 4)   # (batch_size*num_views, 3, 4)

        vertices = vertices.transpose(1, 2) # (batch_size, 3, num_vertices)
        ones = torch.ones(batch_size, 1, num_vertices).to(device)   # (batch_size, 1, num_vertices)
        vertices_homogeneous = torch.cat((vertices, ones), axis=-2) # (batch_size, 4, num_vertices)

        vertices_homogeneous = vertices_homogeneous.unsqueeze(1).repeat(1,num_views,1,1).contiguous()    # (batch_size, num_views, 4, num_vertices)
        vertices_homogeneous = vertices_homogeneous.view(-1, 4, num_vertices)   # (batch_size*num_views, 4, num_vertices)

        # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
        vertices_image = camera_extrinsics.bmm(vertices_homogeneous)    # (batch_size*num_views, 3, num_vertices)
       
        vertices_depth = torch.clone(vertices_image[:,2,:])             # (batch_size*num_views, num_vertices)       
        min_depth, max_depth = torch.floor(vertices_depth.min(dim=-1).values), torch.ceil(vertices_depth.max(dim=-1).values)
        vertices_depth = (vertices_depth-min_depth.unsqueeze(-1))/(max_depth.unsqueeze(-1)-min_depth.unsqueeze(-1))

        # Transformation from 3D camera coordinate system to the undistorted image plane 
        mask = (vertices_image.abs() < eps)
        mask[:,:2,:] = False
        vertices_image[mask] = 1.0 # Avoid division by zero

        vertices_image[:,0,:] = vertices_image[:,0,:] / vertices_image[:,2,:]  
        vertices_image[:,1,:] = vertices_image[:,1,:] / vertices_image[:,2,:]  
        vertices_image[:,2,:] = 1.0

        # Transformation from image coordinates to the final image coordinates with the camera intrinsics
        vertices_image = camera_intrinsics.bmm(vertices_image)      # (batch_size*num_views, 3, num_vertices)    
        vertices_image = vertices_image.transpose(1,2)              # (batch_size*num_views, num_vertices, 3)
        vertices_image[:,:,2] = vertices_depth

        # Normalize image coordinates to [-1,1]
        x_min, x_max = torch.floor(vertices_image[:,:,0].min(dim=1).values), torch.ceil(vertices_image[:,:,0].max(dim=1).values)
        y_min, y_max = torch.floor(vertices_image[:,:,1].min(dim=1).values), torch.ceil(vertices_image[:,:,1].max(dim=1).values)
        x_center, y_center = (x_max+x_min)/2.0, (y_max+y_min)/2.0
        normalizing_scale = 1.0/torch.max(x_max-x_min, y_max-y_min)
        vertices_image[:,:,0] = 2*normalizing_scale.unsqueeze(-1)*(vertices_image[:,:,0]-x_center.unsqueeze(-1))
        vertices_image[:,:,1] = 2*normalizing_scale.unsqueeze(-1)*(vertices_image[:,:,1]-y_center.unsqueeze(-1))

        # Get vertices per face
        # faces = self.faces.unsqueeze(1).repeat(1,num_views,1,1).contiguous().view(batch_size*num_views, num_faces*3) # (batch_size*num_views, num_faces*3)
        faces = faces.contiguous().view(batch_size*num_views, num_faces*3) # (batch_size*num_views, num_faces*3)
        face_vertices_image = torch.gather(vertices_image, dim=1, index=faces[:,:].unsqueeze(-1).repeat(1,1,3)) # (batch_size*num_views, num_faces*3, 3)
        face_vertices_image = face_vertices_image.contiguous().view(-1, num_faces, 3, 3) # (batch_size*num_views, num_faces, 3, 3)

        # Render inverse depth images
        face_vertices_z = -face_vertices_image[:,:,:,-1]  # (batch_size, num_faces, 3)
        face_vertices_xy = torch.clone(face_vertices_image[:,:,:,:2]) # (batch_size, num_faces, 3, 2)
        face_vertices_xy[:,:,:,1] = -face_vertices_xy[:,:,:,1]
        face_features = 1.0-face_vertices_image[:,:,:,-1].unsqueeze(-1) # (batch_size, num_faces, 3, feature_dim)

        rendered_images, _ = kaolin.render.mesh.rasterize(depth_rendering_size, depth_rendering_size, face_vertices_z, face_vertices_xy, face_features)   
        rendered_images = rendered_images.contiguous().view(batch_size*num_views, depth_rendering_size, depth_rendering_size)  # (batch_size*num_views, image_size, image_size)

        # Sample depth images for every projected vertex
        img_grid = vertices_image[:,:,:2] # (batch_size*num_views, num_vertices, 2)
        img_grid = img_grid.unsqueeze(2) # (batch_size*num_views, num_vertices, 1, 2)
        sampled_image_inv_depths = F.grid_sample(rendered_images.view(-1, 1, depth_rendering_size, depth_rendering_size), img_grid, padding_mode='zeros', align_corners=False)    # (batch_size*num_views, 1, 1, num_vertices)
        sampled_image_inv_depths = sampled_image_inv_depths.contiguous().view(-1, num_vertices) # (batch_size*num_views, num_vertices)

        # Compute difference in depth between each sampled, projected vertex and the vertex.
        # If the sampled depth of the projected point is lower than the vertex depth, it is occluded.
        vertex_visibility_mask = vertices_depth-depth_eps <= 1.0-sampled_image_inv_depths
        vertex_visibility = vertex_visibility_mask.float()
        vertex_visibility = vertex_visibility.contiguous().view(batch_size, num_views, num_vertices)

        return vertex_visibility

    @staticmethod
    def vertex_by_face(num_vertices, faces):
        row = faces.flatten()
        col = np.array([range(faces.shape[0])] * 3).T.flatten()
        indices = np.stack((row, col)).tolist()
        data = [1.0] * col.shape[0]
        return torch.sparse_coo_tensor(indices, data, (num_vertices, faces.shape[0]),)


    def render_normals_and_depth(self, vertices, camera_intrinsics, camera_extrinsics, radial_distortions=None, depth_eps=3e-3, eps=1e-7, depth_rendering_height=200, depth_rendering_width=200,
                         return_depth=False, segmentation_masks=None, render_point_map=False, needs_projection=True, normalize_normals=False, **kwargs):

        device = vertices.device
        if needs_projection:
            batch_size, num_vertices, _ = vertices.shape
            num_faces, _ = self.faces.shape
            _, num_views, _, _ = camera_extrinsics.shape

            faces = self.faces.unsqueeze(0).repeat(batch_size*num_views,1,1).to(device) # (batch_size*num_views, num_faces, 3)
            camera_intrinsics = camera_intrinsics.contiguous().view(-1, 3, 3)   # (batch_size*num_views, 3, 3)
            camera_extrinsics = camera_extrinsics.contiguous().view(-1, 3, 4)   # (batch_size*num_views, 3, 4)

            verts_world = vertices.clone()
            vertices = vertices.transpose(1, 2) # (batch_size, 3, num_vertices)
            ones = torch.ones(batch_size, 1, num_vertices).to(device)   # (batch_size, 1, num_vertices)
            vertices_homogeneous = torch.cat((vertices, ones), axis=-2) # (batch_size, 4, num_vertices)

            vertices_homogeneous = vertices_homogeneous.unsqueeze(1).repeat(1,num_views,1,1).contiguous()    # (batch_size, num_views, 4, num_vertices)
            vertices_homogeneous = vertices_homogeneous.view(-1, 4, num_vertices)   # (batch_size*num_views, 4, num_vertices)

            # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
            # vertices_image is (batch_size*num_views, 3, num_vertices)
            vertices_camera_coords = camera_extrinsics.bmm(vertices_homogeneous)
            # vertices_image_rotated_3d = camera_extrinsics[:,:3,:3].bmm(vertices_homogeneous[:,:3,:]) # (batch_size*num_views, 3, num_vertices)
        else:
            batch_size, num_views, num_vertices, _ = vertices.shape
            num_faces, _ = self.faces.shape
            
            faces = self.faces.unsqueeze(0).repeat(batch_size*num_views,1,1).to(device)
            camera_intrinsics = camera_intrinsics.contiguous().view(-1, 3, 3)   # (batch_size*num_views, 3, 3)
            camera_extrinsics = camera_extrinsics.contiguous().view(-1, 3, 4)   # (batch_size*num_views, 3, 4)
            # vertices_image_rotated_3d = vertices.transpose(1, 2) # (batch_size*num_views, 3, num_vertices)
            vertices_camera_coords = vertices.view(-1, num_vertices, 3) # (batch_size*num_views, 3, num_vertices)
            vertices_camera_coords = vertices_camera_coords.transpose(1, 2) # (batch_size*num_views, num_vertices, 3)

        # Store original depth values (Z-component)
        # vertices_depth is (batch_size*num_views, num_vertices)
        vertices_depth = torch.clone(vertices_camera_coords[:,2,:])             

        # Get components for division from vertices_camera_coords
        X_cam = vertices_camera_coords[:,0,:] # (B*V, Nv)
        Y_cam = vertices_camera_coords[:,1,:] # (B*V, Nv)
        Z_cam = vertices_camera_coords[:,2,:] # (B*V, Nv)

        # Avoid division by zero by clamping Z_cam to a minimum 'eps' value
        # This is differentiable.
        Z_cam_safe = Z_cam.clamp(min=eps)

        # Calculate normalized X and Y coordinates (non-in-place)
        X_normalized = X_cam / Z_cam_safe
        Y_normalized = Y_cam / Z_cam_safe
        
        # Z-component becomes 1.0 for projection onto the undistorted image plane
        Z_normalized = torch.ones_like(Z_cam)

        # Reconstruct the points in the normalized image plane (non-in-place)
        # This tensor now holds the (x/z, y/z, 1.0) coordinates
        # `stack` along dim=1 to get (B*V, 3, Nv)
        points_in_normalized_image_plane = torch.stack([X_normalized, Y_normalized, Z_normalized], dim=1) 

        # Apply radial distortions if enabled
        if radial_distortions is not None:
            radial = radial_distortions.contiguous().view(-1, 2)            # (B*V,2)
            k1 = radial[:, 0].unsqueeze(-1)                     # (B*V,1)
            k2 = radial[:, 1].unsqueeze(-1)                     # (B*V,1)

            # Get X and Y components from the current points_in_normalized_image_plane
            # (B*V, 3, Nv) -> indices `[0]` and `[1]`
            x = points_in_normalized_image_plane[:, 0, :]                         # (B*V,Nv)
            y = points_in_normalized_image_plane[:, 1, :]                         # (B*V,Nv)
            z_val = points_in_normalized_image_plane[:, 2, :]                     # (B*V,Nv), which is 1.0

            r2 = x.pow(2) + y.pow(2)
            radial_factor = 1 + k1 * r2 + k2 * r2.pow(2)

            # Compute new distorted x and y values (non-in-place)
            x_distorted = x * radial_factor
            y_distorted = y * radial_factor

            # Reconstruct the points_in_normalized_image_plane after distortion (non-in-place)
            points_in_normalized_image_plane = torch.stack([x_distorted, y_distorted, z_val], dim=1)

        # Transformation from undistorted image coordinates to the final image coordinates with camera intrinsics
        # points_in_normalized_image_plane is (B*V, 3, Nv)
        # camera_intrinsics is (B*V, 3, 3)
        vertices_image_final = camera_intrinsics.bmm(points_in_normalized_image_plane) # (B*V, 3, Nv)
        
        # Transpose to (B*V, Nv, 3) for consistency with typical (x,y,z) last dimension
        vertices_image_final = vertices_image_final.transpose(1,2)              # (batch_size*num_views, num_vertices, 3)

        # Now, replace the Z-coordinate with the actual depth (`vertices_depth`)
        # This is the final 3D point in image space (X_pixel, Y_pixel, Depth_camera_coords)
        # Avoid in-place modification of `vertices_image_final`
        final_X_pixel = vertices_image_final[:, :, 0]
        final_Y_pixel = vertices_image_final[:, :, 1]
        
        # Reconstruct the `vertices_image` (main variable name from original code) with correct components
        vertices_image = torch.stack([final_X_pixel, final_Y_pixel, vertices_depth], dim=2) # (B*V, Nv, 3)

        # Normalize image coordinates to [-1,1] for Kaolin rasterizer
        Bv, Nv, _ = vertices_image.shape
        W, H = depth_rendering_width, depth_rendering_height           # your target res

        # Calculate normalized screen coordinates (non-in-place)
        screen_X = 2.0 * (vertices_image[:,:,0]) / (W-1) - 1      # +X right
        screen_Y = 2.0 * (vertices_image[:,:,1]) / (H-1) - 1      # +Y *down*
        screen_Z = vertices_image[:,:,2] # The depth remains the same as `vertices_depth`

        # Reconstruct the vertices_image with screen coordinates (non-in-place)
        vertices_image = torch.stack([screen_X, screen_Y, screen_Z], dim=2) # (B*V, Nv, 3)

        # Get vertices per face
        faces = faces.contiguous().view(batch_size*num_views, num_faces*3) # (batch_size*num_views, num_faces*3)
        # `torch.gather` is differentiable
        face_vertices_image = torch.gather(vertices_image, dim=1, index=faces[:,:].unsqueeze(-1).repeat(1,1,3)) # (batch_size*num_views, num_faces*3, 3)
        face_vertices_image = face_vertices_image.contiguous().view(-1, num_faces, 3, 3) # (batch_size*num_views, num_faces, 3, 3)


        # Face vertices Z-component (for depth in rasterizer)
        face_vertices_z = -face_vertices_image[:,:,:,-1]  # (batch_size*num_views, num_faces, 3)

        # Face vertices XY-components (for screen position in rasterizer)
        face_vertices_xy_orig = face_vertices_image[:,:,:,:2] # (batch_size*num_views, num_faces, 3, 2)
        
        # Invert Y for screen coordinates (non-in-place)
        # `face_vertices_xy_orig` is already a clone of a slice, but modifying it in-place is still bad.
        face_vertices_xy = torch.stack([
            face_vertices_xy_orig[:,:,:,0],      # X-component
            -face_vertices_xy_orig[:,:,:,1]      # Inverted Y-component (non-in-place)
        ], dim=3) # (batch_size*num_views, num_faces, 3, 2)

        # Get vertex normals and prepare face features
        # vertex_normals = self.vertex_normals(vertices_image_rotated_3d.transpose(1, 2)) # (batch_size*num_views, num_vertices, 3)
        vertex_normals = self.vertex_normals(vertices_camera_coords.transpose(1, 2)) # (batch_size*num_views, num_vertices, 3)
        
        # Invert specific axes of normals as per comment (non-in-place)
        normals_X = vertex_normals[..., 0]
        normals_Y = -vertex_normals[..., 1] # Non-in-place inversion
        normals_Z = -vertex_normals[..., 2] # Non-in-place inversion
        vertex_normals_transformed = torch.stack([normals_X, normals_Y, normals_Z], dim=-1)
            

        face_vertex_normals = self.face_vertices(vertex_normals_transformed, self.faces.unsqueeze(0).repeat(batch_size*num_views,1,1).to(device)) # (batch_size*num_views, num_faces, 3, 3)
        
        # Normalize normals to [0, 1] range for visualization/rendering as color
        face_features = face_vertex_normals * 0.5 + 0.5

        # --- append σ (or log-σ² / precision) if provided -------------
        if 'vertex_uncertainty' in kwargs and kwargs['vertex_uncertainty'] is not None:
            v_sigma = kwargs['vertex_uncertainty']               # (B,Nv,1)
            v_sigma = v_sigma.unsqueeze(1).repeat(1, num_views, 1, 1).view(-1, num_vertices, 1) # (B*V,Nv,1)

            f = self.faces.unsqueeze(0).repeat(batch_size*num_views, 1, 1).to(device)

            bs, nv = v_sigma.shape[:2]
            bs, nf = f.shape[:2]
            device = v_sigma.device
            f = f + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
            v_sigma = v_sigma.reshape((bs * nv, 1))
            fv_sigma = v_sigma[f.long()]

            face_features = torch.cat((face_features, fv_sigma), dim=-1)

        if return_depth:
            # print(-face_vertices_z.min(), -face_vertices_z.max())
            face_features = torch.cat((face_features, -face_vertices_z.unsqueeze(-1)), dim=-1) # (batch_size*num_views, num_faces, 3, 4)


        if render_point_map:
            # world‑space XYZ as colour‑like features
            v_world = verts_world.unsqueeze(1).repeat(1, num_views, 1, 1).view(-1, num_vertices, 3)
            fv_xyz  = self.face_vertices(v_world, faces)                # (B*V,F,3,3)
            face_features = torch.cat((face_features, fv_xyz), dim=-1)  # (B*V,F,3,6)


        # if segmentation_masks is not None:
        #     visualize_seg_mask = kwargs.get('visualize_seg_mask', True)

        #     # ------------- vertex IDs ----------------------------------
        #     # segmentation_masks is (B, Nv)  [int64]
        #     print(segmentation_masks.shape)
        #     seg_ids = segmentation_masks.to(device).long()                # (B, Nv)
        #     seg_ids = seg_ids.unsqueeze(1).repeat(1, num_views, 1)        # (B, V, Nv)
        #     seg_ids = seg_ids.view(-1, num_vertices)                      # (B*V, Nv)

        #     print(seg_ids.shape)

        #     # gather per-face-vertex IDs   (B*V, F, 3, 1)
        #     face_vertex_ids = torch.gather(
        #         seg_ids, 1, faces.unsqueeze(-1)
        #     ).view(-1, num_faces, 3, 1).float()      # float for rasteriser

        #     # ------------- rasterise ID image --------------------------
        #     id_render, _, _ = kaolin.render.mesh.dibr_rasterization(
        #         depth_rendering_height, depth_rendering_width,
        #         face_vertices_z, face_vertices_xy, face_vertex_ids,
        #         face_normals_z=face_normals_z
        #     )
        #     # id_render … (B*V, H, W, 1)
        #     id_image = id_render[..., 0].round().clamp(0, _MAX_ID).long()

        #     face_features = torch.cat((face_features, id_image.unsqueeze(-1)), dim=-1) # (B*V, F, 3, 4)

        # --- attach per-face-vertex class IDs as one more feature channel --------------
        if segmentation_masks is not None:
            # segmentation_masks: (B, Nv) with integer 0/1 labels
            seg_ids = segmentation_masks.to(device).long()                # (B, Nv)
            seg_ids = seg_ids.unsqueeze(1).repeat(1, num_views, 1)        # (B, V, Nv)
            seg_ids = seg_ids.view(-1, num_vertices)                      # (B*V, Nv)

            # gather IDs per face-vertex, shape (B*V, F, 3, 1), float for rasterization
            face_vertex_ids = torch.gather(seg_ids, 1, faces).view(-1, num_faces, 3, 1).float()

            # concatenate as the LAST feature channel so we can split it out later
            face_features = torch.cat((face_features, face_vertex_ids), dim=-1)
   

        # ------------- camera-space face normal (for culling) -----------------
        # Grab the vertices in CAMERA coordinates (before any axis flips):
        v_cam      = vertices_camera_coords.transpose(1,2)          # B*V × Nv × 3
        faces_flat = faces.view(batch_size*num_views, -1)              # B*V × F*3
        v0, v1, v2 = torch.gather(v_cam, 1, faces_flat[:,0::3].unsqueeze(-1).repeat(1,1,3)), \
                    torch.gather(v_cam, 1, faces_flat[:,1::3].unsqueeze(-1).repeat(1,1,3)), \
                    torch.gather(v_cam, 1, faces_flat[:,2::3].unsqueeze(-1).repeat(1,1,3))

        v0, v1, v2 = v0.view(-1, num_faces, 3), v1.view(-1, num_faces, 3), v2.view(-1, num_faces, 3)
        face_normals_cam = torch.nn.functional.normalize(
                torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
        # Kaolin’s camera looks toward −Z, so front faces satisfy  n·(0,0,−1) < 0:
        face_normals_z = -(face_normals_cam[..., 2])   # B*V × F   ( >0 ⇒ front )


        # ------------- rasterise with culling ---------------------------------
        rendered, mask, _ = kaolin.render.mesh.dibr_rasterization(
                depth_rendering_height, depth_rendering_width,
                face_vertices_z, face_vertices_xy, face_features,
                face_normals_z=face_normals_z)


        # split them out -------------------------------------------------
        c = 3                               # first 3 are normals
        out = {'normal_images': rendered[..., :c]}

        if 'vertex_uncertainty' in kwargs and kwargs['vertex_uncertainty'] is not None:

            # mask is (B*V, H, W), so unsqueeze to broadcast:
            # mask4 = mask.unsqueeze(-1).float()                   # (B*V, H, W, 1)

            # wherever mask==1 keep rendered uncertainty, wherever mask==0 set to 1:
            # uncertainty = uncertainty * mask4 + (1.0 - mask4)    # fill background with 1
            out['uncertainty_images'] = rendered[..., c:c+1] # * mask4 + -100*(1 - mask4)  # fill background with 1
            c += 1

        if segmentation_masks is not None:
            out['segmentation_images'] = rendered[..., c:c+1]
            c += 1

        if return_depth:
            out['depth_images'] = rendered[..., c:c+1]

        return out




def depth_to_pointmap_robust(
    depth,                  # (B, V, H, W)  –  depth in metric units
    K,                      # (B, V, 3, 3)
    extr,                   # (B, V, 3, 4)  –  world→camera  (R|t)
    extr_ref=None,          # (B, 1, 3, 4)  –  world→refCam (R|t), or None
    rotated_views=None,
):
    """
    Returns a dense point-map  (B, V, H, W, 3)
    expressed either
    • in *world* coordinates  (extr_ref is None), or
    • in the *refCam* frame   (extr_ref supplied).

    Uses full K^{-1} back-projection, which correctly handles rotated-camera intrinsics.
    """
    import torch

    B, V, H, W = depth.shape
    device     = depth.device

    # 1) Pixel grid (u,v) with correct HxW layout
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')       # vv: y(row), uu: x(col)

    ones  = torch.ones_like(uu, dtype=torch.float32, device=device)
    pix_h = torch.stack((uu.float(), vv.float(), ones), dim=-1)         # (H,W,3)
    pix_h = pix_h.view(1, 1, H, W, 3).expand(B, V, -1, -1, -1)          # (B,V,H,W,3)

    # Use full K^{-1} to back-project rays for any valid K (including rotated views)
    K_inv = torch.linalg.inv(K)                                         # (B,V,3,3)
    dirs  = torch.einsum('bvhwc,bvcd->bvhwd', pix_h, K_inv)             # (B,V,H,W,3)

    # Normalize homogeneous scale and multiply by depth
    z  = depth.unsqueeze(-1)                                            # (B,V,H,W,1)
    dz = dirs[..., 2:3]
    dz_safe = torch.where(dz.abs() < 1e-8, torch.full_like(dz, 1e-8), dz)

    x = dirs[..., 0:1] / dz_safe * z
    y = dirs[..., 1:2] / dz_safe * z
    pts_cam = torch.cat((x, y, z), dim=-1)                              # (B,V,H,W,3)
    # 3) cam → world  (X_w = (x_c - t) @ R^T), with extr being world→cam (R|t)
    R   = extr[..., :3]                                # (B,V,3,3)
    t   = extr[..., 3:].unsqueeze(-2)                  # (B,V,1,3)

    pts_cam_flat = pts_cam.reshape(B*V, -1, 3)         # (B*V, HW, 3)
    R_t          = R.transpose(-1, -2).reshape(B*V, 3, 3)
    t_flat       = t.reshape(B*V, 1, 3)

    pts_world_flat = (pts_cam_flat - t_flat) @ R_t     # (B*V, HW, 3)
    pts_world      = pts_world_flat.view(B, V, H, W, 3)

    if extr_ref is None:
        return pts_world                               # in world frame

    # 4) world → refCam (x_ref = X_w @ R_ref^T + t_ref)
    Rr = extr_ref[..., :3]                             # (B,1,3,3)
    tr = extr_ref[..., 3:].unsqueeze(-2)               # (B,1,1,3)

    pts_ref = pts_world @ Rr.transpose(-1, -2) + tr    # (B,V,H,W,3)
    return pts_ref


# ---------------------------------------- XYZ ➋ → RGB ➌ ----------------------

def pointmap_to_rgb(pts, pts_min=None, pts_max=None):
    """
    Converts a 3D pointmap (3, H, W) to an RGB image (H, W, 3).

    Args:
        pts (np.ndarray): 3D point map of shape (3, H, W)

    Returns:
        np.ndarray: RGB image of shape (H, W, 3) scaled to 0–255 dtype=np.uint8
    """
    assert pts.shape[0] == 3, "Input must have 3 channels (X, Y, Z)"
    
    # Normalize each channel to 0–1
    if pts_min is None:
        pts_min = pts.min(axis=(1, 2), keepdims=True)
    if pts_max is None:
        pts_max = pts.max(axis=(1, 2), keepdims=True)

    normalized = (pts - pts_min) / \
                 (pts_max - pts_min + 1e-8)
    
    # Stack and convert to uint8
    rgb = (normalized * 255)
    rgb = np.clip(rgb, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    
    return rgb, pts_min, pts_max
