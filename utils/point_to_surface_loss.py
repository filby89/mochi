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
from torch import nn
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance

# def compute_s2m_distance(scan_vertices, predicted_vertices, predicted_faces, faces_masks, return_array=False):
#     mesh_face_vertices = index_vertices_by_faces(predicted_vertices, predicted_faces) # (batch_size, num_faces, num_vertices, 3)
#     distances, face_indices, _ = point_to_mesh_distance(scan_vertices, mesh_face_vertices)
#     if return_array:
#         return distances.abs().sqrt()
#     return distances.abs().sqrt().squeeze()

import torch

def batch_chamfer_distance(p1, p2, gmo_sigma=1.0):
    """
    Calculates the batch Chamfer distance between two point clouds.

    Args:
        p1 (torch.Tensor): A batch of point clouds with shape (B, N, D),
                           where B is the batch size, N is the number of points,
                           and D is the dimension of each point.
        p2 (torch.Tensor): A batch of point clouds with shape (B, M, D),
                           where M is the number of points.

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the Chamfer distance
                      for each pair of point clouds in the batch.
    """

    # randomly sample points from p2
    num_scan=20000
    if p2.shape[1] > num_scan:
        p2 = p2[:, torch.randperm(p2.shape[1])[:num_scan], :]


    robustifier = lambda x: GMO(x, sigma=gmo_sigma) if gmo_sigma > 1e-8 else x
    # Expand dimensions to facilitate broadcasting
    # p1_expanded shape: (B, N, 1, D)
    # p2_expanded shape: (B, 1, M, D)
    p1_expanded = p1.unsqueeze(2)
    p2_expanded = p2.unsqueeze(1)

    # Calculate the squared Euclidean distance between each point in p1 and p2
    # The subtraction is broadcasted, resulting in a shape of (B, N, M, D)
    # The sum along the last dimension (D) gives the squared distances
    # dist_matrix shape: (B, N, M)
    dist_matrix = torch.sum((p1_expanded - p2_expanded) ** 2, dim=3)

    # Find the minimum distance from each point in p1 to any point in p2
    # min_dist_p1 shape: (B, N)
    min_dist_p1, _ = torch.min(dist_matrix, dim=2)

    # Find the minimum distance from each point in p2 to any point in p1
    # min_dist_p2 shape: (B, M)
    min_dist_p2, _ = torch.min(dist_matrix, dim=1)

    # Calculate the mean of these minimum distances for each point cloud in the batch
    # cham_dist_p1 is the forward distance
    # cham_dist_p2 is the backward distance
    # print(min_dist_p1.shape, min_dist_p2.shape, min_dist_p1.mean(), min_dist_p2.mean())
    # raise
    # min_dist_p1 = robustifier(min_dist_p1)
    # min_dist_p2 = robustifier(min_dist_p2)

    # print("Chamfer distances:", min_dist_p1.shape, min_dist_p2.shape, min_dist_p1.mean(), min_dist_p2.mean())
    # raise

    # ---- trim the top `trim_ratio` fraction ---------------------------------
    def trimmed_mean(d, ratio):
        if ratio <= 0.0:
            return d.mean(dim=1)
        k = int((1.0 - ratio) * d.shape[1])
        topk, _ = torch.topk(d, k=k, largest=False, dim=1)
        return topk.mean(dim=1)

    cham_dist_p1 = trimmed_mean(min_dist_p1, 0.1)
    cham_dist_p2 = trimmed_mean(min_dist_p2, 0.1)

    # The Chamfer distance is the sum of the forward and backward distances
    cham_dist = cham_dist_p1 + cham_dist_p2

    return cham_dist


# def compute_s2m_distance(
#     scan_vertices,
#     predicted_vertices,
#     predicted_faces,
#     masks=None,
#     registration_vertices_for_regions=None,
#     return_region_info=False,
#     exclude_regions=None,
#     exclude_from_full=None,
# ):
#     mesh_face_vertices = index_vertices_by_faces(predicted_vertices, predicted_faces)
#     distances_sq, face_indices, _ = point_to_mesh_distance(scan_vertices, mesh_face_vertices)

#     # print(scan_vertices.shape, predicted_vertices.shape, predicted_faces.shape, mesh_face_vertices.shape, registration_vertices_for_regions.shape)
#     if registration_vertices_for_regions is not None:
#         # use registrations to calculate the region distances for fairness
#         mesh_registration_vertices = index_vertices_by_faces(registration_vertices_for_regions, predicted_faces)
#         _, face_indices, _ = point_to_mesh_distance(scan_vertices, mesh_registration_vertices)


#     all_distances = distances_sq.abs().sqrt().squeeze()

#     face_indices_flat = face_indices.squeeze()

#     region_labels = None
#     region_name_to_index = {}
#     if masks is not None and return_region_info:
#         region_labels = torch.full(
#             face_indices_flat.shape,
#             -1,
#             device=face_indices.device,
#             dtype=torch.long,
#         )

#     exclude_regions = set(exclude_regions or [])
#     exclude_from_full = list(exclude_from_full or [])


#     excluded_mask = torch.zeros(face_indices_flat.shape, device=face_indices_flat.device, dtype=torch.bool)
#     distances_dict = {}

#     if masks is not None:
#         for region_name, region_face_indices in masks.items():
#             if not isinstance(region_face_indices, torch.Tensor):
#                 region_face_indices = torch.tensor(region_face_indices, device=face_indices_flat.device, dtype=face_indices_flat.dtype)

#             is_in_region_mask = torch.isin(face_indices_flat, region_face_indices)

#             region_distances = all_distances[is_in_region_mask]

#             if region_name in exclude_regions:
#                 excluded_mask |= is_in_region_mask
#                 distances_dict[region_name] = region_distances.new_empty((0,))
#             else:
#                 distances_dict[region_name] = region_distances
#             if return_region_info:
#                 region_index = region_name_to_index.setdefault(region_name, len(region_name_to_index))
#                 region_labels[is_in_region_mask] = region_index

#     if excluded_mask.any():
#         valid_mask = ~excluded_mask
#         distances_dict["full"] = all_distances[valid_mask]
#     else:
#         distances_dict["full"] = all_distances

#     if return_region_info:
#         region_info = None
#         if masks is not None:
#             region_info = {
#                 "labels": region_labels,
#                 "name_to_index": region_name_to_index,
#             }
#         return distances_dict, region_info

#     return distances_dict
def compute_s2m_distance(
    scan_vertices,
    predicted_vertices,
    predicted_faces,
    masks=None,
    registration_vertices_for_regions=None,
    return_region_info=False,
    exclude_regions=None,
    exclude_from_full=None,
):
    # print(predicted_vertices.shape, predicted_faces.shape)
    mesh_face_vertices = index_vertices_by_faces(predicted_vertices, predicted_faces)
    distances_sq, face_indices, _ = point_to_mesh_distance(scan_vertices, mesh_face_vertices)

    # If provided, use registrations to (re)assign regions for fairness
    if registration_vertices_for_regions is not None:
        mesh_registration_vertices = index_vertices_by_faces(registration_vertices_for_regions, predicted_faces)
        _, face_indices, _ = point_to_mesh_distance(scan_vertices, mesh_registration_vertices)

    all_distances = distances_sq.abs().sqrt().squeeze()
    face_indices_flat = face_indices.squeeze()

    region_labels = None
    region_name_to_index = {}
    if masks is not None and return_region_info:
        region_labels = torch.full(
            face_indices_flat.shape,
            -1,
            device=face_indices.device,
            dtype=torch.long,
        )

    # Normalize inputs
    exclude_regions = set(exclude_regions or [])
    exclude_from_full = list(exclude_from_full or [])

    excluded_mask = torch.zeros(
        face_indices_flat.shape, device=face_indices_flat.device, dtype=torch.bool
    )
    distances_dict = {}

    # Region-wise buckets + track per-region masks for later "full_minus_*"
    region_to_vertexmask = {}

    if masks is not None:
        for region_name, region_face_indices in masks.items():
            if not isinstance(region_face_indices, torch.Tensor):
                region_face_indices = torch.tensor(
                    region_face_indices,
                    device=face_indices_flat.device,
                    dtype=face_indices_flat.dtype
                )

            is_in_region_mask = torch.isin(face_indices_flat, region_face_indices)
            region_to_vertexmask[region_name] = is_in_region_mask  # save for full_minus

            region_distances = all_distances[is_in_region_mask]

            if region_name in exclude_regions:
                excluded_mask |= is_in_region_mask
                # still populate an empty tensor for consistency
                distances_dict[region_name] = region_distances.new_empty((0,))
            else:
                distances_dict[region_name] = region_distances

            if return_region_info:
                region_index = region_name_to_index.setdefault(region_name, len(region_name_to_index))
                region_labels[is_in_region_mask] = region_index

    # "full" (respecting exclude_regions)
    if excluded_mask.any():
        valid_mask = ~excluded_mask
        distances_dict["full"] = all_distances[valid_mask]
    else:
        distances_dict["full"] = all_distances

    # "full_minus_<region>" for every entry in exclude_from_full
    # Each one excludes: (a) that specific region, and (b) anything in exclude_regions.
    if masks is not None and len(exclude_from_full) > 0:
        for region_name in exclude_from_full:
            # If region not found in masks, skip silently
            region_mask = region_to_vertexmask.get(region_name, None)
            if region_mask is None:
                continue
            mask_minus = ~region_mask  # remove this region
            if excluded_mask.any():
                mask_minus = mask_minus & (~excluded_mask)  # also honor globally excluded regions
            distances_dict[f"full_minus_{region_name}"] = all_distances[mask_minus]

    if return_region_info:
        region_info = None
        if masks is not None:
            region_info = {
                "labels": region_labels,
                "name_to_index": region_name_to_index,
            }
        return distances_dict, region_info

    return distances_dict

    
def GMO(sqr_input, sigma):
    '''
    Compute the square root of the Geman-McClure robustifier. As the input is assumed to be squared,
    the sign of the input is not considered. 
    '''
    eps = 1e-16
    sqr_sigma = sigma**2
    return torch.sqrt((sqr_sigma * (sqr_input / (sqr_sigma + sqr_input)))+eps)

class PointToSurfaceLoss(nn.Module):
    def __init__(self, gmo_sigma=1.0):
        super().__init__()
        
        if gmo_sigma <= 1e-8:
            self.robustifier = lambda x: x
        else:
            self.robustifier = lambda x: GMO(x, sigma=gmo_sigma)

    def forward(self, points, mesh_vertices, mesh_faces, return_distances=False, masks=None):
        '''
        Given a point and a mesh surface, compute the distance to the closest point in the mesh surface. 
        
        Args:
            points: (batch_size, num_points, 3)
            mesh_vertices: (batch_size, num_vertices, 3)
            mesh_faces: (num_faces, 3)
        '''

        batch_size, num_points, _ = points.shape
        mesh_face_vertices = index_vertices_by_faces(mesh_vertices, mesh_faces) # (batch_size, num_faces, num_vertices, 3)

        # Get closest mesh triangles for each point
        _, face_idx, _ = point_to_mesh_distance(points, mesh_face_vertices)
        closest_triangle_vertices = torch.gather(mesh_face_vertices, dim=1, index=face_idx[:, :, None, None].expand(-1, -1, 3, 3)).view(batch_size, num_points, 3, 3)

        # (Optional) Masking based on valid face indices per batch
        if masks is not None:
            valid_mask = torch.isin(face_idx, masks).int()


        # Compute barycentric embedding of every point into the closest triangle
        a, b, c = closest_triangle_vertices[:, :, 0, :], closest_triangle_vertices[:, :, 1, :], closest_triangle_vertices[:, :, 2, :]   # (batch_size, num_points, 3)
        bcoords = PointToSurfaceLoss.barycentric_coords(a, b, c, points)    # (batch_size, num_points, 3)

        closest_points = torch.multiply(a, bcoords[:, :, 0].unsqueeze(-1)) + \
                        torch.multiply(b, bcoords[:, :, 1].unsqueeze(-1)) + \
                        torch.multiply(c, bcoords[:, :, 2].unsqueeze(-1))   # (batch_size, num_points, 3)
        square_distances = (closest_points - points).square().sum(-1)
        robust_distances = self.robustifier(square_distances)
        # distance_loss = robust_distances.mean(-1).mean()

        if masks is not None:
            masked_robust_distances = robust_distances*valid_mask
        else:
            masked_robust_distances = robust_distances

        distance_loss = masked_robust_distances.mean() if masked_robust_distances.numel() > 0 else torch.tensor(0.0, device=points.device, requires_grad=True)

        if return_distances:
            return distance_loss, masked_robust_distances
        else:
            return distance_loss

    @staticmethod
    def barycentric_coords(a, b, c, q, eps=1e-16):
        '''
        Compute Barycentric coordinates of q projected on the triangle spanned by the vertices a, b, and c
        '''

        v1 = b-a
        v2 = c-a
        n = torch.cross(v1, v2)
        n_dot_n = torch.sum(n*n, dim=-1)
        n_dot_n[n_dot_n < eps] = 1.0
        
        w = q - a
        gamma = torch.sum(torch.cross(v1, w)*n, dim=-1)/n_dot_n
        beta = torch.sum(torch.cross(w, v2)*n, dim=-1)/n_dot_n
        alpha = 1.0-beta-gamma
        return torch.stack((alpha, beta, gamma), dim=-1)
