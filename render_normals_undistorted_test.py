import os
import sys
import shutil
import argparse
import numpy as np
import imageio
from datasets.face_align_dataset_mpi import FaceAlignDatasetMPI
from utils.mesh_helper import MeshHelper
import utils.mesh_renderer as mesh_renderer
import torch
import cv2
import json
from tqdm import tqdm
# these are the indices of the mediapipe landmarks that correspond to the mediapipe landmark barycentric coordinates provided by FLAME2020
# mediapipe_indices = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
#         55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
#        381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
#        144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
#        168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
#          0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
#         87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
#        308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
#        415]


# meshes_root = "/fast/pfilntisis/TEMPEH_data/data/training_data/meshes_new"
# in_mediapipe_path = "/fast/pfilntisis/TEMPEH_data/data/aws_data/downsampled_images_4_no_grid/downsampled_images_4_mediapipe_landmarks_fp32"
in_image_path = "/fast/pfilntisis/TEMPEH_data/data/aws_data/downsampled_images_4_no_grid/downsampled_images_4"
in_calibration_path = "/fast/pfilntisis/TEMPEH_data/data/aws_data/downsampled_images_4_no_grid/calibrations"
in_dense_path = "/fast/pfilntisis/blended_in_flames/evals/inference/lora-large-ep180/viz"

in_registration_path = "/fast/pfilntisis/TEMPEH_data/data/aws_data/registrations"
in_meshes_path = "/fast/pfilntisis/TEMPEH_data/data/aws_data/meshes_npz"


d = json.load(open("assets/meshes_list.json", 'r'))
start = sys.argv[1] if len(sys.argv) > 1 else 0
end = sys.argv[2] if len(sys.argv) > 2 else len(d)
if len(sys.argv) > 3:
    out_grid_root = sys.argv[3]
else:
    out_grid_root = "/fast/pfilntisis/TEMPEH_data/data/aws_data/undistorted_downsampled_4_culled_grid"

if len(sys.argv) > 4:
    undistort = bool(int(sys.argv[4]))
else:
    undistort = True


out_dir_normals = f"{out_grid_root}/color_normals"
out_dir_rgb = f"{out_grid_root}/color_images"
out_dir_depth = f"{out_grid_root}/color_depth"
out_dir_intrinsics = f"{out_grid_root}/color_cameras"
# out_dir_fan_landmarks = f"{out_grid_root}/fan_landmarks"
# out_dir_fan_landmarks_3D = f"{out_grid_root}/fan_landmarks_3D_v2"
# out_dir_mediapipe_landmarks = f"{out_grid_root}/mediapipe_landmarks"
out_dir_dense_landmarks = f"{out_grid_root}/color_dense_landmarks"


print('Processing meshes from', start, 'to', end, 'out dir:', out_grid_root, 'undistort:', undistort)

# # save tmp
json_out = f"assets/meshes_list_{start}_{end}.json"
with open(json_out, 'w') as f:
    json.dump(d[int(start):int(end)], f, indent=4)



# json_out = []
# for subject in os.listdir(in_meshes_path):
#     subject_path = os.path.join(in_meshes_path, subject)
#     if not os.path.isdir(subject_path):
#         continue
#     for sequence in os.listdir(subject_path):
#         sequence_path = os.path.join(subject_path, sequence)
#         if not os.path.isdir(sequence_path):
#             continue
#         for mesh in os.listdir(sequence_path):
#             mesh_path = os.path.join(sequence_path, mesh)
#             if not mesh.endswith('.npz'):
#                 continue

#             mesh_nr = mesh.split('.')[1]

#             image_path = f"{out_dir_rgb}/{subject}/{sequence}/{mesh_nr}/"

#             # print(len(os.listdir(image_path)), "images found in", image_path)
#             if os.path.exists(image_path) and len(os.listdir(image_path)) == 1:
#                 print(f"Skipping (already exists): {image_path}")
#                 continue

#             json_out.append([
#                 subject,
#                 sequence,
#                 mesh_nr
#             ])

# print(f"Found {len(json_out)} meshes to process.")
# # write to tmp
# json_path = "assets/meshes_list_test.json"
# with open(json_path, 'w') as f:
#     json.dump(json_out, f, indent=4)

# raise


# json_out = "assets/meshes_list_test.json"

# print(f"Processing meshes from {start} to {end}. Total: {len(d[int(start):int(end)])}")


dataset = FaceAlignDatasetMPI(data_list_fname=json_out,
                                # dataset_root_dir="/fast/pfilntisis/FaMoS",
                                image_dir=in_image_path,
                                calibration_dir=in_calibration_path,
                                # scan_dir=os.path.join(root, "sampled_scan_points"),
                                scan_dir=in_meshes_path,
                                registration_root_dir=in_registration_path,
                                image_file_ext='png', 
                                return_full_scan=True,
                                image_resize_factor=4,
                                undistort_images=undistort,
                                dense_landmarks_dir=in_dense_path,
                                )

suffix = "color"

vis_dir = "col45"
visualize = True
os.makedirs(out_dir_depth, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(out_dir_normals, exist_ok=True)
os.makedirs(out_dir_rgb, exist_ok=True)
# os.makedirs(out_dir_fan_landmarks, exist_ok=True)
# os.makedirs(out_dir_mediapipe_landmarks, exist_ok=True)
os.makedirs(out_dir_dense_landmarks, exist_ok=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
save_as_grid = True

for sample in tqdm(dataloader, desc="Rendering normals", total=len(dataloader)):
    subject, sequence, frame = sample['subject'][0], sample['sequence'][0], sample['frame'][0]
    camera_names = sample['camera_names']

    try:
        vertices = sample['v_scan'][0].numpy()
        faces = sample['f_scan'][0].numpy() #.cpu().numpy()
        registration_v = sample['v_registration']
        registration_f = sample['f_registration']
        
        # print(sample.keys())
        mesh_helper = MeshHelper(num_vertices=vertices.shape[0], faces=faces)

        vertices = torch.from_numpy(vertices).unsqueeze(0).float().cuda()

        camera_intrinsics = sample[f'{suffix}_camera_intrinsics'][0].unsqueeze(0).cuda()
        camera_extrinsics = sample[f'{suffix}_camera_extrinsics'][0].unsqueeze(0).cuda()
        radial_distortions = sample[f'{suffix}_camera_distortions'][0].unsqueeze(0).cuda()
        camera_centers = sample[f'{suffix}_camera_centers'][0].unsqueeze(0).cuda()

        # fan_landmarks = sample['color_camera_landmarks'][0].cpu().numpy()
        # mediapipe_landmarks = sample['color_camera_mediapipe_landmarks'][0].cpu().numpy()

        # mediapipe_landmarks = mediapipe_landmarks[..., mediapipe_indices, :]

        dense_landmarks = sample[f'{suffix}_camera_dense_landmarks'][0].cpu().numpy()
        # print(dense_landmarks.shape)

        color_images = sample[f'{suffix}_images'][0]
        print(radial_distortions)

        render_out = mesh_helper.render_normals_and_depth(vertices, camera_intrinsics, camera_extrinsics, radial_distortions=radial_distortions,
                                        depth_rendering_height=color_images[0].shape[1],
                                        depth_rendering_width=color_images[0].shape[2],
                                        return_depth=True)
        v, d = render_out['normal_images'], render_out['depth_images']

        # print("Color images shape:", color_images.shape, "v shape:", v.shape)
        if visualize:
            color_images = color_images.cpu().numpy()
            color_images_new = []
            for i in range(len(color_images)):
                color_image_new = dataset.denormalize_image(color_images[i].transpose(1, 2, 0))
                color_image_new = (color_image_new * 255).astype(np.uint8).copy()  # Convert to uint8 for saving
                # landmarks = fan_landmarks[i]
                # for landmark in landmarks:
                #     cv2.circle(color_image_new, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

                # # for landmark in mediapipe_landmarks[i]:
                #     cv2.circle(color_image_new, (int(landmark[0]), int(landmark[1])), 2, (255, 0, 0), -1)
                for landmark in dense_landmarks[i]:
                    cv2.circle(color_image_new, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), -1)
                    
                color_images_new.append(color_image_new)
            # stack color images horizontally
            color_images = np.hstack(color_images_new)



            v = np.hstack([v[i].cpu().numpy() for i in range(v.shape[0])])
            v = v * 255  
            v = v.astype(np.uint8) 

            d = np.hstack([d[i].cpu().numpy() for i in range(d.shape[0])])
            # normalize between 500 and 2000
            d = (d - 500) / (2000 - 500)
            d = (d * 255).astype(np.uint8)  # Convert to uint8 for saving

            # make 3 channel
            d = np.concatenate([d,d,d], axis=-1)

            overlay = cv2.addWeighted(color_images, 0.5, v, 0.5, 0)

            grid = cv2.vconcat([
                color_images,
                v,
                d,
                overlay
            ])
            imageio.imwrite(f"{vis_dir}/{subject}_{sequence}_{frame}.jpg", grid)
        else:
            # v = v.cpu()
            # color_images = color_images.cpu()
            # d = d.cpu()
            
            if save_as_grid:
                color_images = np.hstack([color_images[i].cpu().numpy().transpose(1, 2, 0) for i in range(len(color_images))])
                color_images = dataset.denormalize_image(color_images)  # Denormalize images to [0, 1] range
                color_images = (color_images * 255).astype(np.uint8)  # Convert to uint8 for saving
                v = np.hstack([v[i].cpu().numpy() for i in range(v.shape[0])])
                v = v * 255  
                v = v.astype(np.uint8) 

                d = np.hstack([d[i].cpu().numpy() for i in range(d.shape[0])])

                os.makedirs(f"{out_dir_normals}/{subject}/{sequence}/{frame}", exist_ok=True)
                os.makedirs(f"{out_dir_rgb}/{subject}/{sequence}/{frame}", exist_ok=True)
                os.makedirs(f"{out_dir_depth}/{subject}/{sequence}/{frame}", exist_ok=True)
                
                imageio.imwrite(f"{out_dir_normals}/{subject}/{sequence}/{frame}/{sequence}.{frame}.png", v)
                imageio.imwrite(f"{out_dir_rgb}/{subject}/{sequence}/{frame}/{sequence}.{frame}.png", color_images)
                np.save(f"{out_dir_depth}/{subject}/{sequence}/{frame}/{sequence}.{frame}.npy", d)
                os.makedirs(f"{out_dir_intrinsics}/{subject}/{sequence}/{frame}", exist_ok=True)

                intrinsics = camera_intrinsics[0].cpu().numpy()
                extrinsics = camera_extrinsics[0].cpu().numpy()
                centers = camera_centers[0].cpu().numpy()
                radial_distortions = radial_distortions[0].cpu().numpy()
                print(intrinsics.shape, extrinsics.shape, centers.shape, radial_distortions.shape)
                # assert radial_distortions.sum() == 0 
                assert intrinsics.shape == (16, 3, 3)
                assert extrinsics.shape == (16, 3, 4)
                assert centers.shape == (16, 3)
                assert radial_distortions.shape == (16, 2)
                print(centers)
                np.savez(f"{out_dir_intrinsics}/{subject}/{sequence}/{frame}/{sequence}.{frame}_intrinsics.npz", 
                    intrinsics=intrinsics, 
                    extrinsics=extrinsics, 
                    centers=centers, 
                    radial_distortions=radial_distortions)
                # landmarks now
                # print(fan_landmarks.shape)
                # os.makedirs(f"{out_dir_fan_landmarks}/{subject}/{sequence}/{frame}", exist_ok=True)
                # np.save(f"{out_dir_fan_landmarks}/{subject}/{sequence}/{frame}/{sequence}.{frame}.npy", fan_landmarks)
                # print('Saved:', f"{out_dir_fan_landmarks}/{subject}/{sequence}/{frame}/{sequence}.{frame}.npy")

                # print(mediapipe_landmarks.shape)
                # os.makedirs(f"{out_dir_mediapipe_landmarks}/{subject}/{sequence}/{frame}", exist_ok=True)
                # np.save(f"{out_dir_mediapipe_landmarks}/{subject}/{sequence}/{frame}/{sequence}.{frame}.npy", mediapipe_landmarks)
                # print('Saved:', f"{out_dir_mediapipe_landmarks}/{subject}/{sequence}/{frame}/{sequence}.{frame}.npy")

                # print(dense_landmarks.shape)
                os.makedirs(f"{out_dir_dense_landmarks}/{subject}/{sequence}/{frame}", exist_ok=True)
                np.save(f"{out_dir_dense_landmarks}/{subject}/{sequence}/{frame}/{sequence}.{frame}.npy", dense_landmarks)
                print('Saved:', f"{out_dir_dense_landmarks}/{subject}/{sequence}/{frame}/{sequence}.{frame}.npy")

            else:
                for i in range(v.shape[0]):
                    v_i = v[i].cpu().numpy()
                    v_i = (v_i * 255).astype(np.uint8)
                    camera_name = camera_names[i][0]

                    color_image = color_images[i].cpu().numpy().transpose(1, 2, 0)
                    color_image = dataset.denormalize_image(color_image)  # Denormalize images to [0, 1] range
                    color_image = (color_image * 255).astype(np.uint8)  # Convert to uint8 for saving
                    # grid = np.hstack([color_image, v_i])
                    os.makedirs(f"{out_dir_normals}/{subject}/{sequence}/{frame}", exist_ok=True)
                    imageio.imwrite(f"{out_dir_normals}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}.png", v_i)
                    # print('Saved:', f"{out_dir_normals}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}.png")

                    # save color image
                    os.makedirs(f"{out_dir_rgb}/{subject}/{sequence}/{frame}", exist_ok=True)
                    imageio.imwrite(f"{out_dir_rgb}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}.png", color_image)
                    # print('Saved color image:', f"{out_dir_rgb}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}.png")


                    os.makedirs(f"{out_dir_depth}/{subject}/{sequence}/{frame}", exist_ok=True)
                    d_i = d[i].cpu().numpy()
                    np.save(f"{out_dir_depth}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}.npy", d_i)
                    # print('Saved depth:', f"{out_dir_depth}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}.npy")

                    # save intrinsics
                    os.makedirs(f"{out_dir_intrinsics}/{subject}/{sequence}/{frame}", exist_ok=True)
                    intrinsics = camera_intrinsics[0][i].cpu().numpy()
                    # print('intrinsics shape:', intrinsics.shape)
                    # print('intrinsics:', intrinsics)
                    np.save(f"{out_dir_intrinsics}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}_intrinsics.npy", intrinsics)
                # print('Saved intrinsics:', f"{out_dir_intrinsics}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}_intrinsics.npy")

                # imageio.imwrite(f"{out_dir}/{subject}/{sequence}/{frame}/{sequence}.{frame}.{camera_name}.png", v_i)
    except Exception as e:
        print(e)        
        raise e
        continue
