# Common data-path flags shared by every stage script. Edit the paths here to
# match your machine; the four stage launchers source this file.

DATA_FLAGS=(
  -tdl   /fast/pfilntisis/TEMPEH_data/data/aws_data/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json
  -vdl   /fast/pfilntisis/TEMPEH_data/data/aws_data/eight_subj__all_seq_frames_per_seq_5_val.json
  --scan-directory       /fast/pfilntisis/TEMPEH_data/data/aws_data/meshes_npz
  --processed-directory  /fast/pfilntisis/TEMPEH_data/data/aws_data/registrations
  --image-directory          /fast/pfilntisis/TEMPEH_data/data/aws_data/distorted_downsampled_4_culled_grid/color_images_v2
  --normals-image-directory  /fast/pfilntisis/TEMPEH_data/data/aws_data/distorted_downsampled_4_culled_grid/color_normals_numpy
  --depths-image-directory   /fast/pfilntisis/TEMPEH_data/data/aws_data/distorted_downsampled_4_culled_grid/color_depth
  --calibration-directory    /fast/pfilntisis/TEMPEH_data/data/aws_data/distorted_downsampled_4_culled_grid/color_cameras
  --dense-landmarks-dir      /fast/pfilntisis/TEMPEH_data/data/aws_data/distorted_downsampled_4_culled_grid/color_dense_landmarks
)
