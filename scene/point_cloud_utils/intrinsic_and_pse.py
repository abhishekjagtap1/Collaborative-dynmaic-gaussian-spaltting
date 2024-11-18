import numpy as np

# Given projection matrix
south_1_proj = np.asarray(
                            [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
                             [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
                             [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]],
                            dtype=np.float32)

# Given intrinsic matrix
intrinsic_south_2 = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                            [0.0, 1362.7757568359375, 580.6482296623581],
                                                            [0.0, 0.0, 1.0]], dtype=np.float32)

# Compute the inverse of the intrinsic matrix
K_inv = np.linalg.inv(intrinsic_south_2)
--port
6019
--expname
"After_CVPR_fail/Yellow_collab_two_infra"

# Extract the extrinsic matrix [R | t] by multiplying K^-1 with the projection matrix
extrinsic_matrix = K_inv @ south_1_proj

# Extract the rotation matrix R and the translation vector t
R = extrinsic_matrix[:, :3]
t = extrinsic_matrix[:, 3]

print("Extrinsic Matrix [R|t]:\n", extrinsic_matrix)
print("Rotation Matrix R:\n", R)
print("Translation Vector t:\n", t)

transformation_matrix_base_to_camera_south_1 = np.array([
                            [0.891382638626301, 0.37756862104528707, -0.07884507325924934, 25.921784677055939],
                            [0.2980421080238165, -0.6831891949380544, -0.6660273169946723, 13.668310799382738],
                            [-0.24839844089507856, 0.5907739097931769, -0.7525203649548087, 18.630430017833277],
                            [0, 0, 0, 1]], dtype=float)
transformation_matrix_lidar_to_base_south_1 = np.array([
                            [0.247006, -0.955779, -0.15961, -16.8017],
                            [0.912112, 0.173713, 0.371316, 4.66979],
                            [-0.327169, -0.237299, 0.914685, 6.4602],
                            [0.0, 0.0, 0.0, 1.0], ], dtype=float)

extrinsic_matrix_lidar_to_camera_south_1 = np.matmul(
                            transformation_matrix_base_to_camera_south_1,
                            transformation_matrix_lidar_to_base_south_1)
camera_to_lidar_extrinsics_south_1 = np.linalg.inv(extrinsic_matrix_lidar_to_camera_south_1)
print("Extrinsic Matrixfgerregergr [R|t]:\n", camera_to_lidar_extrinsics_south_1)
