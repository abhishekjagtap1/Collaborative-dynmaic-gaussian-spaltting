translation_variations = [
    [0.1, 0, 0],  # Right
    [-0.1, 0, 0],  # Left
    [0, 0.1, 0],  # Up
    [0, -0.1, 0],  # Down
    [0.1, 0.1, 0],  # Right and Up
    [0.1, -0.1, 0],  # Right and Down
    [-0.1, 0.1, 0],  # Left and Up
    [-0.1, -0.1, 0],  # Left and Down
    [0, 0, 0.1],  # Forward
    [0, 0, -0.1],  # Backward
    [0.05, 0, 0],  # Slight Right
    [-0.05, 0, 0],  # Slight Left
    [0, 0.05, 0],  # Slight Up
    [0, -0.05, 0],  # Slight Down
    [0.05, 0.05, 0],  # Slight Right and Up
    [0.05, -0.05, 0],  # Slight Right and Down
    [-0.05, 0.05, 0],  # Slight Left and Up
    [-0.05, -0.05, 0],  # Slight Left and Down
    [0, 0, 0.05],  # Slight Forward
    [0, 0, -0.05]  # Slight Backward
]

novel_poses_vehicle = generate_translated_poses(extrinsic_v, translation_variations)