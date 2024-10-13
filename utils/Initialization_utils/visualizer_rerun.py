import numpy as np
import rerun as rr

# Initialize the rerun viewer session
rr.init("gaussian_splats_example", spawn=True)

# Number of Gaussian splats
num_points = 50

# Create some random 2D points
points = np.random.rand(num_points, 2)

# Create random intensities for the splats
intensities = np.random.rand(num_points)

# Add Gaussian splats with rerun (visualizing the points as splats)
rr.log_points("splats", points, radii=0.1 + 0.3 * intensities, colors=np.random.rand(num_points, 3))

# Optionally, run the rerun viewer until it is manually closed
rr.spawn()
