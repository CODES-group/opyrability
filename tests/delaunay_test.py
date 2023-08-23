import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

AOS =np.load('aos_dma_mr.npy')


# Generate random 2D points as an example
np.random.seed(0)
num_points = 20
points = np.random.rand(num_points, 2)

# Perform Delaunay triangulation
tri = Delaunay(points)

# Get the simplices from the triangulation
simplices = tri.simplices

# Function to remove overlapping simplices
def remove_overlapping_simplices(simplices):
    unique_simplices = []
    for simplex in simplices:
        if all(np.all(simplex != us) for us in unique_simplices):
            unique_simplices.append(simplex)
    return np.array(unique_simplices)

unique_simplices = remove_overlapping_simplices(simplices)

# Plotting the Delaunay triangulation and the removed simplices
plt.triplot(points[:, 0], points[:, 1], simplices, label='Delaunay Triangulation')
plt.triplot(points[:, 0], points[:, 1], unique_simplices, label='Removed Overlapping Simplices', color='r')

plt.plot(points[:, 0], points[:, 1], 'o', label='Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Delaunay Triangulation with Removed Overlapping Simplices')
# plt.legend()
plt.show()