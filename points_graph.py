import torch

import matplotlib.pyplot as plt
import numpy as np

N_bounds = 30

N_interior = 30

bounds_points = np.linspace(0, 1, N_bounds)

zeros = np.zeros(N_bounds)

ones = np.full((N_bounds, 1), 1)

x_cords = np.random.rand(N_interior)
y_cords = np.random.rand(N_interior)

x = np.linspace(0, 1, N_interior)
y = np.linspace(0, 1, N_interior)

X, Y = np.meshgrid(x[1:-1], y[1:-1])


# Create the plot
fig, ax = plt.subplots()
size=3
size_edge=5
ax.plot(zeros, bounds_points, 'rx', markersize=size_edge)
ax.plot(ones, bounds_points, 'rx', markersize=size_edge)
ax.plot(bounds_points, zeros, 'rx', markersize=size_edge)
ax.plot(bounds_points, ones, 'rx', markersize=size_edge)
ax.plot(X, Y, 'bo', markersize=size)

# Set equal aspect ratio
ax.set_aspect('equal')

# Set limits and grid
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.grid()
plt.title('Set of Training Points')
# Show plot
plt.show()