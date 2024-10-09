import torch

import matplotlib.pyplot as plt
import numpy as np

N_bounds = 800

N_interior = 30

bounds_points = np.linspace(0, 1, N_bounds)

zeros = np.zeros(N_bounds)

ones = np.full((N_bounds, 1), 1)

x_cords = np.random.rand(N_interior)
y_cords = np.random.rand(N_interior)

x = np.linspace(0, 1, N_interior)
y = np.linspace(0, 1, N_interior)

X, Y = np.meshgrid(x[1:-1], y[1:-1])


plt.plot(zeros, bounds_points, 'rx')
plt.plot(ones, bounds_points, 'rx')

plt.plot(bounds_points, zeros, 'rx')
plt.plot(bounds_points, ones, 'rx')

plt.plot(X, Y, 'bo')

print(len(X))

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()

plt.show()