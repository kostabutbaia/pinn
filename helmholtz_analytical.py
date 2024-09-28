import matplotlib.pyplot as plt
import numpy as np

n = 2
m = 2

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(n * np.pi * X) * np.sin(m * np.pi * Y)

plt.contourf(X, Y, Z, levels=100, cmap='jet')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()