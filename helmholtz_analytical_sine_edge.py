import matplotlib.pyplot as plt
import numpy as np

n = 2
m = 2

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = 1/np.sinh(np.sqrt(4*np.pi**2-1))*np.sin(2*np.pi*X)*np.sinh(Y*np.sqrt(4*np.pi**2-1))

plt.contourf(X, Y, Z, levels=100, cmap='jet')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()