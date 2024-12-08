import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 1, 100)
Y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(X, Y)

k = 10

m = np.sqrt(k**2 - 4*np.pi**2)

U = 1/np.sin(m)*np.sin(2*np.pi*X)*np.sin(m*Y)

fig, ax = plt.subplots()


ax.contourf(X, Y, U, levels=100, cmap='viridis')
ax.set_aspect('equal')

for c in ax.collections:
    c.set_edgecolor("face")

plt.show()
