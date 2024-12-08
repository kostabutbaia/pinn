import matplotlib.pyplot as plt
import numpy as np
N = 50
x_space = np.linspace(0, 4, 4 * N)

plt.plot(x_space[:N+1], np.sin(2*np.pi*x_space[:N+1]),'b')
plt.plot(x_space[N:2*N], np.zeros(50),'b')
plt.plot(x_space[2*N:3*N], np.zeros(50),'b')
plt.plot(x_space[3*N:4*N], np.zeros(50),'b')

plt.grid()
plt.axvline(x=1, color='black', linestyle='--')
plt.axvline(x=2, color='black', linestyle='--')
plt.axvline(x=3, color='black', linestyle='--')
plt.ylim([-1, 1])
plt.xlim([0, 4])
plt.text(0.5, 1.05, 'Top Edge', fontsize=10, color='black', ha='center')
plt.text(1.5, 1.05, 'Bottom Edge', fontsize=10, color='black', ha='center')
plt.text(2.5, 1.05, 'Left Edge', fontsize=10, color='black', ha='center')
plt.text(3.5, 1.05, 'Right Edge', fontsize=10, color='black', ha='center')
plt.show()