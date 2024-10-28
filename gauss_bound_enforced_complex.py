import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
from anim import create_gif_contour
from plotter import make_contour_plot
from matplotlib import cm

from utils import get_all_points

L = 1

show = False
N = 30

sigma = 0.001
A = 800
num_hidden_nodes=40

epochs = 50000

epochs_bf = 10000

name = f'bound_enforced_{epochs}'


a = 40**2
b = 0

""" PINNs """

x_space = np.linspace(0, L, N)
y_space = np.linspace(0, L, N)

x_train, y_train = get_all_points(x_space, y_space)

class Model(nn.Module):
    def __init__(self, in_features=2, h1=num_hidden_nodes, h2=num_hidden_nodes, h3=num_hidden_nodes, h4=num_hidden_nodes, h5=num_hidden_nodes, h6=num_hidden_nodes, h7=num_hidden_nodes, h8=num_hidden_nodes, out_features=2):
        super().__init__()
        self.omega = 30
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.fc6 = nn.Linear(h5, h6)
        self.fc7 = nn.Linear(h6, h7)
        self.fc8 = nn.Linear(h7, h8)
        self.out = nn.Linear(h8, out_features)
        self.apply(self.init_weights_siren)
    
    def init_weights_siren(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-1 / self.omega, 1 / self.omega)
            m.bias.data.uniform_(-1 / self.omega, 1 / self.omega)
        
    def forward(self, x, y):
        u = torch.cat([x, y], dim=1)
        u = self.omega * u
        u = torch.sin(self.fc1(u))
        u = torch.sin(self.fc2(u))
        u = torch.sin(self.fc3(u))
        u = torch.sin(self.fc4(u))
        u = torch.sin(self.fc5(u))
        u = torch.sin(self.fc6(u))
        u = torch.sin(self.fc7(u))
        u = torch.sin(self.fc8(u))
        u = self.out(u)
        u = x*(x-L)*y*(y-L)*u
        u_real = u[:, 0].unsqueeze(1)  # First output is real part
        u_imag = u[:, 1].unsqueeze(1)  # Second output is imaginary part
        return u_real, u_imag

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
count = 0
def loss_function(X, Y, u_real, u_imag):
    global count
    count += 1
    # derivatives for real part
    u_x_real = torch.autograd.grad(u_real, X, grad_outputs=torch.ones_like(u_real), create_graph=True)[0]
    u_xx_real = torch.autograd.grad(u_x_real, X, grad_outputs=torch.ones_like(u_x_real), create_graph=True)[0]

    u_y_real = torch.autograd.grad(u_real, Y, grad_outputs=torch.ones_like(u_real), create_graph=True)[0]
    u_yy_real = torch.autograd.grad(u_y_real, Y, grad_outputs=torch.ones_like(u_y_real), create_graph=True)[0]

    # derivatives for imaginary part
    u_x_imag = torch.autograd.grad(u_imag, X, grad_outputs=torch.ones_like(u_imag), create_graph=True)[0]
    u_xx_imag = torch.autograd.grad(u_x_imag, X, grad_outputs=torch.ones_like(u_x_imag), create_graph=True)[0]

    u_y_imag = torch.autograd.grad(u_imag, Y, grad_outputs=torch.ones_like(u_imag), create_graph=True)[0]
    u_yy_imag = torch.autograd.grad(u_y_imag, Y, grad_outputs=torch.ones_like(u_y_imag), create_graph=True)[0]

    # Loss function for the real part
    pde_loss_real = torch.mean((u_xx_real + u_yy_real + a * u_real - b * u_imag + A * torch.exp(-((X-L/2)**2+(Y-L/2)**2)/sigma))**2)

    # Loss function for the imaginary part
    pde_loss_imag = torch.mean((u_xx_imag + u_yy_imag + a * u_imag + b * u_real)**2)

    # total loss is the sum of both real and imaginary parts
    total_loss = pde_loss_real + pde_loss_imag

    if count % 100 == 0:
        print(f'epoch: {count}, total_loss: {total_loss.detach().numpy()}')
    
    return total_loss


# Training loop
X_plot, Y_plot = np.meshgrid(x_space, y_space)
losses = []
frames = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # forward propagation
    u_real, u_imag = model.forward(x_train, y_train)
    # calculate loss
    loss = loss_function(x_train, y_train, u_real, u_imag)
    losses.append(loss.detach().numpy())
    # Backpropagate
    loss.backward()
    # Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        frame = []
        for i, x_points in enumerate(X_plot):
            lst = []
            for x in x_points:
                u_r, _ = model.forward(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[Y_plot[i][0]]], dtype=torch.float32))
                lst.append(u_r.detach().numpy()[0][0])
            frame.append(np.array(lst))
        frames.append(np.array(frame))
        # print(f"Epoch {epoch}: Loss = {loss.item()}")



count_bf = 0
def closure():
    global count_bf
    count_bf += 1
    l_bfgs_optimizer.zero_grad()  # Zero gradients
    u_real, u_imag = model.forward(x_train, y_train)
    loss = loss_function(x_train, y_train, u_real, u_imag)
    losses.append(loss.detach().numpy())
    loss.backward()  # Compute gradients
    if count_bf % 100 == 0:
        frame = []
        for i, x_points in enumerate(X_plot):
            lst = []
            for x in x_points:
                u_r, _ = model.forward(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[Y_plot[i][0]]], dtype=torch.float32))
                lst.append(u_r.detach().numpy()[0][0])
            frame.append(np.array(lst))
        frames.append(np.array(frame))
    return loss

# Initialize L-BFGS optimizer
l_bfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=epochs_bf, history_size=10)

# Fine-tuning with L-BFGS
print('Start fine tunning')
model.train()
l_bfgs_optimizer.step(closure)
print('Fine-tuning complete.')


""" Get Results """

# Plot Error
plt.plot(range(len(losses)), losses)
plt.grid()
plt.xlabel('epoch')
plt.xlabel('Error')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X_plot, Y_plot, frames[-1], cmap=cm.coolwarm)
plt.show()

text = fr"""
    PINN solution of 2D helmholtz equation on $[0,1]x[0,1]$
    $u_{{xx}}+u_{{yy}} + k^2 u = Ae^{{-\frac{{(x-0.5)^2+(y-0.5)^2}}{{\sigma}}}}$
    $u(x,0)=0 \quad u(x, 1)=0 \quad u(0,y)=0 \quad u(1, y)=0$

    Where:
    - k={a}+i{b}
    - A={A}
    - $\sigma$={sigma}

    PINN parameters:
    - activation: $\sin$
    - number of hidden nodes: 8
    - number of nodes in hidden layer: {num_hidden_nodes}
    - weights init parameter: {model.omega}

    Training:
    - number of points: {N}
    - Adam epochs: {epochs}, L-BFGS epochs: {epochs_bf}
    - Boundary condition enforced in NN output
    """

make_contour_plot(
    X_plot,
    Y_plot,
    frames[-1],
    text,
    show,
    name
)

create_gif_contour(
    f'gauss_{epochs}',
    X_plot,
    Y_plot,
    frames,
    text
)