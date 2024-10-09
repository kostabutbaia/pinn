import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from utils import get_all_points
from anim import create_anim_gif_text
from plotter import make_plot

N = 150

num_hidden_nodes = 20

k = 20

lambda_b = 1
lambda_pde = 1

A = 800
sigma = 0.001

x_space = np.linspace(0, 1, N)

x_points = torch.tensor(x_space.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

class Model(nn.Module):
    def __init__(self, in_features=1, h1=num_hidden_nodes, h2=num_hidden_nodes, h3=num_hidden_nodes, h4=num_hidden_nodes, h5=num_hidden_nodes, h6=num_hidden_nodes, h7=num_hidden_nodes, h8=num_hidden_nodes, out_features=1):
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
        
    def forward(self, u):
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
        return u

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def loss_function(X, u_pred):
    # derivatives
    u_x = torch.autograd.grad(u_pred, X, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    pde_loss = torch.mean((u_xx + k**2 * u_pred + A * torch.exp(-((X-0.5)**2)/sigma))**2)

    # boundary condition loss
    b1 = model(torch.tensor([[0.0]], requires_grad=False, dtype=torch.float32))  # u(0) = 0
    b2 = model(torch.tensor([[1.0]], requires_grad=False, dtype=torch.float32))   # u(1) = 0

    boundary_loss = torch.mean(b1**2) + torch.mean(b2**2)
    # total loss
    total_loss = pde_loss + lambda_b*boundary_loss

    return total_loss


# Training loop
epochs = 50000
losses = []
frames = []
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # forward propagation
    u_pred = model.forward(x_points)
    # calculate loss
    loss = loss_function(x_points, u_pred)
    losses.append(loss.detach().numpy())
    # Backpropagate
    loss.backward()
    # Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        frames.append(u_pred.detach().numpy())
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# plot error
plt.plot(range(epochs), losses)
plt.grid()
plt.show()

text = fr"""
    PINN solution of 1D helmholtz equation on $[0,1]$
    $\frac{{d^2 u(x)}}{{dx^2}} + k^2 u(x) = Ae^{{\frac{{-(x-0.5)^2}}{{\sigma}}}}$
    $u(0)=0\quad u(1)=0$

    Where:
    - k={k}
    - A={A}
    - $\sigma$={sigma}

    PINN parameters:
    - activation: $\sin$
    - number of hidden nodes: 8
    - number of nodes in hidden layer: {num_hidden_nodes}
    - weights init parameter: {model.omega}

    Training:
    - number of points: {N}
    - epochs: {epochs}
    - loss decrease: {"{:.2f}".format(losses[1]/losses[0]*100, 3)}%
    - $\lambda_b={lambda_b}$, $\lambda_{{pde}}={lambda_pde}$
    """

make_plot(
    x_space,
    frames[-1],
    text,
    'x',
    'u(x)',
    ''
)
create_anim_gif_text(f'helmholtz_1D_{epochs}', text, 1, frames, x_space)