import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from anim import create_anim_gif
from utils import get_all_points

L = 1
Nx = 15
tf = 2
Nt = 100
alpha = 1

x_space = np.linspace(0, L, Nx)
t_space = np.linspace(0, tf, Nt)

X_train, T_train = get_all_points(x_space, t_space)

def initial_condition(x):
    return np.sin(2 * np.pi * x)

class Model(nn.Module):
    def __init__(self, in_features=2, h1=8, h2=8, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x, t):
        u = torch.cat([x, t], dim=1)
        u = F.tanh(self.fc1(u))
        u = F.tanh(self.fc2(u))
        u = self.out(u)
        return u

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
def loss_function(X, T, u_pred):
    # derivatives
    u_t = torch.autograd.grad(u_pred, T, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, X, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    pde_loss = torch.mean((u_t - alpha * u_xx)**2)

    # boundary condition loss
    left_boundary = model(torch.zeros(len(T_train), 1), T_train)  # u(0, t) = 0
    right_boundary = model(torch.full((len(T_train), 1), L), T_train)   # u(L, t) = 0
    boundary_loss = torch.mean(left_boundary**2) + torch.mean(right_boundary**2)
    
    # initial condition loss
    u0_pred = model(X_train, torch.tensor([[0.0]]*len(X_train)))  # t = 0
    initial_loss = torch.mean((u0_pred - torch.tensor(initial_condition(X_train.detach().numpy()), dtype=torch.float32))**2)
    
    # total loss
    total_loss = pde_loss + boundary_loss + initial_loss
    
    return total_loss


# Training loop
epochs = 20000
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # forward propagation
    u_pred = model.forward(X_train, T_train)
    # calculate loss
    loss = loss_function(X_train, T_train, u_pred)
    losses.append(loss.detach().numpy())
    # Backpropagate
    loss.backward()
    # Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# plot error
plt.plot(range(epochs), losses)
plt.grid()
plt.show()


# TRAINED MODEL

frames = []

space_tensor = torch.tensor(x_space.reshape(-1, 1), dtype=torch.float32)

for t in t_space:
    u_at_t = model.forward(space_tensor, torch.full((len(space_tensor), 1), t))
    frames.append(u_at_t)


create_anim_gif('heat_nn', L, [frame.detach().numpy() for frame in frames], x_space)