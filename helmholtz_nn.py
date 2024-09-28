import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from anim import create_anim_gif
from utils import get_all_points

N = 30

k = np.pi * np.sqrt(8)

x_space = np.linspace(0, 1, N)
y_space = np.linspace(0, 1, N)

x_train, y_train = get_all_points(x_space, y_space)

class Model(nn.Module):
    def __init__(self, in_features=2, h1=8, h2=8, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x, y):
        u = torch.cat([x, y], dim=1)
        u = F.tanh(self.fc1(u))
        u = F.tanh(self.fc2(u))
        u = self.out(u)
        return u

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
def loss_function(X, Y, u_pred):
    # derivatives
    u_x = torch.autograd.grad(u_pred, X, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    u_y = torch.autograd.grad(u_pred, Y, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, Y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    pde_loss = torch.mean((u_xx + u_yy + k * u_pred)**2)

    # boundary condition loss
    b1 = model(torch.zeros(len(x_train), 1), y_train)  # u(0, y) = 0
    b2 = model(torch.full((len(x_train), 1), 1), y_train)   # u(1, y) = 0

    b3 = model(x_train, torch.zeros(len(y_train), 1))  # u(x, 0) = 0
    b4 = model(x_train, torch.full((len(y_train), 1), 1),)   # u(x, 1) = 0

    boundary_loss = torch.mean(b1**2) + torch.mean(b2**2) + torch.mean(b3**2) + torch.mean(b4**2)

    # target condition
    t_point = model(torch.tensor([[0.75]], requires_grad=True), torch.tensor([[0.25]], requires_grad=True))
    t_value = torch.tensor([1.0])
    loss_target = torch.mean((t_point - t_value)**2)
    # total loss
    total_loss = pde_loss + boundary_loss + loss_target
    
    return total_loss


# Training loop
epochs = 10000
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # forward propagation
    u_pred = model.forward(x_train, y_train)
    # calculate loss
    loss = loss_function(x_train, y_train, u_pred)
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

X_plot, Y_plot = np.meshgrid(x_space, y_space)
U_plot = []

for i, x_points in enumerate(X_plot):
    lst = []
    for x in x_points:
        lst.append(model.forward(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[Y_plot[i][0]]], dtype=torch.float32)).detach().numpy()[0][0])
    U_plot.append(lst)


plt.contourf(X_plot, Y_plot, U_plot, levels=100, cmap='jet')
plt.colorbar()
plt.title('Solution to Helmholtz Equation on Square [0,1]x[0,1] with Zero Boundary Conditions')
plt.xlabel('x')
plt.ylabel('y')
plt.show()