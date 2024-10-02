import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from utils import get_all_points

N = 30

k = 1
x_space = np.linspace(0, 1, N)
y_space = np.linspace(0, 1, N)

x_train, y_train = get_all_points(x_space, y_space)

class Model(nn.Module):
    def __init__(self, in_features=2, h1=8, h2=8, h3=8, h4=8, h5=8, h6=8, h7=8, h8=8, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.fc6 = nn.Linear(h5, h6)
        self.fc7 = nn.Linear(h6, h7)
        self.fc8 = nn.Linear(h7, h8)
        self.out = nn.Linear(h8, out_features)
    
    def forward(self, x, y):
        u = torch.cat([x, y], dim=1)
        u = F.tanh(self.fc1(u))
        u = F.tanh(self.fc2(u))
        u = F.tanh(self.fc3(u))
        u = F.tanh(self.fc4(u))
        u = F.tanh(self.fc5(u))
        u = F.tanh(self.fc6(u))
        u = F.tanh(self.fc7(u))
        u = F.tanh(self.fc8(u))
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
    b4 = torch.sin(2 * np.pi * x_train) - model(x_train, torch.full((len(y_train), 1), 1))   # u(x, 1) = 0

    boundary_loss = torch.mean(b1**2) + torch.mean(b2**2) + torch.mean(b3**2) + torch.mean(b4**2)

    # total loss
    total_loss = pde_loss + boundary_loss
    
    return total_loss

X_plot, Y_plot = np.meshgrid(x_space, y_space)
# Training loop
epochs = 10000
losses = []
frames = []
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
        frame = []
        for i, x_points in enumerate(X_plot):
            lst = []
            for x in x_points:
                lst.append(model.forward(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[Y_plot[i][0]]], dtype=torch.float32)).detach().numpy()[0][0])
            frame.append(np.array(lst))
        frames.append(np.array(frame))
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# plot error
plt.plot(range(epochs), losses)
plt.grid()
plt.show()
plt.contourf(X_plot, Y_plot, frames[-1], levels=100, cmap='jet')
plt.colorbar()
plt.title(f'sine edge, k={k}, epochs={epochs}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig, ax = plt.subplots()
contour = ax.contourf(X_plot, Y_plot, frames[0], levels=100, cmap='jet')
writer = PillowWriter(fps=5)

with writer.saving(fig, f'sine_edge_{epochs}.gif', 100):
    for frame in frames:
        for c in contour.collections:
            c.remove()
        contour = ax.contourf(X_plot, Y_plot, frame, levels=100, cmap='jet')
        writer.grab_frame()