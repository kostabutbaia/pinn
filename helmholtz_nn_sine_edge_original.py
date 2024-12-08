import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from plotter import make_contour_plot

from utils import get_all_points

N = 30

k = 10
x_space = np.linspace(0, 1, N)
y_space = np.linspace(0, 1, N)

x_points = torch.linspace(0, 1, N).view(-1, 1)
y_points = torch.linspace(0, 1, N).view(-1, 1)

x_train, y_train = get_all_points(x_space, y_space)

num_hidden_nodes = 50
epochs = 6000

class Model(nn.Module):
    def __init__(self, in_features=2, h1=num_hidden_nodes, h2=num_hidden_nodes, h3=num_hidden_nodes, h4=num_hidden_nodes, h5=num_hidden_nodes, h6=num_hidden_nodes, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.fc6 = nn.Linear(h5, h6)
        self.out = nn.Linear(h6, out_features)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, x, y):
        u = torch.cat([x, y], dim=1)
        u = torch.sin(self.fc1(u))
        u = torch.sin(self.fc2(u))
        u = torch.sin(self.fc3(u))
        u = torch.sin(self.fc4(u))
        u = torch.sin(self.fc5(u))
        u = torch.sin(self.fc6(u))
        u = self.out(u)
        u = x*(x-1)*y*(y-1)*u + y*torch.sin(2*np.pi*x)
        return u

model = Model()

# x_points = torch.linspace(0, 1, 20).reshape(-1,1)
# y_points = torch.ones(20).reshape(-1,1)
# u = model.forward(x_points, y_points)

# plt.plot(x_points, u.detach().numpy())
# plt.grid()
# plt.show()




optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
def loss_function(X, Y, u_pred, epoch, weight):
    # derivatives
    u_x = torch.autograd.grad(u_pred, X, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    u_y = torch.autograd.grad(u_pred, Y, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, Y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    pde_loss = torch.mean((u_xx + u_yy + k**2 * u_pred)**2)

    # boundary condition loss
    b1 = model(torch.zeros(len(x_points), 1), y_points)  # u(0, y) = 0
    b2 = model(torch.full((len(x_points), 1), 1), y_points)   # u(1, y) = 0

    b3 = model(x_points, torch.zeros(len(y_points), 1))  # u(x, 0) = 0
    b4 = torch.sin(2 * np.pi * x_points) - model(x_points, torch.full((len(y_points), 1), 1))   # u(x, 1) = sin

    boundary_loss = torch.mean(b1**2) + torch.mean(b2**2) + torch.mean(b3**2) + torch.mean(b4**2)

    # if epoch % 100 == 0:
    #     print(f'{boundary_loss=}')
    #     print(f'{pde_loss=}')
    
    # total loss
    total_loss = pde_loss + weight*boundary_loss
    
    return pde_loss, boundary_loss, total_loss


# Training loop
losses_pde = []
losses_b = []
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # forward propagation
    u_pred = model.forward(x_train, y_train)
    # calculate loss
    pde_loss, boundary_loss, loss = loss_function(x_train, y_train, u_pred, epoch, 1)
    if epoch % 1 == 0:
        losses_pde.append(pde_loss.detach().numpy())
        losses_b.append(boundary_loss.detach().numpy())
    # Backpropagate
    loss.backward()
    # Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# plot error

# losses_pde = list(filter(lambda x: x < 10, losses_pde))

# window_size = 100  # You can adjust this size for smoothing
# pde = np.convolve(pde, np.ones(window_size)/window_size, mode='valid')

fig, ax = plt.subplots()

ax.plot(range(len(losses_pde)), losses_pde, 'b', label='PDE loss')
# ax.plot(range(len(losses_b)), losses_b, 'r', label='boundary loss')

plt.xlabel('epoch')
plt.ylabel('error')
plt.grid()
plt.legend()
plt.show()


X_plot, Y_plot = np.meshgrid(x_space, y_space)
U_plot = []

for i, x_points in enumerate(X_plot):
    lst = []
    for x in x_points:
        lst.append(model.forward(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[Y_plot[i][0]]], dtype=torch.float32)).detach().numpy()[0][0])
    U_plot.append(np.array(lst))

U_plot = np.array(U_plot)

fig, ax = plt.subplots()


ax.contourf(X_plot, Y_plot, U_plot, levels=100, cmap='viridis')
ax.set_aspect('equal')

for c in ax.collections:
    c.set_edgecolor("face")

plt.savefig('test.pdf')
plt.show()