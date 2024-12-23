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

N = 50

k = 15
lambda_b = 100
lambda_pde = 1

num_hidden_nodes=15

x_space = np.linspace(0, 1, N)
y_space = np.linspace(0, 1, N)

x_train, y_train = get_all_points(x_space, y_space)

class Model(nn.Module):
    def __init__(self, in_features=2, h1=num_hidden_nodes, h2=num_hidden_nodes, h3=num_hidden_nodes, h4=num_hidden_nodes, out_features=1):
        super().__init__()
        self.omega = 30
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_features)

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
        u = self.out(u)
        # u = x*(x-1)*y*(y-1)*u
        return u

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
count = 0
def loss_function(X, Y, u_pred):
    global count
    count += 1
    # derivatives
    u_x = torch.autograd.grad(u_pred, X, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    u_y = torch.autograd.grad(u_pred, Y, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, Y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    pde_loss = torch.mean((u_xx + u_yy + k**2 * u_pred - k**2*torch.sin(k*X)*torch.sin(k*Y))**2)

    # boundary condition loss
    # b1 = model(torch.zeros(len(x_train), 1), y_train)  # u(0, y) = 0
    # b2 = model(torch.full((len(x_train), 1), 1), y_train)   # u(1, y) = 0

    # b3 = model(x_train, torch.zeros(len(y_train), 1))  # u(x, 0) = 0
    # b4 = model(x_train, torch.full((len(y_train), 1), 1))   # u(x, 1) = 0


    # boundary_loss = torch.mean(b1**2) + torch.mean(b2**2) + torch.mean(b3**2) + torch.mean(b4**2)
    # total loss
    total_loss = lambda_pde*pde_loss

    if count % 100 == 0:
        print(f'pde_loss: {lambda_pde*pde_loss.detach().numpy()}')
    
    return total_loss


# Training loop
X_plot, Y_plot = np.meshgrid(x_space, y_space)
epochs = 50000
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


def closure():
    l_bfgs_optimizer.zero_grad()  # Zero gradients
    u_pred = model(x_train, y_train)
    loss = loss_function(x_train, y_train, u_pred)
    loss.backward()  # Compute gradients
    return loss

# 4. Initialize L-BFGS optimizer
l_bfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=15000, history_size=10)

# Fine-tuning with L-BFGS
print('Start fine tunning')
model.train()
l_bfgs_optimizer.step(closure)

print('Fine-tuning complete.')

# plot error
plt.plot(range(epochs), losses)
plt.grid()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X_plot, Y_plot, frames[-1], cmap=cm.coolwarm)
plt.show()

text = fr"""
    PINN solution of 2D helmholtz equation on $[0,1]x[0,1]$
    $u_{{xx}}+u_{{yy}} + k^2 u = k^2\sin(kx)\sin(ky)$
    $u(x,0)=0 \quad u(x, 1)=0 \quad u(0,y)=0 \quad u(1, y)=0$

    Where:
    - k={k}

    PINN parameters:
    - activation: $\sin$
    - number of hidden nodes: 8
    - number of nodes in hidden layer: {num_hidden_nodes}
    - weights init parameter: {model.omega}

    Training:
    - number of points: {N}
    - epochs: {epochs}
    - $\lambda_b={lambda_b}$, $\lambda_{{pde}}={lambda_pde}$
    """

make_contour_plot(
    X_plot,
    Y_plot,
    frames[-1],
    text
)

create_gif_contour(
    f'sinsin_{epochs}',
    X_plot,
    Y_plot,
    frames,
    text
)