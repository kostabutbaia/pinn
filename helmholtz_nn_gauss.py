import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
from anim import create_gif_contour
from plotter import make_contour_plot

from utils import get_all_points

N = 30

k = 20
lambda_b = 100
lambda_pde = 0.1

sigma = 0.001
A = 800
num_hidden_nodes=8

x_space = np.linspace(0, 1, N)
y_space = np.linspace(0, 1, N)

x_train, y_train = get_all_points(x_space, y_space)

class Model(nn.Module):
    def __init__(self, in_features=2, h1=num_hidden_nodes, h2=num_hidden_nodes, h3=num_hidden_nodes, h4=num_hidden_nodes, h5=num_hidden_nodes, h6=num_hidden_nodes, h7=num_hidden_nodes, h8=num_hidden_nodes, out_features=1):
        super().__init__()
        self.omega = 60
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

    pde_loss = torch.mean((u_xx + u_yy + k**2 * u_pred + A * torch.exp(-((X-0.5)**2+(Y-0.5)**2)/sigma))**2)

    # boundary condition loss
    b1 = model(torch.zeros(len(x_train), 1), y_train)  # u(0, y) = 0
    b2 = model(torch.full((len(x_train), 1), 1), y_train)   # u(1, y) = 0

    b3 = model(x_train, torch.zeros(len(y_train), 1))  # u(x, 0) = 0
    b4 = model(x_train, torch.full((len(y_train), 1), 1))   # u(x, 1) = 0


    boundary_loss = torch.mean(b1**2) + torch.mean(b2**2) + torch.mean(b3**2) + torch.mean(b4**2)
    # total loss
    total_loss = lambda_pde*pde_loss + lambda_b*boundary_loss

    # if count % 100 == 0:
    #     print(f'pde_loss: {lambda_pde*pde_loss.detach().numpy()}, boundary loss: {lambda_b*boundary_loss.detach().numpy()}, boundary contr: {lambda_b*boundary_loss.detach().numpy()/total_loss.detach().numpy()}%')
    
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

# plot error
plt.plot(range(epochs), losses)
plt.grid()
plt.show()

text = fr"""
    PINN solution of 2D helmholtz equation on $[0,1]x[0,1]$
    $u_{{xx}}+u_{{yy}} + k^2 u(x) = Ae^{{-\frac{{(x-0.5)^2+(y-0.5)^2}}{{\sigma}}}}$
    $u(x,0)=0 \quad u(x, 1)=0 \quad u(0,y)=0 \quad u(1, y)=0$

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
    - $\lambda_b={lambda_b}$, $\lambda_{{pde}}={lambda_pde}$
    """

make_contour_plot(
    X_plot,
    Y_plot,
    frames[-1],
    text
)

create_gif_contour(
    f'gauss_{epochs}',
    X_plot,
    Y_plot,
    frames,
    text
)