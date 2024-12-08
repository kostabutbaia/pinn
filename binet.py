import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
from anim import create_gif_contour
from matplotlib import cm
from plotter import make_contour_plot

from utils import split_array_N
from scipy.special import hankel1

""" Parameters """

num_hidden_nodes=50
k = 15

N = 50
N_TRAIN = 50

epochs = 5000

plot_points_num = 25

""" Points """

cords = torch.linspace(0.0001, 0.9999, N)
ones = torch.full((N, 1), 1)
zeros = torch.zeros(N, 1)

train_cords = torch.linspace(0, 1, N_TRAIN)
ones_train = torch.full((N_TRAIN, 1), 1)
zeros_train = torch.zeros(N_TRAIN, 1)

dy = 1/(N-1)

""" Plot Points """
X_plot = torch.linspace(0.1, 0.9, plot_points_num)
Y_plot = torch.linspace(0.1, 0.9, plot_points_num)
X_plot, Y_plot = torch.meshgrid(X_plot, Y_plot)

points_tensor = torch.stack([X_plot.flatten(), Y_plot.flatten()], dim=1)

""" Boundary Points for integral calculation """

top_boundary_points = torch.column_stack((cords, ones))
bottom_boundary_points = torch.column_stack((cords, zeros))

left_boundary_points = torch.column_stack((zeros, cords))
right_boundary_points = torch.column_stack((ones, cords))

boundary_points = torch.concatenate((
    top_boundary_points, 
    bottom_boundary_points,
    left_boundary_points, 
    right_boundary_points
))

""" Boundary Points for training """

top_boundary_points_train = torch.column_stack((train_cords, ones_train))
bottom_boundary_points_train = torch.column_stack((train_cords, zeros_train))

left_boundary_points_train = torch.column_stack((zeros_train, train_cords))
right_boundary_points_train = torch.column_stack((ones_train, train_cords))

train_boundary_points = torch.concatenate((
    top_boundary_points_train, 
    bottom_boundary_points_train,
    left_boundary_points_train, 
    right_boundary_points_train
))

""" Normals for integral calculation """

top_normal = torch.column_stack((zeros, ones))
bottom_normal = torch.column_stack((zeros, -ones))

left_normal = torch.column_stack((-ones, zeros))
right_normal = torch.column_stack((ones, zeros))

normals = torch.concatenate((
    top_normal, 
    bottom_normal,
    left_normal, 
    right_normal
))

""" Binet """

def boundary_condition(points):
    result = torch.zeros(points.shape[0], dtype=torch.float)
    for i, point in enumerate(points):
        if abs(point[1] - 1.0) < 0.001:
            result[i] = torch.sin(2 * np.pi * point[0])
        else:
            result[i] == 0

    return result


def fundamental_normal_derivative(x, y):
    r = torch.norm(x - y, dim=-1)
    hankel_prime = -(1j / 4) * k * torch.tensor(hankel1(1, k * r))
    dr_dn = torch.sum((x-y) * normals, dim=1) / (r+1e-8)
    return hankel_prime * dr_dn

class Model(nn.Module):
    def __init__(self, boundary_points, dy, in_features=2, h1=num_hidden_nodes, h2=num_hidden_nodes, h3=num_hidden_nodes, h4=num_hidden_nodes, out_features=1):
        super().__init__()
        # boundary points
        self.boundary_points = boundary_points
        self.dy = dy
        # NN
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

    def h(self, points):
        h = torch.tanh(self.fc1(points))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        h = torch.tanh(self.fc4(h))
        h_values = self.out(h)

        return h_values
        
    def forward(self, points):
        h_values = self.h(self.boundary_points).view(1, -1)[0]

        u_at_points = torch.zeros(points.shape[0], dtype=torch.cfloat)

        for i, point in enumerate(points):
            green_vals = fundamental_normal_derivative(point, self.boundary_points)
            u_at_point = torch.sum(green_vals * h_values * self.dy)
            u_at_points[i] = u_at_point

        return u_at_points.real

frames = []
losses = []

def train_model(model: nn.Module, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute the network output (boundary integral)
        u_pred = model(train_boundary_points)

        # Compute the loss with respect to the boundary condition
        boundary_values = boundary_condition(train_boundary_points)

        loss = torch.mean((u_pred + 1/2*model.h(train_boundary_points).view(1,-1)[0] - boundary_values)**2)

        # print(f'{u_pred=}')
        # print(f'{model.h(train_boundary_points).view(1,-1)[0]=}')
        # print(f'{boundary_values=}')

        # exit()
        losses.append(loss)
        # Backpropagation and optimization
        loss.backward()

        optimizer.step()

        # if epoch % 5 == 0:
        #     values = model.forward(points_tensor)
        #     U = split_array_N(values.detach().numpy(), plot_points_num)
        #     frames.append(U)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


    return model

model = Model(boundary_points, dy)

train_model(model, epochs)

x_space = np.linspace(0, 4, 4 * N_TRAIN)

u_pred = model(train_boundary_points)

plt.plot(x_space, 1/2*model.h(train_boundary_points).view(1,-1)[0].detach().numpy() + u_pred.detach().numpy(), 'b')

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

values = model.forward(points_tensor)
U = split_array_N(values.detach().numpy(), plot_points_num)
frames.append(U)

text = fr"""
    BINet solution of 2D helmholtz equation on $[0,1]x[0,1]$
    $u_{{xx}}+u_{{yy}} + k^2 u = 0, k={k}$
    $u(x,0)=0 \quad u(x, 1)=\sin(2\pi x) \quad u(0,y)=0 \quad u(1, y)=0$

    BINet parameters:
    - activation: $\tanh$
    - number of hidden layers: 4
    - number of nodes in hidden layer: {num_hidden_nodes}
    - Xavier weights initialization

    Training:
    - number of training points: {N_TRAIN}, number of integral points: {N}
    - Adam epochs: {epochs}
    - final loss: {losses[-1]}
    """

make_contour_plot(
    X_plot,
    Y_plot,
    frames[-1],
    text,
    True,
    ''
)

# create_gif_contour(
#     f'sine_edge_{epochs}',
#     X_plot,
#     Y_plot,
#     frames,
#     'Test'
# )

# plt.contourf(X_plot, Y_plot, frames[-1], levels=100, cmap='viridis')

# plt.show()