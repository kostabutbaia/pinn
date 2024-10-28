import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt


x_space = torch.linspace(0, 1, 300)

num_hidden_nodes=10
epochs = 10000

class Model(nn.Module):
    def __init__(self, in_features=1, h1=num_hidden_nodes, h2=num_hidden_nodes, h3=num_hidden_nodes, h4=num_hidden_nodes, out_features=1):
        super().__init__()

        # NN
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_features)
    
    def forward(self, points):
        h = torch.sin(self.fc1(points))
        h = torch.sin(self.fc2(h))
        h = torch.sin(self.fc3(h))
        h = torch.sin(self.fc4(h))
        h = self.out(h)

        return h
    
def train_model(model: nn.Module, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute the network output (boundary integral)
        u_pred = model(x_space.view(-1, 1))

        loss = torch.mean((u_pred.view(1, -1)[0] - torch.sin(20*np.pi*x_space))**2)

        # Backpropagation and optimization
        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

model = Model()

train_model(model, epochs)

plt.plot(x_space, model(x_space.view(-1, 1)).view(1, -1)[0].detach().numpy())
plt.grid()
plt.plot(x_space, torch.sin(20*np.pi*x_space), 'r--')
plt.show()