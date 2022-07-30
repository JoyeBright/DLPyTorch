import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
np.random.seed(42)

x = np.random.rand(100, 1)
y = 1 + 2 * x

idx = np.arange(100)
np.random.shuffle(idx)

train_idx = idx[:80]
val_idx = idx[80:]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

print(type(x_train), type(x_train_tensor), x_train_tensor.type())

# Training Step
def make_train_step(model, loss_fn, optimizer):
    
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()    
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return train_step

# Model
class ManualLinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * x
# Model 2
class LayerLinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Instances
# Now we can create a model and send it at once to the device
model = ManualLinearRegression().to(device)
model2 = LayerLinearRegression().to(device)

# We can also inspect its parameters using its state_dict
print(model2.state_dict())

# Hyperparamaters
lr = 1e-1
n_epochs = 1000
# Loss function and Optimizer
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model2.parameters(), lr=lr)

train_step = make_train_step(model2, loss_fn, optimizer)
losses = []

for e in range(n_epochs):
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)

print(model2.state_dict())