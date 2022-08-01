import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
np.random.seed(42)

x = np.random.rand(100, 1)
y = 1 + 2 * x

# We don't load whole training data to the graphic card's RAM i.e., not using .to(device)!
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

dataset = TensorDataset(x_tensor, y_tensor)

train_dataset, val_dataset = random_split(dataset, [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=4)

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
val_losses = []

for e in range(n_epochs):
    for x_batch, y_batch in train_loader:
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = train_step(x_batch, y_batch)
        losses.append(loss)

        with torch.no_grad:
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                model2.eval()
                yhay = model2(x_val)
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())

print(model2.state_dict())