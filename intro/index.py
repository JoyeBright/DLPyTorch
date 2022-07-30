import numpy as np
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim
import torch.nn as nn
torch.manual_seed(0)
from torchviz import make_dot


np.random.seed(42)

# python and numpy

x = np.random.rand(100, 1)
y = 1 + 2 * x

idx = np.arange(100)
np.random.shuffle(idx)

train_idx = idx[:80]
val_idx = idx[80:]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)

lr = 1e-1

n_epochs = 1000

for e in range(n_epochs):

    yhat = a + b * x_train

    error = (y_train - yhat)

    loss = (error ** 2).mean()

    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    print(a_grad, b_grad)

    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)

# sklearn

linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])

# PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

print(type(x_train), type(x_train_tensor), x_train_tensor.type())

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

print(a, b)

# Defining a SGD optimizer to update the parameters
optimizer = optim.SGD([a, b], lr=lr)

# Defining a MSE loss function
loss_fn = nn.MSELoss(reduction="mean")

for epoch in range(n_epochs):

    yhat = a + b * x_train_tensor
    # error = y_train_tensor - yhat
    # loss = (error ** 2).mean()

    loss = loss_fn(y_train_tensor, yhat)

    loss.backward()
    print(a.grad)
    print(b.grad)

    # with torch.no_grad():
    #     a -= lr * a.grad
    #     b -= lr * b.grad
    
    # a.grad.zero_()
    # b.grad.zero_()

    optimizer.step()

    optimizer.zero_grad()


    
print(a, b)
# visualization
make_dot(loss)