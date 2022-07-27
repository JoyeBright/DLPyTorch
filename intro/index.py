import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)

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

linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])