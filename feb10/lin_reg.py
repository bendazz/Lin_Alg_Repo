import torch

X = torch.tensor([
    [3.0],
    [6.0],
    [9.0]
])

Y = torch.tensor([
    [7.0],
    [2.0],
    [5.0]
])

w = torch.tensor([
    [0.0]
])

b = torch.tensor([
    [0.0]
])

Yhat = X@w+b
r = Y - Yhat
loss = r.T@r/3
print(loss)