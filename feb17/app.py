import torch

X = torch.tensor([
    [2.0],
    [5.0],
    [8.0]
])

Y = torch.tensor([
    [3.0],
    [7.0],
    [1.0]
])

w = torch.tensor([
    [0.0]
],requires_grad = True)

b = torch.tensor([
    [0.0]
],requires_grad = True)

lr = .01

epochs = 1000 

for epoch in range(epochs):
    Yhat = X@w + b
    r = Yhat - Y 
    loss = r.T@r/3
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad 
        b -= lr*b.grad  
    w.grad.zero_()
    b.grad.zero_()
 



X = torch.tensor([
    [6.0]
])

prediction = X@w+b

print(prediction)