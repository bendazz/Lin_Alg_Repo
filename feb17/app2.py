import torch
import torch.nn as nn  
import torch.optim as optim 

torch.manual_seed(1)

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

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = .01)

epochs = 1000

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model.weight)
print(model.bias)


X = torch.tensor([
    [6.0]
])

prediction = model(X)

print(prediction)