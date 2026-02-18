import torch 
import torch.nn as nn
import torch.optim as optim 
import pandas as pd  
import numpy as np  

data = pd.read_csv('data.csv')

features = torch.tensor(data.drop('Price',axis = 1).to_numpy()).float()
target = torch.tensor(data['Price'].to_numpy()).float().reshape(-1,1)

fm = features.mean()
fs = features.std()
tm = target.mean()
ts = target.std()

X = (features - fm)/fs
Y = (target - tm)/ts

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = .1)

epochs = 100

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

features = torch.tensor([
    [1500.0]
])

X = (features - fm)/fs
prediction = model(X)
print(prediction*ts + tm)