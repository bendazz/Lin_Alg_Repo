import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd

torch.manual_seed(42)

data = pd.read_csv('data.csv')
features = torch.tensor(data.drop('Risk',axis = 1).values).float()

Y = torch.tensor(data['Risk'].map({'Healthy':0,'At Risk':1})).float().reshape(-1,1)

fm = features.mean(axis = 0, keepdim = True)
fs = features.std(axis = 0, keepdim = True)

X = (features - fm)/fs

model = nn.Linear(3,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(),lr=.1)

epochs = 250

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

torch.save({
    'fm':fm,
    'fs':fs,
    'parameters':model.state_dict()
},'model.pth')