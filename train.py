import pandas as pd
import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim  

data = pd.read_csv('linearly_separable_2d.csv')
X = torch.tensor(data.drop('y',axis=1).values).float()
Y = torch.tensor(data['y'].values).float().reshape(-1,1)

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs,Y)
    loss.backward()
    optimizer.step()
    print(loss.item())

torch.save(model.state_dict(),'model.pth')

