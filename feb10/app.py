import torch
import pandas as pd
import numpy as np    

df = pd.read_csv('data.csv')
X = torch.tensor(df.drop("Y",axis = 1).to_numpy()).float()
Y = torch.tensor(df["Y"].to_numpy()).reshape(-1,1)

w = torch.tensor([
    [-3.0],
    [-.3],
    [-2.6],
    [-1.5]
])

b = torch.tensor([
    [1.2]
])

Yhat = X@w + b

r = Yhat - Y  
loss = (r.T@r)/5
print(loss)