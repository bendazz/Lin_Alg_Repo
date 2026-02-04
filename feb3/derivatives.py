import torch

x = torch.tensor(-2.0,requires_grad = True)
y = torch.tensor(-1.0,requires_grad = True)
f = (2*y**2 + 2*x*y - y**2*x**2) / (3*x*y**2 + 3*x**2 + 4*y*x + 3)
f.backward()
print(x.grad)
print(y.grad)



