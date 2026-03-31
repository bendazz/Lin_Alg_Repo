from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from grid import save_image_grid
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,),(0.3081,))
    ])
)

test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,),(0.3081,))
    ])
)



loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 1000,
    shuffle = False
)


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = .001)
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    total = 0
    correct = 0
    for images,labels in loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += labels.size(0)
        correct += (output.argmax(1) == labels).sum().item()
    test_total = 0
    test_correct = 0
    with torch.no_grad():
        for images,labels in test_loader:
            output = model(images)
            test_total += labels.size(0)
            test_correct += (output.argmax(1) == labels).sum().item()


    #print(total_loss/len(loader))
    print(correct/total)
    print(test_correct/test_total)
    print("---------------")

torch.save(model.state_dict(),'model.pth')
    




