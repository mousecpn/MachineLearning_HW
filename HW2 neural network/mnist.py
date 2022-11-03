import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import alexnet,AlexNet_Weights
epochs = 10
batch_size = 64
lr = 0.01
momentum = 0.5
log_interval = 10

# dataset init
train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True, 
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.RandomCrop((24,24)),
                                               transforms.Normalize((0.1307,),(0.3081,))
                                           ])
                                          )
test_dataset = torchvision.datasets.MNIST('./data/', train=False, download=True, 
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,),(0.3081,))
                                           ])
                                          )
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

# Model
class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(64),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(128),nn.ReLU())

        self.pool = nn.MaxPool2d(stride=2,kernel_size=2)
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # torch.sum(x).backward()
        x = self.pool(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x


# model init
# model = Model().cuda()
model = alexnet(weights=AlexNet_Weights).cuda()
CELoss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

# training
batch_done = 0
logs = []
for i in range(epochs):
    model.train()
    scheduler.step()
    for data, label in train_loader:
        data = F.interpolate(data,(224,224))
        data = torch.cat((data,data,data),1)
        output = model(data.cuda())
        loss = CELoss(output,label.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_done += 1
        if batch_done % log_interval == 0:
            logs.append([batch_done,loss.item()])
            print('Epoch {}: {}/{} loss:{}'.format(i, (batch_done)%len(train_loader), len(train_loader), loss.item()))
    # evaluation
    model.eval()
    correct = 0
    for data, label in test_loader:
        data = F.interpolate(data,(224,224))
        data = torch.cat((data,data,data),1)
        output = model(data.cuda())
        _,pred = torch.max(output, dim=1)
        correct += float(torch.sum(pred == label.cuda()))

    print('test_acc:{}'.format(correct/len(test_dataset)))


# loss curve visualization
logs = np.array(logs)
# plt.plot(logs[:,0],logs[:,1])

