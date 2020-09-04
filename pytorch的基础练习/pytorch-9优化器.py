#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:05:10 2019

@author: wuzhixin
"""

import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


x = torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))


torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_out):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.prediction = torch.nn.Linear(n_hidden,n_out)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.prediction(x)
        return x
net_SGD = Net(1,10,1)
net_Momentum = Net(1,10,1)
net_RMSprop = Net(1,10,1)
net_Adam = Net(1,10,1)
nets = [net_SGD, net_Adam,net_RMSprop,net_Momentum]


optimizer_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
optimizer_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizer_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
optimiezer_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
optimiezers = [optimizer_SGD, optimizer_Adam, optimizer_RMSprop, optimiezer_Momentum]


loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]

for epoch in range(EPOCH):
    print(epoch)
    for step,(batch_x,batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        
        for net, opt, l_his in zip(nets,optimiezers,losses_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data)
labels = [ 'SGD','Adam','RMSprop','Momentum']
for i ,l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('steps')
plt.ylabel('loss')
plt.ylim((0,0.2))
plt.show()