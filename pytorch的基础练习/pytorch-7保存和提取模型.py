#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:24:32 2019

@author: wuzhixin
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)#一维变二维
y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x), Variable(y)


def save():
    net1 = torch.nn.Sequential(torch.nn.Linear(1,10),
                        torch.nn.ReLU(),
                        torch.nn.Linear(10,1))
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.3)
    loss_func = torch.nn.MSELoss()
    
    for t in range(100):
        prediction1 = net1(x)
        loss = loss_func(prediction1,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(), prediction1.data.numpy(),'r-',lw=5)
    plt.show()
    torch.save(net1,'net.pkl')#entire net
    torch.save(net1.state_dict(),'net_params.pkl')#patameters
    
    
    
def restore_net():
    net2 = torch.load('net.pkl')
    prediction2 = net2(x)
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(), prediction2.data.numpy(),'r-',lw=5)
    plt.show()
    
    
def restore_params():
    net3 = torch.nn.Sequential(torch.nn.Linear(1,10),
                        torch.nn.ReLU(),
                        torch.nn.Linear(10,1))
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction3 = net3(x)
    
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(), prediction3.data.numpy(),'r-',lw=5)
    plt.show()
    
save()
restore_net()
restore_params()
