#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:16:41 2019

@author: wuzhixin
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.zeros(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)#int

x,y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_out):
        super(Net,self).__init()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_out)
        
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
#methond 2
net2 = torch.nn.Sequential(torch.nn.Linear(2,10),
                           torch.nn.ReLU(),
                           torch.nn.Linear(10,2))
print(net2)