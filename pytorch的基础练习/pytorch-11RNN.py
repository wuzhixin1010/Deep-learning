#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:23:16 2019

@author: wuzhixin
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
INPUT_SIZE = 28
TIME_SIZE = 28

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     #训练数据
    transform=torchvision.transforms.ToTensor(),
# 将PIL.Image 或 numpy.ndarray 转化为torch.FloatTensor大小为shape (C x H x W)并且在[0.0, 1.0]上标准化
    download=DOWNLOAD_MNIST,
)

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        
        self.rnn = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=64,
                num_layers=1,
                batch_first=True,)
        self.out = nn.Linear(64,10)
        
    def forward(self,x):
        r_out, (h_n,h_c) = self.rnn(x,None)#lstm包含两个hidden_state
        out = self.out(r_out[:,-1,:])#(batch,time step,input)
        return out
rnn = RNN()
print(rnn)

net = RNN()

optimizer = torch.optim.Adam(net.parameters, lr=LR)
loss_func = torch.nn.MSELoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # 清楚梯度
        loss.backward()                                 # 反向传播，计算梯度
        optimizer.step()                                #应用梯度，优化

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

#测试
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
        
