#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import numpy as np


class Highway(nn.Module):
    def __init__(self,x_conv_out:torch.tensor):#[batch, e_word]
        super(Highway,self).__init__()
        #创建两个Linear层 x_conv_out=e_word -> e_word
        self.x_proj_layer = nn.Linear(len(x_conv_out[0,:]), len(x_conv_out[0,:]), bias=True)
        self.x_gate_layer = nn.Linear(len(x_conv_out[0,:]), len(x_conv_out[0,:]), bias=True)


    def forward(self, x_conv_out) :
        #ones = torch.tensor(np.ones((2,2)),dtype=float)
        #self.x_proj_layer.weight = torch.nn.Parameter(ones)
        #self.x_gate_layer.weight = torch.nn.Parameter(ones)
        x_proj = torch.relu(self.x_proj_layer(x_conv_out))#(b,k,e_word)
        x_gate = torch.sigmoid(self.x_gate_layer(x_conv_out))#(b,k,e_word))


        batch_size = len(x_conv_out[:,0])
        #k_num = len(x_conv_out[0,:,0])
        x_highway = torch.zeros_like(x_conv_out)
        ### torch.mul(a,b):b中的每个元素在a中都乘一遍，返回shape = (b.shape*a.shape)
        for i_batch in range(batch_size):
            #for i_kernel in range(k_num):
            x_highway[i_batch,:] = torch.mul(x_proj[i_batch,:], x_gate[i_batch,:]) + torch.mul(torch.add(-x_gate[i_batch, :], 1), x_conv_out[i_batch,:])
                #(b,e_word)



        return x_highway

        _




### END YOUR CODE 

#if __name__ == '__main__':
    #test_x = torch.ones(2,3,2,dtype=float)
    #net = Highway(test_x)
    #net.x_proj_layer.weight = [[1,2],[3,4]]
    #net.x_gate_layer.weight = [[1, 2], [3, 4]]
    #word_test = net(test_x)


