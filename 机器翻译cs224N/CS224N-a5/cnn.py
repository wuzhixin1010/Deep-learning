#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
#e_word = 50    #每个词的embedding_size
#k = 5
#m_word = 21 #每个单词的字符个数
#e_char = 4   #每个字母的embedding_size

class CNN(nn.Module):
    def __init__(self, e_char:int, e_word:int, k:int, m_word ):
        super(CNN,self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=k, stride=1, bias=True)
        self.max_pool = nn.MaxPool1d(m_word-k+1)

    def forward(self, x_reshaped: torch.Tensor) :#(batch, input_channel=e_char, m_word)
        x_conv = self.conv(x_reshaped) #(batch, output_channel=filter_num=e_word, m_word-k+1)
        x_conv_out = self.max_pool(torch.relu(x_conv))#(batch,filter,1)

        return x_conv_out   #(batch,filter,1)






### END YOUR CODE

if __name__ == '__main__':
    batch_size = 2
    e_char = 4
    m_word = 10
    x_test = torch.randn(batch_size, e_char, m_word)
    net = CNN()
    x_test_out = net(x_test)
    print(x_test_out.shape)