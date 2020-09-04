#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:23:10 2019

@author: wuzhixin
"""

import torch
from torch.autograd import Variable
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad=True)
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)


v_out.backward()

print(t_out)
print(variable.grad)
#v_out = 1/4*sum(var*var)
#var 变成tensor,才能变成np
print(variable.data)