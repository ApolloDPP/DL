# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:04:33 2020

@author: ApolloDPP
"""

import torch
import torch.nn.functional as F
import numpy as np

a=np.random.rand(10)

#调用pytorch中softmax
a=torch.tensor(a)
b=F.softmax(a,dim=None)
print('pytorch中softmax结果：',b)

#按照数学定义实现softmax函数
def softmax_1(array):
    index=len(array)
    exp_array=np.zeros(index)
    softmax_out=np.zeros(index)
    
    for i in range(index):
        exp_array[i]=np.exp(array[i])

    sum_exp=np.sum(exp_array)
    for j in range(index):
        softmax_out[j]=exp_array[j]/sum_exp

    return softmax_out

#调用自定义softmaxx函数
c=softmax_1(a)
print('自定义中softmax结果：',c)
    

