# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:31:42 2021

@author: hc
"""

# Single hidden layer - 50 units
n_h = 50
# use the over-parameterized softmax function described in section 14.3.2 

# hidden layer we will use a sigmoid non-linearity activation function (is it important ?)

#Each hidden unit has as many input
#unit weights are there are features, eg. 8, features: M = 8 
#plus one (the bias), 
M=8
unit_params =(M+1)*n_h


#The softmax is computed deterministically
#from C units (as many as there are classes in the output dataset), eg: 9 output classes : C = 9
C = 9
softmax = (n_h+1)*C

param = unit_params + softmax
print(param)