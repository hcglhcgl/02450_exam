# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:31:42 2021

@author: hc
"""

# Single hidden layer - 6 units
n_h = 6
# use the over-parameterized softmax function described in section 14.3.2 

# hidden layer we will use a sigmoid non-linearity activation function (is it important ?)

#Each hidden unit has as many input
#unit weights are there are features, eg. 7, features: M = 7 
#plus one (the bias), 
M=4
unit_params =(M+1)*n_h


#The softmax is computed deterministically
#from C units (as many as there are classes in the output dataset), eg: 4 output classes : C = 4
C = 3
softmax = (n_h+1)*C

param = unit_params + softmax
print(param)