# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:31:42 2021

@author: hc
"""

# Single hidden layer - 10 
n_h = 10
# use the over-parameterized softmax function described in section 14.3.2 

# hidden layer we will use a sigmoid non-linearity

#Each hidden unit has as many input
#unit weights are there are features M = 7 plus one
#(the bias), 
M=7
unit_params =(M+1)*n_h


#The softmax is computed deterministically
#from C = 4 units (as many as there are classes in the
#output dataset)
C = 4
softmax = (n_h+1)*C

param = unit_params + softmax
print(param)