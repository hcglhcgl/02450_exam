# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:55:20 2021

@author: hc
"""
from matplotlib.pyplot import figure, show
from toolbox_02450 import rocplot, confmatplot

#The probabilities that a guess is correct:
p=[0.45,0.55,0.6,0.68,0.72,0.75,0.9,0.91]


#The true class labels:
#y=[1,1,0,0,0,1,0,1]
#y=[0,0,1,1,0,0,1,1]
y=[1,0,0,1,1,0,0,1]
#y=[0,1,0,1,0,1,0,1]

figure(1)
rocplot(p, y)

show()