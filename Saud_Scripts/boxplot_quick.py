# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 00:01:48 2021

@author: sosi
"""

#Book Question 7.2

import matplotlib.pyplot as plt
import numpy as np
x1=[1]*40
x2=[2]*10
x3=[3]*10
x = x1+x2+x3
print(x)

plt.figure()
plt.boxplot(x)