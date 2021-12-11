# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 05:22:34 2021

@author: sosi
"""

#Function to graph the boundaries (in 2d) of a classification problem
#Known: Classification functions
#Unknown: How will the 2d plot look like after applying the classification funs?

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_boundary(df):
    """
    Plot the class boundaries.
    Input: Pandas dataframe that must contain the following:
        x1: x-axis data, usually randomally generated
        x2: y-axis data, usually randomally generated
        Category: The categories for each observation, as predicted by a class rule
    Output: Labeled plot of the decision boundaries
    """
    groups = df.groupby("Category")
    plt.figure()
    for name, group in groups:
        plt.plot(group.x1, group.x2, marker="o", linestyle="", label=name, color=name)
        plt.legend()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
#How many random data points
size = 10000

#Random data boundaries (match the exam question scale)
low = -1
high = 1

#Generate random data
x1 = np.random.uniform(low=low, high=high, size=size)
x2 = np.random.uniform(low=low, high=high, size=size)
df = pd.DataFrame({"x1":x1, "x2":x2})

#Classes; most problems have only 2 or 3. Use color names for plotting
class0="black"
class1="red"
# class2="yellow"

#Classificaiton functions and rules(given in the question):
#Lambda functions format in python: lambda [input vars]:[functions statement]
cA = lambda x1,x2:np.linalg.norm(np.array((x1,x2)).T,axis=1) >= 0.5
cB = lambda x1,x2:np.linalg.norm(np.array((x1+1,x2-1)).T,np.inf,axis=1) >1
cC = lambda x1,x2:np.linalg.norm(np.array((x1-1,x2+1)).T,1,axis=1) >2
dec_tree = lambda A,B,C: np.where((A & B) | (~A & C),class1,class0) #if true, class 1, else: class 0

c_funcs = [
    cA,
    cB,
    cC
    ]

#Apply the classification funcs to the data IN ORDER of the c_funcs list:
for i in range(len(c_funcs)):
    df[f"dx{i+1}"]=c_funcs[i](df.x1,df.x2)
#Debugging:
# df["A"] = df.dx1 >= 0.5
# df["B"] = df.dx2 > 1.0
# df["C"] = df.dx3 > 2
df["Category"] = dec_tree(df.dx1,df.dx2,df.dx3) #add/del more variables if needed (df.dx1,df.dx2...)

#Plotting
plot_boundary(df)