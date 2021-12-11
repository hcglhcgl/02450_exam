# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 02:53:42 2021

@author: sosi
"""

#Function to graph the boundaries (in 2d) of a classification problem
#Known: Classification functions
#Unknown: How will the 2d plot look like after applying the classification funs?

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def solve_and_plot(df, solutions, xlabel="x1", ylabel="x2"):
    """
    Classify x1,x2 based on the decision rules
    Input: df --> entire dataframe. It must contain the discrimination function results (dx1,dx2...dxn)
           solutions: Functions used to classify. These are the question choices
    Output: None. Plots the resulting classification to show boundaries
    """
    
    for sol in range(len(solutions)):
        df["Category"] = solutions[sol](df.dx1) #might need to change the input (dx1,dx2)
        #plotting
        groups = df.groupby("Category")
        plt.figure()
        print(f"Solution {sol+1} status:")
        print(df["Category"].value_counts(), "\n")
        for name, group in groups:
            plt.plot(group.x1, group.x2, marker="o", linestyle="", label=name, color=name)
        plt.legend()
        plt.title(f"Option {sol+1}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

#How many random data points
size = 10000

#Random data boundaries (match the exam question scale)
low = -1
high = 1

#Classes; most problems have only 2 or 3
class0="black"
class1="red"
# class3="yellow"
# class4="green"
# class5="black"

w0=2
w1=0
w2=0
w3=10

#Classificaiton functions (given in the question):
z1 = lambda w0,w1,w2,w3,x1,x2:1/(1+np.exp(-(w0+w1*x1+w2*x2+w3*x1*x2)))
z2 = lambda :1-z1
# z2 = lambda x,y:np.maximum(0,x2-1) + y-2 #two vars version

#Answear options as functions
# A = lambda dx1,dx2: np.where((dx1 == 0) & (dx2 == 0),class1,class2)
# B = lambda dx1,dx2: np.where((dx1 == 1) & (dx2 == 1),class1,class2)
# C = lambda dx1,dx2: np.where((dx1 == 1) & (dx2 == 0),class1,class2)
D = lambda dx1: np.where(dx1 > 0.5,class1,class0)

#COMMENT OUT THE OPTIONS YOU DON'T WANT (CTRL+1); to save time, test 1 option
q_options = [
# A,
# B,
# C,
D
]

#Generate random data
x1 = np.random.uniform(low=low, high=high, size=size)
x2 = np.random.uniform(low=low, high=high, size=size)
df = pd.DataFrame({"x1":x1, "x2":x2})

#Apply the classificaiton functions to the data
df["dx1"]=z1(w0,w1,w2,w3,df.x1,df.x2)
df["dx2"]=z2

#Plot the Q answears based on the classificaiton functions
solve_and_plot(df,q_options, xlabel="x1", ylabel="x2")