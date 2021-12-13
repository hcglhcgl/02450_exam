# -*- coding: utf-8 -*-

#Quick calc for Jeffery and McNemmer

#Lecture 7
from toolbox_02450 import *
import scipy.stats as st

#Toolbox usage:
# mcnemar(y_true, yhatA, yhatB)

#Jeffery interval
n = 14 #observations
m = 8 #correct classified
alpha = 0.05 #always = 0.05

a = m + 0.5
b = n-m +0.5

#Confidence Intervals
CI_L = st.beta.ppf(alpha/2,a,b)
CI_H = st.beta.ppf(1-alpha/2,a,b)
theta = a/(a+b)

#Jeffery Intervals Results
print(f"a={a}")
print(f"b={b}")
print(f"CI_L={CI_L}")
print(f"CI_H={CI_H}")
print(f"Theta={theta}")

#Compare two classifiers (mcnemmer test)
n12 = 28
n21 = 35
N = n12+n21
m = min(n12,n21)
theta = 1/2 #always
# st.binom.cdf(x,n,p)
p_val = 2*st.binom.cdf(m,N,theta)
print(f"p-val={p_val:.5f}")