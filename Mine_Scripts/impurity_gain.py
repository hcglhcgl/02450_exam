# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:31:38 2021

@author: hc
"""
# classiﬁcation tree based on Hunt’s algorithm
# we use the classiﬁcation error impurity measure
no_observations_0 = 33+28+30+29
#We see biggest class in 0 has 33 observations
I_r = 1 - 33/no_observations_0
#We see biggest class in 1 has 5 observations
no_observations_1_left = 4+2+3+5
I_1_left = 1 - 5/no_observations_1_left
no_observations_1_right = (33-4)+(28-2)+(30-3)+(29-5)
I_1_right = 1 - 29/no_observations_1_right
#Biggest class in 1 has 1 observation, and theres one in total
I_2_left = 1 - 1/1
delta = I_r - ((no_observations_1_left/no_observations_0)*I_1_left) - I_2_left
print(delta)

#Attempt for specific split


no_observations_1_left = 4+2+3+5
I_1_left = 1 - 5/no_observations_1_left

#Biggest class in 1 has 1 observation, and theres one in total
no_observations_2_left = 0+1+0+0
I_2_left = 1 - 1/no_observations_2_left


no_observations_2_right= 4+1+3+5
I_2_right = 1 - 1/no_observations_2_right

delta = I_1_left - ((no_observations_2_right/no_observations_1_left)*I_2_right) - I_2_left
print(delta)