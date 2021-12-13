import numpy as np
from math import exp

w1=np.array([[423.49],[48.16]])
w2=np.array([[0.0],[-46.21]])
w3=np.array([[0.0],[-27.89]])
w4=np.array([[418.94],[-26.12]])

# Always put 1 as first number for some reason.
obsA=(np.array([1,14]))
obsB=(np.array([1,16]))
obsC=(np.array([1,18]))


yA1=obsA@w1
yA2=obsA@w2
yA3=obsA@w3
yA4=obsA@w4

yB1=obsB@w1
yB2=obsB@w2
yB3=obsB@w3
yB4=obsB@w4

yC1=obsC@w1
yC2=obsC@w2
yC3=obsC@w3
yC4=obsC@w4


pA1 = 1 / (1+(exp(-yA1)))

pB1 = 1 / (1+(exp(-yB1)))

pC1 = 1 / (1+(exp(-yC1)))

print("For observation A, weight 1 P(y=1|yhat) is %.4f \n" %(pA1))
print("For observation B, weight 1 P(y=1|yhat) is %.4f \n" %(pB1))
print("For observation C, weight 1 P(y=1|yhat) is %.4f \n" %(pC1))

pA2 = 1 / (1+(exp(-yA2)))

pB2 = 1 / (1+(exp(-yB2)))

pC2 = 1 / (1+(exp(-yC2)))

print("For observation A, weight 2 P(y=4|yhat) is %.4f \n" %(pA2))
print("For observation B, weight 2 P(y=4|yhat) is %.4f \n" %(pB2))
print("For observation C, weight 2 P(y=4|yhat) is %.4f \n" %(pC2))

pA3 = 1 / (1+(exp(-yA3)))

pB3 = 1 / (1+(exp(-yB3)))

pC3 = 1 / (1+(exp(-yC3)))

print("For observation A, weight 3 P(y=4|yhat) is %.4f \n" %(pA3))
print("For observation B, weight 3 P(y=4|yhat) is %.4f \n" %(pB3))
print("For observation C, weight 3 P(y=4|yhat) is %.4f \n" %(pC3))

pA4 = 1 / (1+(exp(-yA4)))

pB4 = 1 / (1+(exp(-yB4)))

pC4 = 1 / (1+(exp(-yC4)))

print("For observation A, weight 4 P(y=4|yhat) is %.4f \n" %(pA4))
print("For observation B, weight 4 P(y=4|yhat) is %.4f \n" %(pB4))
print("For observation C, weight 4 P(y=4|yhat) is %.4f \n" %(pC4))