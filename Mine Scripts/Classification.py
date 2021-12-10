import numpy as np
from math import exp

w1=np.array([[1.2],[-2.1],[3.1]])
w2=np.array([[1.2],[-1.7],[2.9]])
w3=np.array([[1.3],[-1.1],[2.2]])


obsA=(np.array([1,-1.4,2.6]))
obsB=(np.array([1,-0.6,-1.6]))
obsC=(np.array([1,2.1,5]))
obsD=(np.array([1,0.7,3.8]))


yA1=obsA@w1
yA2=obsA@w2
yA3=obsA@w3

yB1=obsB@w1
yB2=obsB@w2
yB3=obsB@w3

yC1=obsC@w1
yC2=obsC@w2
yC3=obsC@w3

yD1=obsD@w1
yD2=obsD@w2
yD3=obsD@w3


pA4 = 1 / (1+(exp(yA1) + exp(yA2) + exp(yA3)))


pB4 = 1 / (1+(exp(yB1) + exp(yB2) + exp(yB3)))


pC4 = 1 / (1+(exp(yC1) + exp(yC2) + exp(yC3)))


pD4 = 1 / (1+(exp(yD1) + exp(yD2) + exp(yD3)))


print("For observation A, P(y=4|yhat) is %.4f \n" %(pA4))
print("For observation B, P(y=4|yhat) is %.4f \n" %(pB4))
print("For observation C, P(y=4|yhat) is %.4f \n" %(pC4))
print("For observation D, P(y=4|yhat) is %.4f \n" %(pD4))