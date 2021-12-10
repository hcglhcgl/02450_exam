from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

"""
score: array of real points
y: array of real classes assigned to each score
"""
score = np.array([5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4])
y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])

roc_auc_score = roc_auc_score(y, score)
print(roc_auc_score)

roc_x = []
roc_y = []
min_score = min(score)
max_score = max(score)
thr = np.linspace(min_score, max_score, 30)
FP = 0
TP = 0
N = sum(y)
P = len(y) - N

for (i, T) in enumerate(thr):
    for i in range(0, len(score)):
        if score[i] > T:
            if y[i] == 1:
                TP = TP + 1
            if y[i] == 0:
                FP = FP + 1
    roc_x.append(FP / float(N))
    roc_y.append(TP / float(P))
    FP = 0
    TP = 0

plt.scatter(roc_x, roc_y)
plt.plot([0, max(roc_x)], [0, max(roc_y)], marker='o')
plt.show()
