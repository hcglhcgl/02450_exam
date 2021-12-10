from sklearn import metrics
import matplotlib.pyplot as plt

trueLabels = [1,1,0,1,0,0]
predicted = [1,1,1,0,0,0]
fpr, tpr, _ = metrics.roc_curve(trueLabels,predicted)
plt.plot(fpr,tpr)

print("AUC = ",metrics.roc_auc_score(trueLabels,predicted))