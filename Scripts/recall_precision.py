"""
                    ACTUAL CLASS
 P
 R         True Positive | False Positive
 E         
 D          
 I         False Negative | True Negative 
 C
 T
 E
 D    
"""

data = [
    [18, 9],
    [12, 15]
]
TP = data[0][0]
FP = data[0][1]
FN = data[1][0]
TN = data[1][1]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("precision: {0}".format(precision))
print("recall: {0}".format(recall))
print("accuracy: {0}".format(accuracy))
