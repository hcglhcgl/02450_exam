


tp = 36
tn = 39
fp = 10
fn = 15

precision = tp/(tp+fp)
recall = tp/(tp+fn)
trueNegativeRate = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
error = (fp+fn)/(fp+fp+tn+fn)

predictedPositiveConditionRate= (tp+fp)/(tp+fp+tn+fn)
Fmeasure = 2*(precision*recall)/(precision+recall)