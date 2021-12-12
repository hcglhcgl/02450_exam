import numpy as np
from scipy.special import comb

# There is a comb function for Python which does 'n choose k'                                                                                            
# only you can't apply it to an array right away                                                                                                         
# So here we vectorize it...                                                                                                                             
def myComb(a,b):
  return comb(a,b,exact=True)

vComb = np.vectorize(myComb)

def get_tp_fp_tn_fn(cooccurrence_matrix):
  tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int),2).sum()
  tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int),2).sum()
  tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
  fp = tp_plus_fp - tp
  fn = tp_plus_fn - tp
  tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

  return [tp, fp, tn, fn]

if __name__ == "__main__":                                                                                                                       
  cooccurrence_matrix = np.array([[ 114,  0,  32], [ 0,  119,  0], [ 8,  0,  60]])

  # Get the stats                                                                                                                                        
  tp, fp, tn, fn = get_tp_fp_tn_fn(cooccurrence_matrix)

  print ("TP: ",tp," FP: ",fp, "TN: ",tn, "FN: ",fn)

  # Print the measures:                                                                                                                                  
  print ("Rand index: ", (float(tp + tn) / (tp + fp + fn + tn)))

  precision = float(tp) / (tp + fp)
  recall = float(tp) / (tp + fn)

  print ("Precision: ", precision)
  print ("Recall: ", recall)
  print ("F1: " , ((2.0 * precision * recall) / (precision + recall)))