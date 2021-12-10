import sklearn as sk
from sklearn import metrics
import numpy as np
import similarityMeasuresAll as simil



o1 = [1,0,1,0,0,1] #12
o2 = [1,0,1,0,1,0] 
o3 = [1,0,0,0,1] #13
o4 = [1,1,1,0,0,1]
o5 = [1,0,1,0,0,1]
o6 = [0,0,1,1,0,1]
o7 = [1,1,1,1,1,1]
o10 = [0,1,0,1,0,1,1,0]
vector1 = o1
vector2 = o2

jaccard = simil.similarity(vector1,vector2,method = "Jaccard")
cosine = simil.similarity(vector1,vector2,method = "Cosine") 
extendedJacc = simil.similarity(vector1,vector2,method = "ExtendedJaccard")
correlation = simil.similarity(vector1,vector2,method = "Correlation") 
smc = simil.similarity(vector1,vector2,method = "SMC")
print("jac = ",jaccard)
print("smc = ",smc)
print("cos = ",cosine)
'''
           'SMC', 'smc'             : Simple Matching Coefficient
           'Jaccard', 'jac'         : Jaccard coefficient 
           'ExtendedJaccard', 'ext' : The Extended Jaccard coefficient
           'Cosine', 'cos'          : Cosine Similarity
           'Correlation', 'cor'     : Correlation coefficient
'''
from scipy.misc import comb

labels = [0,1,0,0,0,0,1,1]  #true labels of data corresponding to the array order below
clustered= [0,0,0,0,0,1,1,1] #cluster number each data point belongs to. Absolute number is irrelevant.


def rand_index_score(clusters, classes):

    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

rand = rand_index_score(clustered,labels)
print("Rand: ",rand)


#jaccard
import itertools 

def jaccardIndex(labels1, labels2):
    """
    Computes the Jaccard similarity between two sets of clustering labels.

    The value returned is between 0 and 1, inclusively. A value of 1 indicates
    perfect agreement between two clustering algorithms, whereas a value of 0
    indicates no agreement. For details on the Jaccard index, see:
    http://en.wikipedia.org/wiki/Jaccard_index

    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)

    @param labels1 iterable of cluster labels
    @param labels2 iterable of cluster labels
    @return the Jaccard similarity value
    """
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)



jaccIndex = jaccardIndex(labels,clustered)
print("Jaccard Index (for clusters): ",jaccIndex)