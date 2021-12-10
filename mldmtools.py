import pandas as pd
import numpy as np
import itertools as IT

"""
Use this toolbox either as an imported module, or in e.g. ipython with the magic %paste command 
to import w.e. is in you clipboard; that could be the jaccard function or one of the other tools.

Also, cheap hack to import it as a module:

import sys
sys.path.append(r'/Users/Thorbjorn/DTU/1st semester/Data mining/MLDMExam') # <-- put your path here
from mldmtools import *

Note to self:
Itemset finder with regex

^\s+([01]\s+){1}1\s+([01]\s+){4}1
"""



def jaccard(a,b):
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    _11 = 0
    _not00 = 0
    for x,y in zip(a,b):
        if x and y:
            _11 += 1
            _not00 += 1
        elif x or y:
            _not00 += 1
    return float(_11)/float(_not00)

"""
Ugly use of jaccard:
for i1 in range(5):
    for i2 in range(5):
        if i1 >= i2:
            continue
        print i1+1, i2+1, 1 - jaccard(_ns[i1], _ns[i2])
"""

def simple_matching_coefficient(a,b):
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    matching = 0
    for x,y in zip(a,b):
        if x == y:
            matching += 1
       
    return float(matching)/len(a)

def cosine_similarity(a,b):
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    dp = dot(a,b)
    eLenA = euclidian_length(a)
    eLenB = euclidian_length(b)

    return dp / (eLenA * eLenB)

def dot(a,b):
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    return sum([x*y for x,y in zip(a,b)])

def euclidian_length(a):
    return np.sqrt(sum([x*x for x in a]))

def gini(vec):
    # It is 1 minus the sum over i of p_i^2, where p_i is the fraction of records belonging to class i.
    asseries = pd.Series(vec)
    valc = asseries.value_counts()
    normed = valc / len(vec)
    powered = normed * normed
    summed = powered.sum()
    return (1 - summed)

def classification_error(vec):
    # 1 - max over i(p(i|t)), 
    # p(i|t) is the fraction of objects belonging to class i on node t
    return 1 - float(pd.Series(vec).value_counts()[0])/len(vec)

def purity_gain(parent, children, measure_method=gini):
    """
    Usage: purity_gain([1,0,1,0], [[0,0],[1,1]], gini)
    Example from Spring2011 Q2:
        bsplit = list('s'*12 + 'w'*10 + 'c'*10 + 'u'*8) # REMEMBER to put them in a list!
        a1 = list('s'*4 + 'w'*8 + 'c'*3 + 'u'*1)
        a2 = list('s'*8 + 'w'*2 + 'c'*7 + 'u'*7)

        purity_gain(bsplit, [a1, a2], classification_error)
    """
    # break early:
    children[0][1] # breaks if you're e.g. just passing a 1d list of chars
    #It is the impurity of the parent minus the sum over i of 
    #   (the number of records associated with the child node i divided by the total number of records in the parent node, 
    #   multiplied by the impurity measure of the child node i)
    pval = measure_method(parent)
    pl = float(len(parent))
    chvals = [measure_method(x) * len(x) / pl for x in children]
    
    return pval - sum(chvals)


def least_square(A, y):
    """
    Intercept is placed first
    """
    A = np.vstack([np.ones(len(A)), A]).T
    return np.linalg.lstsq(A, y)[0]


def standardise(vec, ddof=0):
    """
    subtract mean, then divide by standard deviation
    ddof is used as N - ddof in the divisor in std
    """
    vm = np.mean(vec)
    vs = np.std(vec, ddof=ddof)
    return [(x - vm)/vs for x in vec]

def pca_accountability(singular_values):
    """
    Answers the eternal question: How much variability does the nth principal component account for?
    Returns the fraction of variability accounting for each of the elements in singular_values, order preserved.
    """

    squared = [x*x for x in singular_values]
    svsum = float(sum(squared))
    ratio = [x / svsum for x in squared]
    return ratio


def classification_stats(TP, TN, FP, FN):
    count = TP + TN + FP + FN
    error_rate = float(FP + FN) / count
    TPR = TP / float(TP + FN)
    FPR = FP / float(TN + FP)
    FNR = FN / float(TP + FN)
    TNR = TN / float(TN + FP)
    sensitivity = TPR
    specificity = TN
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    F1 = 2 * TP / float(2 * TP + FP + FN)

    output = {
        'count': count,
        'error_rate': error_rate,
        'TPR': TPR,
        'FPR': FPR,
        'FNR': FNR,
        'TNR': TNR,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'F1': F1
    }

    return output


def support(itemset):
    """
    Returns the support value for the given itemset
    itemset is a pandas dataframe with one row per basket, and one column per item
    """

    # Get the count of baskets where all the items are 1
    baskets = itemset.iloc[:,0].copy()
    for col in itemset.columns[1:]:
        baskets = baskets & itemset[col]

    return baskets.sum() / float(len(baskets))

def confidence(df, antecedentCols, consequentCols):
    """
    df is a pandas dataframe
    antecedentCols are the labels for the columns/items that make up the antecedent in the association rule
    consequentCols are the labels for the columns/items that make up the consequent in the association rule
    """
    top = support(df[antecedentCols + consequentCols])
    bottom = support(df[antecedentCols])

    return top/bottom



def itemsets(df, support_min):
    """
    df is a pandas dataframe, with each row being a basket, and each column being an item
    You can really really really benefit from writing the table in a text editor, copying it,
    and using pd.from_clipboard().
    """
    itemsets = []
    n = len(df)
    for itsetSize in np.arange(1, len(df.columns) + 1): # Start with 1-itemsets, keep going till n_attributes-itemsets
        for combination in IT.combinations(df.columns, itsetSize):
            sup = support(df[list(combination)])
            if sup > support_min:
                itemsets.append(set(combination))

    return itemsets



# Consider implementing k-nearest neighbours for 


# Wrapper methods

def mean(vals):
    return pd.Series(vals).mean()

def median(vals):
    return pd.Series(vals).median()

def mode(vals):
    return pd.Series(vals).mode()

def valuerange(vals):
    ser = pd.Series(vals)
    return ser.max() - ser.min()


