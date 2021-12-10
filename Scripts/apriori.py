import numpy as np
import pandas as pd
from basic_operations import choose_mode
from apyori import apriori


def mat2transactions(X, labels):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        l = [labels[i] for i in l]
        T.append(l)
    return T


def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:
            conf = o.confidence
            supp = r.support
            x = ", ".join(list(o.items_base))
            y = ", ".join(list(o.items_add))
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)" % (x, y, supp, conf))
            frules.append((x, y))
    return frules


def print_rules(string_mat, min_s, min_c):
    """
    :param string_mat: string form of binary matrix
    :param min_s: minimum support value
    :param min_c: minimum confidence value
    :return:
    """
    data = choose_mode(string_mat)
    print(data)
    labels = []
    [labels.append(str(x)) for x in range(data.shape[1])]
    T = mat2transactions(data, labels)
    # Get the associations
    rules = apriori(T, min_support=min_s, min_confidence=min_c)
    print_apriori_rules(rules)


mat = '''Y AY Y AN OAY OAN PAY PAN
S1 1 0 1 0 1 0
S2 1 0 1 0 0 1
S3 0 1 0 1 1 0
S4 0 1 1 0 1 0
S5 0 1 1 0 1 0
NS1 0 1 1 0 1 0
NS2 0 1 0 1 1 0
NS3 1 0 0 1 0 1
NS4 0 1 1 0 1 0
NS5 0 1 1 0 1 0'''

print_rules(mat, 0.52, 0)
