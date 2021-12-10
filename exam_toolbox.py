"""
Functions created to use for the exam in the course 02450 Introduction to machine learning and datamining at the Technical University of Denmark

Author: Lukas Leindals
"""
__version__ = "Revision: 2019-12-15"

import toolbox_extended as te  # pip install --ignore-installed ml02450
import toolbox_02450 as tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import re
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import label_binarize
from apyori import apriori
from sklearn.cluster import KMeans
import sklearn.metrics as metrics


class prep_tools:
    def latex_to_df(self, x, cols, show=True, O_names=False, use_int=False):
        """
        makes a data frame from a latex table
        this function works well with the tool "mathpix"

        this function only works for matrices with numericals, as only the numericals are inserted in the matrix
        ------------------------------------
        parameters:
        -----------
        cols = number of columns
        x = list with parts of table, must be inputtet as [["x1"], ["x2"]] where x1 and x2 are the latex code for the table, for just one string input [["x1"]]

        -------------------------------------------
        output:
        ---------
        obs ther outputtet df might have some trouble being subsettet, use df.loc[] for positional subsetting
        """

        df = pd.DataFrame()
        row = 0
        for i in range(len(x)):
            s = x[i][row]

            # s = re.sub(r"\{0\}", " 0 ", s)
            # s = re.sub("{", " ", s)
            # s = re.sub("}", " ", s)
            # s = re.sub(" 0 ", " 0.0 ", s)
            # s = re.sub(" 1 ", " 1.0 ", s)

            values = re.findall(r"-*\d+\.*\d*", s)
            values = [float(i) for i in values]

            if use_int:
                values = [int(i) for i in values]

            df_new = pd.DataFrame()

            for col in range(cols):
                if O_names:
                    df_new["O{}".format(col + 1)] = values[col::cols]
                else:
                    df_new[col] = values[col::cols]

            df = df.append(df_new)

            # if O_names:
            #     df = df.set_index(["O{}".format(i) for i in range(5)])
            # else:
            df = df.reset_index(drop=True)

        if show:
            print(df)
        return df

    def join_labels(self, labels):
        """
        labels = list of lists with indices for different labels
        """
        x = np.array(labels)

        n, m = x.shape

        label = np.zeros(n * m)

        for i in range(n):
            for j in x[i]:
                label[j - 1] = i

        label = [int(i) for i in label]

        return label


class pca_calc:
    def var_explained(self, S, plot=True, show_df=True):
        """
        S = list of variances of components (can be read from the S/Sigma matrix)
        plot = to plot or not
        show_df = to show df with varaince explained or not
        """
        S = np.array(S)

        # df with variance explained
        df_var_exp = pd.DataFrame(columns=["k", "var_explained"])
        for i in range(len(S)):
            t = np.sum(S[0 : i + 1] ** 2) / np.sum(S ** 2)
            df_var_exp.loc[i] = [i + 1, t]

        # plot of variance explained
        plt.plot(df_var_exp["k"], df_var_exp["var_explained"])
        plt.scatter(df_var_exp["k"], df_var_exp["var_explained"])
        plt.title("Variance explained")
        plt.xlim(np.min(df_var_exp["k"]), np.max(df_var_exp["k"]))
        plt.ylim(0, 1)
        if plot:
            plt.show()

        if show_df:
            print(df_var_exp)
        return df_var_exp

    def proj(self, V, x, components=None, V_full=False):
        """
        V = V matrix from pca
        components = principal components to project onto
        V_full = whether the V matrix contains entire V matrix (if not only the columns of the components to project on is given in V, fx for 2 compenents, V should be a N*2 matrix)
        x = observation to project
        """

        V = np.array(V)
        x = np.array(x)

        if V_full:
            assert components != None
            components = [c - 1 for c in components]
            V = V[:, components]
            # V = V.T
            proj_coord = x @ V
            # proj_coord = p[components]
            components = [c + 1 for c in components]
        else:
            # V = V.T
            proj_coord = x @ V

        print(
            "The projected coordinates on component {} are: {}".format(
                components, proj_coord
            )
        )
        return proj_coord


class cross_val:
    def trained_models(self, k1, k2, S, show_res=True):
        """
        k1 = number of outer folds
        k2 = number of inner folds
        S = number of models
        show_res = whether to show th final amount of trained models
        """

        runs = 0
        pr_model = 0
        for _ in range(k1 + 1):
            for _ in range(k2):
                pr_model += 1
                for _ in range(S):
                    runs += 1

        if show_res:
            print(
                "A total of {} models were trained, as {} models were trained {} times each.".format(
                    runs, S, pr_model
                )
            )

        return runs


class decision_trees:
    def entropy(self, v):
        v = np.array(v)
        l2 = [math.log2(v[i] / sum(v)) for i in range(len(v))]
        l2 = np.array(l2)

        return 1 - np.sum(((v / sum(v)) * l2))

    def purity_gain(self, root, v1, v2, purity_measure, accuracy=False):
        """
        root = list with the size of each class
        v1 = list with the size of each class in the first branch after the split
        v2 = list with the size of each class in the second branch after the split
        purity_measure = string with the purity measure to use
        """
        root = np.array(root)
        v1 = np.array(v1)
        v2 = np.array(v2)

        v = np.array([v1, v2])

        acc = (np.max(v1) + np.max(v2)) / np.sum(v)
        if accuracy:
            print("The accuracy of the split is {}".format(acc))

        Iv = 0

        if purity_measure == "gini":
            for i in range(2):
                Iv += te.gini(v[i]) * sum(v[i]) / sum(root)

            purity = te.gini(root) - Iv

        if purity_measure == "class_error":
            for i in range(2):
                Iv += te.class_error(v[i]) * sum(v[i]) / sum(root)

            purity = te.class_error(root) - Iv

        if purity_measure == "entropy":
            for i in range(2):
                Iv += self.entropy(v[i]) * sum(v[i]) / sum(root)

            purity = te.class_error(root) - Iv

        print(
            "The purity gain for the split is {} with the purity measure {}".format(
                purity, purity_measure
            )
        )
        return purity


class ensemble:
    def plot_confusion_matrix(
        self, cm, title="Confusion matrix", cmap=plt.cm.get_cmap(name="Blues")
    ):
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["positive", "negative"])  # , rotation=45)
        plt.yticks(tick_marks, ["positive", "negative"])
        plt.tight_layout
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        fmt = "d"  #'.2f' if normalize else 'd'
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.show()

    def conf_matrix_stats(self, TP, FN, FP, TN, stats):
        """
        makes confusion matrix

        TP = true positive
        FN = false negative
        FP = false positive
        FN = false negative
        stats = list of stats to output, should be inputtet as string. Options are:
                precision = "p"
                recall = "r"
                accuracy = "acc"
                error = "err"
                true positive rate = "tpr"
                false positive rate = "fpr"
                show confususion matrix = "show"
                Receiver operating characteristic plot (TPR~FPR plot) = "roc"
                show table with all values (list not necessary)= "all"
        """

        # calculations
        cm = np.array([[TP, FN], [FP, TN]])
        N = TP + FN + FP + TN
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        acc = (TP + TN) / N
        err = (FN + FP) / N
        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)

        if "p" in stats:
            print("The precision is {}".format(p))

        if "r" in stats:
            print("The recall is {}".format(r))

        if "acc" in stats:
            print("The accuracy is {}".format(acc))

        if "err" in stats:
            print("The error is {}".format(err))

        if "tpr" in stats:
            print("The true positive rate is {}".format(TPR))

        if "fpr" in stats:
            print("The false positive rate is {}".format(FPR))

        if "show" in stats:
            self.plot_confusion_matrix(cm)

        if "roc" in stats:
            plt.plot(FPR, TPR)
            plt.title("Receiver operating characteristic")
            plt.ylabel("TPR")
            plt.xlabel("FPR")
            plt.show()

        if "all" in stats:
            all = pd.DataFrame(
                {
                    "Stat": ["Precision", "Recall", "Accuracy", "Error", "TPR", "FPR"],
                    "Value": [p, r, acc, err, TPR, FPR],
                }
            )
            print(all)

    def make_conf_matrix(self, true_val, pred_val, show_conf_matrix=False):
        """
        calculates values for a confusion matrix and various stats
        ------------------------------------------------
        parameters:
        --------------------------------------------
        true_val = list of the correct labels, must be binarised so that 1 = positve class and 0 = negative class
        pred_val = list of the predicted labels, must be binarised so that 1 = positve class and 0 = negative class

        returns the amount of true positives, false positives, false negatives and true negatives
        """
        true_val = np.array(true_val)
        pred_val = np.array(pred_val)

        pred_pos = true_val[pred_val == 1]
        pred_neg = true_val[pred_val == 0]

        TP = np.sum(pred_pos == 1)
        FP = np.sum(pred_pos == 0)
        FN = np.sum(pred_neg == 1)
        TN = np.sum(pred_neg == 0)

        if show_conf_matrix:
            stats = ["all", "show"]
        else:
            stats = "all"

        self.conf_matrix_stats(TP=TP, FP=FP, FN=FN, TN=TN, stats=stats)

        return TP, FP, FN, TN

    def plot_roc(self, true_val, pred_val):
        """
        calculates the fpr and tpr and plots a roc curve
        to compare the outputtet graph with the possible answers, look at where the plot has a elbow
        -----------------------------------------------
        parameters:
        -----------
        true_val = list of the correct labels, must be binarised so that 1 = positve class and 0 = negative class
        pred_val = list of the predicted labels, must be binarised so that 1 = positve class and 0 = negative class

        returns the area under the curve (AUC)
        """
        fpr, tpr, _ = metrics.roc_curve(true_val, pred_val)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()

        return roc_auc


class supervised:
    def knn_dist_pred_2d(self, df, class1, class2, K, show=False):
        """
        calculates predictions given a matrix with euclidean distances, can only handle two classes: red and black
        -------------------------------------------------------
        class1 = list with numbers of observations in the red class (starts at 1)
        class2 = list with numbers of observations in the black class (starts at 1)
        """
        classes = {"red": class1, "black": class2}

        red_ind = [i - 1 for i in classes["red"]]
        black_ind = [i - 1 for i in classes["black"]]

        pred_label = []
        O = [i for i in range(1, df.shape[1] + 1)]

        for row in range(df.shape[0]):
            dist = df.loc[row, :].values
            # sort
            dist_sort = np.argsort(dist)
            k_nearest_ind = dist_sort[1 : K + 1]

            pred_red = 0
            pred_black = 0

            for i in range(K):
                if k_nearest_ind[i] in red_ind:
                    pred_red += 1
                elif k_nearest_ind[i] in black_ind:
                    pred_black += 1
            if pred_red > pred_black:
                pred_label.append("red")
            elif pred_black > pred_red:
                pred_label.append("black")
            elif pred_black == pred_red:
                if k_nearest_ind[0] in red_ind:
                    pred_label.append("red")
                else:
                    pred_label.append("black")
        true_label = []
        for obs in O:
            if obs - 1 in red_ind:
                true_label.append("red")
            elif obs - 1 in black_ind:
                true_label.append("black")

        predictions = pd.DataFrame(
            {"Obs": O, "True_label": true_label, "Predicted_label": pred_label}
        )

        if show:
            print("-" * 100)
            print("The predictions when using the {} nearest neighbors are: ".format(K))
            print(predictions)

        return predictions

    def knn_dist_pred(self, df, classes, K):
        """
        calculates predictions given a matrix with euclidean distances, can only handle two classes: red and black
        -------------------------------------------------------
        class1 = list with numbers of observations in the red class (starts at 1)
        class2 = list with numbers of observations in the black class (starts at 1)
        """

        classes = np.array(classes)

        pred_label = []
        O = [i for i in range(1, df.shape[1] + 1)]

        for row in range(df.shape[0]):

            dist = df.loc[row, :].values
            # sort
            dist_sort = np.argsort(dist)
            k_nearest_ind = dist_sort[1 : K + 1]

            pred_classes = classes[k_nearest_ind]

            unique, counts = np.unique(pred_classes, return_counts=True)

            neighbors = pd.DataFrame({"class": unique, "count": counts})

            lab = neighbors[neighbors["count"] == np.max(neighbors["count"])][
                "class"
            ].values

            if len(lab) == 1:
                pred_label.append(lab[0])
            else:
                pred_label.append(classes[dist_sort[1]])

        predictions = pd.DataFrame(
            {"Obs": O, "True_label": classes, "Predicted_label": pred_label}
        )

        return predictions

    def pred_stats(self, true_labels, pred_labels, show=False):
        """
        calculates stats for a classifier, based on the labels predicted
        --------------------------------------------------
        parameters:
        -----------
        true_labels = list of true labels
        pred_labels = list of predicted labels
        """
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        acc = np.mean(true_labels == pred_labels)
        err = 1 - acc
        results = pd.DataFrame(
            {"Stat": ["Accuracy", "Error rate"], "Value": [acc, err]}
        )

        if show:
            print("-" * 100)
            print("The stats for the predictions are:")
            print(results)

        return results

    def naive_bayes_2class(self, y, df, cols, col_vals, pred_class):
        """
        probability of a naive bayes classifier
        -------------------------------------
        parameters:
        ----------
        y = list of labels (must be 0 and 1's)
        df = data frame with binary data
        cols = columns to condition the probability on
        col_vals = the values the columns are condtioned on
        pred_class = the class you would like to predict the probability of (must be 0 or 1)
        """
        y = np.array(y)
        if pred_class == 1:
            t = np.mean(y)
            suby = df.iloc[y == pred_class, :]
            for i in range(len(cols)):
                p = np.mean(suby.loc[:, cols[i]] == col_vals[i])
                t *= p

            n = np.mean(y)
            suby = df.iloc[y == 0, :]
            for i in range(len(cols)):
                p = np.mean(suby.loc[:, cols[i]] == col_vals[i])
                n *= p

            prob = t / (n + t)

        if pred_class == 0:
            t = 1 - np.mean(y)
            suby = df.iloc[y == pred_class, :]
            for i in range(len(cols)):
                p = np.mean(suby.loc[:, cols[i]] == col_vals[i])
                t *= p

            n = 1 - np.mean(y)
            suby = df.iloc[y == 1, :]
            for i in range(len(cols)):
                p = np.mean(suby.loc[:, cols[i]] == col_vals[i])
                n *= p

            prob = t / (n + t)

        print(
            "The probability that the given class is predicted by the Naïve Bayes classifier is {}".format(
                prob
            )
        )
        return None

    def naive_bayes(self, y, df, cols, col_vals, pred_class):
        """
        probability of a naive bayes classifier
        -------------------------------------
        parameters:
        ----------
        y = list of labels (starting at 0)
        df = data frame with binary data
        cols = columns to condition the probability on (starts at 0)
        col_vals = the values the columns are condtioned on
        pred_class = the class you would like to predict the probability of (starts at 0)
        """
        y = np.array(y)

        probs = []
        for c in range(len(np.unique(y))):
            n = np.mean(y == c)
            suby = df.iloc[y == c, :]
            for i in range(len(cols)):
                p = np.mean(suby.loc[:, cols[i]] == col_vals[i])
                n *= p
            probs.append(n)

        prob = probs[pred_class] / np.sum(probs)

        print(
            "The probability that the given class is predicted by the Naïve Bayes classifier is {}".format(
                prob
            )
        )
        return None


class cluster:
    def fancy_dendrogram(self, *args, **kwargs):
        max_d = kwargs.pop("max_d", None)
        if max_d and "color_threshold" not in kwargs:
            kwargs["color_threshold"] = max_d
        annotate_above = kwargs.pop("annotate_above", 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get("no_plot", False):
            plt.title("Hierarchical Clustering Dendrogram (truncated)")
            plt.xlabel("sample index or (cluster size)")
            plt.ylabel("distance")
            for i, d, c in zip(ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, "o", c=c)
                    plt.annotate(
                        "%.3g" % y,
                        (x, y),
                        xytext=(0, -5),
                        textcoords="offset points",
                        va="top",
                        ha="center",
                    )
            if max_d:
                plt.axhline(y=max_d, c="k")
        return ddata

    def dendro_plot(
        self,
        dist_df,
        Method,
        labels=None,
        sort=False,
        cutoff=None,
        show=True,
        invert_xaxis=True,
    ):
        """
        plots dendrogram given a matrix of distances and a linakge method
        ---------------------------------------------------------
        dist = symmetrical matrix containing distances
        Method = linkage method:
            "single"
            "complete"
            "average"
        orientation = orientation of plot:
            "top"
            "bottom"
            "left"
            "right"
        labels: if not provided labels starting from O1 will be given
        sort: can be set to "ascending" or "descending"
        cutoff = height to cut the tree, if the cutoff line cuts 3 lines, clusters returns labels for 3 clusters

        Output:
            R= dendrogram data
            clusters = cluster labels (0 index), must set cutoff to output
        """

        if labels == None:
            labels = ["O{}".format(i) for i in range(1, dist_df.shape[1] + 1)]

        Z = squareform(dist_df)
        Z = linkage(Z, method=Method)
        # R = dendrogram(y, orientation = orientation, labels = labels, distance_sort = sort,truncate_mode='level', p=3)

        R = cluster.fancy_dendrogram(
            Z,
            leaf_rotation=90.0,
            leaf_font_size=12.0,
            show_contracted=True,
            annotate_above=10,
            max_d=cutoff,
            labels=labels,
        )
        plt.grid()
        if invert_xaxis:
            plt.gca().invert_xaxis()

        if show:
            plt.show()

        if cutoff != None:
            clusters = fcluster(Z, cutoff, criterion="distance")
            clusters = [c - 1 for c in clusters]
            return R, clusters
        else:
            return R

    def cluster_similarity(self, x, y):
        """
        Parameters
        ----------
        x : Cluster A (labels)

        y : Cluster B

        Returns: Similarity Index - Rand und Jaccard
        eller den returnere ikke noget, den printer bare
        -------
        Copyright Peter Pik,
        Danmarks Tekniske Universitet
        """
        x = np.array(x)
        np.array(y)
        f00 = 0
        f01 = 0
        f10 = 0
        f11 = 0
        N = len(y)
        for i in range(N):
            for j in range(i):
                if y[i] != y[j] and x[i] != x[j]:
                    f00 += 1
                    # different class, different cluster
                elif y[i] == y[j] and x[i] == x[j]:
                    f11 += 1
                    # same class, same cluster
                elif y[i] == y[j] and x[i] != x[j]:
                    f10 += 1
                    # same class, different cluster
                else:
                    f01 += 1
                    # different class, same cluster

        rand = np.float(f00 + f11) / (f00 + f01 + f10 + f11)
        jaccard = np.float(f11) / (f01 + f10 + f11)

        similarities = [rand, jaccard]
        names = ["Rand", "Jaccard"]
        # for i in range(2):
        #     print(names[i],": ", similarities[i])

        result = pd.DataFrame({"Measure": names, "Value": similarities})
        print(result)
        return result

    def kmeans_1d(self, x, k, init=None):
        """
        assigns cluster to values in a 1d array
        -----------------------------------------
        parameters:
        -----------
        x = list of values to cluster
        k = number of clusters
        init = values to initialize clusters at
        """
        x = np.array(x).reshape(-1, 1)
        if init == None:
            kmeans = KMeans(n_clusters=k).fit(x)
        else:
            init = np.array(init).reshape(-1, 1)
            kmeans = KMeans(n_clusters=k, init=init).fit(x)

        clusters = kmeans.predict(x)

        centers = []

        for c in np.unique(clusters):
            centers.append(np.mean(x[clusters == c]))

        centers = np.round(centers, 4)
        print("The assigned clusters are: {}".format(clusters))
        print(
            "The cluster centers of the converged k-means algortihm is: {}".format(
                centers
            )
        )
        return None


class similarity:
    def measures(self, x, y):
        """
        Parameters
        ----------
        x : Gruppe A
        y : Gruppe B

        Returns: Similarity - Cosinun, SMC, Jaccard
        eller den returnere ikke noget, den printer bare
        -------
        Copyright Peter Pik,
        Danmarks Tekniske Universitet
        """
        f00 = 0
        f01 = 0
        f10 = 0
        f11 = 0
        x = np.array(x)
        y = np.array(y)
        N = len(y)
        for i in range(N):

            if y[i] != 1 and x[i] != 1:
                f00 += 1
                # different class, different cluster
            elif y[i] == 1 and x[i] == 1:
                f11 += 1
                # same class, same cluster
            elif y[i] == 1 and x[i] != 1:
                f10 += 1
                # same class, different cluster
            else:
                f01 += 1
                # different class, same cluster
        jaccard = np.float(f11) / (f01 + f10 + f11)
        SMC = np.float((f11 + f00) / (f00 + f01 + f10 + f11))
        cos = x.T @ y / (np.linalg.norm(x) * np.linalg.norm(y))
        similarities = [SMC, jaccard, cos]
        names = ["SMC", "Jaccard", "cos"]
        # for i in range(3):
        #     print(names[i],": ", similarities[i])

        result = pd.DataFrame({"Measure": names, "Value": similarities})
        print(result)
        return result


class anomaly:
    def ARD(self, df, obs, K):
        """
        Calculates average relative density
        ----------------------
        parameters:
        ----------------------
        df = symmetric matrix with distances
        obs = the observation to calculate ARD for  (0 index)
        K = the number of nearest neighbors to consider
        """

        O = df.loc[obs, :].values
        dist_sort = np.argsort(O)
        k_nearest_ind = dist_sort[1 : K + 1]

        densO = 1 / (1 / K * np.sum(O[k_nearest_ind]))

        densOn = []

        for n in k_nearest_ind:
            On = df.loc[n, :].values
            dist_sort_n = np.argsort(On)
            k_nearest_ind_n = dist_sort_n[1 : K + 1]
            densOn.append(1 / (1 / K * np.sum(On[k_nearest_ind_n])))

        ARD = densO / (1 / K * np.sum(densOn))

        print("The density for observation O{} is {}".format(obs + 1, densO))
        print(
            "The average relative density for observation O{} is {}".format(
                obs + 1, ARD
            )
        )

        return ARD


class association_mining:
    def mat2transactions(self, X, labels=[]):
        """
        Copyright Tue Herlev
        Technical University of Denmark
        """
        T = []
        for i in range(X.shape[0]):
            l = np.nonzero(X.loc[i, :])[0].tolist()
            if labels:
                l = [labels[i] for i in l]
            T.append(l)
        return T

    def print_apriori_rules(self, rules, prints=True, sup_dig=4, conf_dig=4):
        """
        Copyright Tue Herlev
        Technical University of Denmark
        """
        frules = []
        rules_conf_sup = pd.DataFrame()
        i = 0
        for r in rules:
            for o in r.ordered_statistics:
                conf = round(o.confidence, conf_dig)
                supp = round(r.support, sup_dig)
                x = ", ".join(list(o.items_base))
                y = ", ".join(list(o.items_add))
                if prints:
                    print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)" % (x, y, supp, conf))
                frules.append((x, y))
                rules_conf_sup[i] = np.array([x, y, supp, conf])
                i += 1
        rules_conf_sup = np.transpose(rules_conf_sup)
        rules_conf_sup.columns = ["X itemsets", "Y itemsets", "support", "confidence"]

        return frules, rules_conf_sup

    def rule_stats(self, df, X, Y, index=0):
        """
        calculates support and confidence of a rule
        ----------------------
        parameters:
        -----------
        df = binary dataframe with rows as transactions and colums representing the attributes
        X = list with the X itemsets
        Y = list with the Y itemsets
        """
        X = np.array(X) - index
        Y = np.array(Y) - index

        X_item = df.loc[:, X]
        Y_item = df.loc[:, Y]
        items = pd.concat([X_item, Y_item], axis=1)

        support = np.mean(np.mean(items, axis=1) == 1)
        confidence = np.sum(np.mean(items, axis=1) == 1) / np.sum(
            np.mean(X_item, axis=1) == 1
        )

        the_rule = "{{{}}} ---> {{{}}}".format(X, Y)

        print("The rule is {}".format(the_rule))
        print("The support for the rule is {}".format(support))
        print("The confidence for the rule is {}".format(confidence))

    def apriori_rules(self, df, X, Y, sup, conf):
        """
        calculates rules
        ----------------------
        parameters:
        -----------
        df = binary dataframe with rows as transactions and colums representing the attributes
        X = list with the X itemsets
        Y = list with the Y itemsets
        """
        Y = ["att {}".format(i) for i in Y]
        X = ["att {}".format(i) for i in X]

        names = ["att {}".format(i) for i in range(0, df.shape[1])]
        df.columns = names

        T = self.mat2transactions(df, labels=names)

        rules = apriori(T, min_support=sup, min_confidence=conf)
        frules, rules_conf_sup = self.print_apriori_rules(rules, prints=False)

        # sort by the x itemsets
        rules_conf_sup = rules_conf_sup.sort_values(by="support")
        # reset index
        rules_conf_sup = rules_conf_sup.reset_index(drop=True)

        return frules, rules_conf_sup


# tests
