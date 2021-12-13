"""
Functions created to use for the exam in the course 02450 Introduction to machine learning and datamining at the Technical University of Denmark

Author: Lukas Leindals / Hans Christian Lundberg
"""
__version__ = "Revision: 2021-12-11"

from scipy.spatial import distance
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
import itertools as IT
import scipy.stats as st



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
    def draw_curve_from_diagonal_values(self,values):
        """

        :param values: singular_values e.g [17.4, 7.3, 4.3]
        :return:
        """
        squares = np.square(values)
        sum = np.sum(squares)
        rho = squares / (np.ones(len(squares)) * sum)
        threshold = 0.9

        # Plot variance explained
        plt.figure()
        plt.plot(range(1, len(rho) + 1), rho, 'x-')
        plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
        plt.plot([1, len(rho)], [threshold, threshold], 'k--')
        plt.title('Variance explained by principal components')
        plt.xlabel('Principal component')
        plt.ylabel('Variance explained')
        plt.legend(['Individual', 'Cumulative', 'Threshold'])
        plt.grid()
        plt.show()


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
    def plot_roc_pred(self,truth,probabilities):
        """Generates a ROC curve from true labels and predicted class probabilities

        Args:
            truth (list): List with true class labels
            probabilities (list): List with predicted class probabilities
        """
        plt.figure(1)
        tb.rocplot(probabilities, truth)

        plt.show()  


class supervised:
    def knn_dist_pred_2d(self, df, class1, class2, K, show=False):
        """
        calculates predictions given a matrix with euclidean distances, can only handle two classes: red and black
        -------------------------------------------------------
        class1 = list with numbers of observations in the red class (starts at 1)
        class2 = list with numbers of observations in the black class (starts at 1)
        """
        classes = {"red": class1, "black": class2}

        # Get indexes of of red/black observations
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

    def knn_dist_pred_3d(self, df, class1, class2, class3, K, show=False):
        """
        calculates predictions given a matrix with euclidean distances, can handle tree classes: red, black, blue
        -------------------------------------------------------
        class1 = list with numbers of observations in the red class (starts at 1)
        class2 = list with numbers of observations in the black class (starts at 1)
        class3 = list with numbers of observations in the blue class (starts at 1)
        """
        classes = {"red": class1, "black": class2,"blue": class3}

        # Get indexes of of red/black observations
        red_ind = [i - 1 for i in classes["red"]]
        black_ind = [i - 1 for i in classes["black"]]
        blue_ind = [i - 1 for i in classes["blue"]]

        pred_label = []
        O = [i for i in range(1, df.shape[1] + 1)]

        for row in range(df.shape[0]):
            dist = df.loc[row, :].values
            # sort
            dist_sort = np.argsort(dist)
            k_nearest_ind = dist_sort[1 : K + 1]

            pred_red = 0
            pred_black = 0
            pred_blue = 0

            for i in range(K):
                if k_nearest_ind[i] in red_ind:
                    pred_red += 1
                elif k_nearest_ind[i] in black_ind:
                    pred_black += 1
                elif k_nearest_ind[i] in blue_ind:
                    pred_blue += 1
            if pred_red > pred_black and pred_red > pred_blue:
                pred_label.append("red")
            elif pred_black > pred_red and pred_black > pred_blue:
                pred_label.append("black")
            elif pred_blue > pred_red and pred_blue > pred_black:
                pred_label.append("blue")
            elif pred_black == pred_red == pred_blue:
                if k_nearest_ind[0] in red_ind:
                    pred_label.append("red")
                elif k_nearest_ind[0] in black_ind:
                    pred_label.append("black")
                else:
                    pred_label.append("blue")
        true_label = []
        for obs in O:
            if obs - 1 in red_ind:
                true_label.append("red")
            elif obs - 1 in black_ind:
                true_label.append("black")
            elif obs - 1 in blue_ind:
                true_label.append("blue")

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
        !!!!!!!doesnt really work !!!!!!!! ?!?
        calculates predictions given a matrix with euclidean distances, can handle multiple classes
        -------------------------------------------------------
        class1 = list with coloumn numbers of observations in the red class (starts at 1)
        class2 = list with coloumn numbers of observations in the black class (starts at 1)
        class3 = list with coloumn numbers of observations in the blue class (starts at 1)
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
        pred_class = the class you would like to predict the probability of (starts at 0) <- remember this if y starts on 1
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
        dist_df = symmetrical matrix/dataframe containing distances
        Method = linkage method:
            "single"
            "complete" aka Maximum
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

        R = cluster.fancy_dendrogram(self,
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
        x : Cluster A (labels) = The truth: Example: [1,2,1,1,1,2,2,2,2,1]

        y : Cluster B eg. [1,2,1,1,1,3,1,1,1,1]
        Example from:
        Cutoff at the level of 3 clusters = 3 vertical lines
        Here we see O2 has been seperated, and O6 as well. 
        Since O2 was seperated in the first cluster, we give it 2
        And O6 we give 3
        The rest have not been seperated yet, but majority black so we give it 1
        
        Printer similarity Index - Rand og Jaccard
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
    def distance_between_clusters(self,dist_matrix,cluster1,cluster2):
        """Returns distance between clusters (avg linkage function)

        Args:
            dist_matrix (matrix): Distance matrix between observations
            cluster1: Indexes of cluster 1 observations
            cluster2: Indexes of cluster 2 observations
        """
        distances = []
        for i in cluster1:
            for j in cluster2:
                distances.append(dist_matrix[i][j])
        distances = np.array(distances)
        sum = np.sum(distances)
        elements = len(cluster1) * len(cluster2) 
        print (sum/elements)
        
        

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
    
    def correlation_from_covariance(self,cov_matrix):
        """
        cov_matrix : 2d array of covariance matrix, eg: [[0.2639, 0.0803], [0.0803, 0.0615]]
        Calculates the correlation between the two x's in the covariance matrix
        The correlation coefficient is defined as: p=cov(x,y)/(sigma_x*sigma_y)
        """
        cov = np.array(cov_matrix)
        p = cov[1][0]/(cov[0][0]*cov[1][1])
        print (p)
        return p
        
        
    def similarity(self, X, Y, method):
        '''
        Does it really work ??
        SIMILARITY Computes similarity matrices

        Usage:
            sim = similarity(X, Y, method)

        Input:
        X   N1 x M matrix
        Y   N2 x M matrix 
        method   string defining one of the following similarity measure
            'SMC', 'smc'             : Simple Matching Coefficient
            'Jaccard', 'jac'         : Jaccard coefficient 
            'ExtendedJaccard', 'ext' : The Extended Jaccard coefficient
            'Cosine', 'cos'          : Cosine Similarity
            'Correlation', 'cor'     : Correlation coefficient

        Output:
        sim Estimated similarity matrix between X and Y
            If input is not binary, SMC and Jaccard will make each
            attribute binary according to x>median(x)

        Copyright, Morten Morup and Mikkel N. Schmidt
        Technical University of Denmark '''

        X = np.mat(X)
        Y = np.mat(Y)
        N1, M = np.shape(X)
        N2, M = np.shape(Y)
        
        method = method[:3].lower()
        if method=='smc': # SMC
            #X,Y = binarize(X,Y);
            sim = ((X*Y.T)+((1-X)*(1-Y).T))/M
        elif method=='jac': # Jaccard
            #X,Y = binarize(X,Y);
            sim = (X*Y.T)/(M-(1-X)*(1-Y).T)        
        elif method=='ext': # Extended Jaccard
            XYt = X*Y.T
            sim = XYt / (np.log( np.exp(sum(np.power(X.T,2))).T * np.exp(sum(np.power(Y.T,2))) ) - XYt)
        elif method=='cos': # Cosine
            sim = (X*Y.T)/(np.sqrt(sum(np.power(X.T,2))).T * np.sqrt(sum(np.power(Y.T,2))))
        elif method=='cor': # Correlation
            X_ = st.zscore(X,axis=1,ddof=1)
            Y_ = st.zscore(Y,axis=1,ddof=1)
            sim = (X_*Y_.T)/(M-1)
        return sim


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
class adaboost:
    
    def adaboost(self, delta, rounds):
        """
        delta : list of misclassified observations, 
        0 = correctly classified, 1 = misclassified

        rounds [int] : how many rounds to run
        
        Example: 
        Given a classification problem with 25 observations in total, 
        with 5 of them being misclassified in round 1, the weights can be calculated as:
        miss = np.zeros(25) 
        miss[:5] = 1 
        te.adaboost(miss, 1)

        The weights are printed 
        """
        # Initial weights
        delta = np.array(delta)
        n = len(delta)
        weights = np.ones(n) / n

        # Run all rounds
        for i in range(rounds):
            eps = np.mean(delta == 1)
            alpha = 0.5 * np.log((1 - eps) / eps)
            s = np.array([-1 if d == 0 else 1 for d in delta])

            # Calculate weight vector and normalize it
            weights = weights.T * np.exp(s * alpha)
            weights /= np.sum(weights)

            # Print resulting weights
        for i, w in enumerate(weights):
            print('w[%i]: %f' % (i, w))
            
    def get_alpha_given_w(self, M, last_wrong):
        """
        :param M: Numpy array of weights in dim. Number of Obs x Number of Rounds
        :param last_wrong: Binary vector of wrong classified observations in the last Boosting round.
        :return alpha list
        """
        alpha = []
        for i in range(M.shape[1] - 1):
            er = np.dot(M[:, i], (M[:, i + 1] > M[:, i]))
            alpha.append(1 / 2 * np.log((1 - er) / er))
        er = np.dot(M[:, M.shape[1] - 1], last_wrong)
        alpha.append(1 / 2 * np.log((1 - er) / er))
        print(alpha)
        
        return alpha
class ann:
    
    def logistic(self,x):
        return 1 / (1 + np.exp(-x))


    def rect(self,x):
        return np.max([x, 0])


    def tanh(self,x):
        return np.sinh(x) / np.cosh(x)


    def get_ann(self,w02, weights, matrices, activation='logistic'):
        """
        w02 : the w02 given in the xercise
        weights : list of the weights which have superscript (2)
        matrices : the matrices, usually w_n^(1)
        ann = get_ann(2.84, [3.25, 3.46], [[21.78, -1.65, 0, -13.26, -8.46], [-9.6, -0.44, 0.01, 14.54, 9.5]], "rect")
        y = ann([1, 6.8, 225, 0.44, 0.68])
        
        Example in Spring 2018, Exercise 8:
        an_model = ann_obj.get_ann(0.3799E-6, [-0.3440E-6, 0.0429E-6], [[0.0189, 0.9159, -0.4256], [3.7336, -0.8003, 5.0741]], "logistic")
        
        #  !! Always put "1" as first for some reason !!
        y_1 = an_model([1,0, 3])
        y_2 = an_model([1,24, 0])
        REMEMBER TO PUT "1" AT THE START OF ann([..])
        
        activation: Sigmoid = Logistic Activation
                    Tanh = hyperbolic tangent
                    Rect = Linear activation
        """
        matrices = np.array([np.matrix(m).T for m in matrices])
        weights = np.array(weights)

        activation_func = {
            "logistic": ann.logistic,
            "rect": ann.rect,
            "tanh": ann.tanh
        }[activation]

        def predict_y(x):
            x = np.matrix(x)
            activated_matrices = np.array([activation_func(self,x * m) for m in matrices])
            ann_sum = 0
            for (i, _) in enumerate(activated_matrices):
                ann_sum = ann_sum + (activated_matrices[i] * weights[i])
            return w02 + ann_sum

        return predict_y
    
class gmm:
    def plot_gmm(self,m,cov):
        """Function for plotting GMM contours
        Changing the coordinate system size is done inside the function!!
        Args:
            m ([2d array]): Mu/mean/center. Example: [[1.84],[2.43]]
            cov ([2d array]): Covariance matrix. Example: [[0.2639, 0.0803], [0.0803, 0.0615]]
        """
        m = np.array(m)
        cov = np.array(cov)
        N = 1000

        cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
        cov_det = np.linalg.det(cov)  # determinant of covariance matrix
        # Plotting
        x = np.linspace(-18, 0, N) # Size of coordinate system
        y = np.linspace(-6, 14, N)
        X,Y = np.meshgrid(x,y)
        coe = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
        Z = coe * np.e ** (-0.5 * (cov_inv[0,0]*(X-m[0])**2 + (cov_inv[0,1] + cov_inv[1,0])*(X-m[0])*(Y-m[1]) + cov_inv[1,1]*(Y-m[1])**2))
        plt.contour(X,Y,Z)
        plt.grid()
        plt.show()
        
    def prob_gmm(self, x, weights, means, standard_dev, target_class="all"):
        """Return the mixture prob that x belongs to a class
        A normal distribution is assumed
        Input:
            x: the variable of interest in prob calculations
            weights: a list of weights for all classes
            means: a list of means of the normal distribution for all classes
            standard_dev: a list of standard deviaitons (sigma) (NOT VARIANCES) for all classes
            target_class: desired class to calc the prob for. ZERO INDEX (y=0, or 1, 2 etc)
                        By default, keyword "all" calculates for every class (recommended)
        Example:
        x = 3.19
        weights = [0.19, 0.34, 0.48]
        means = [3.177, 3.181, 3.184]
        standard_dev = [0.0062, 0.0076, 0.0075]
        gm = gmm()
        gm.prob_gmm(x,weights,means,standard_dev,"all") #zero index for target class, we want 2 so y=1

        """
        
        #Error checking for bad user input:
        assert len(weights)==len(means)==len(standard_dev),\
        "The weights, means and std lists must have the same number of items. Check your lists."
        if type(target_class) == int:
            assert target_class < len(means), "Target Class is out of range. Make sure you are starting from 0"
        
        y = target_class
        res = []
        for (w,u,s) in zip(weights, means, standard_dev):
            p_i = w*st.norm.pdf(x=x,loc=u,scale=s)
            res.append(p_i)
        if y == "all":
            for i in range(len(res)):
                print(f"The prob that x={x} belongs to class {i} (0 index) is {res[i]/sum(res)}")
        else:
            print(f"The prob that x={x} belongs to class {y} (0 index) is {res[y]/sum(res)}")
            
class itemset:
    def itemsets(self,df, support_min):
        """
        df: dataframe with each row being a basket, and each column being an item
        support_min: minimum support level
        Remember that the printed itemsets start from 0!
        """
        itemsets = []
        n = len(df)
        for itsetSize in np.arange(1, len(df.columns) + 1): # Start with 1-itemsets, keep going till n_attributes-itemsets
            for combination in IT.combinations(df.columns, itsetSize):
                sup = itemset.support(self,df[list(combination)])
                if sup > support_min:
                    itemsets.append(set(combination))
        print(itemsets)
        return itemsets
    
    def support(self,itemset):
        """
        Returns the support value for the given itemset
        itemset is a pandas dataframe with one row per basket, and one column per item
        """

        # Get the count of baskets where all the items are 1
        baskets = itemset.iloc[:,0].copy()
        for col in itemset.columns[1:]:
            baskets = baskets & itemset[col]

        return baskets.sum() / float(len(baskets))
    
    def confidence(self,df, antecedentCols, consequentCols):
        """
        df is a pandas dataframe
        antecedentCols are the labels for the columns/items that make up the antecedent in the association rule
        For both: Remember index starts at 0!
        consequentCols are the labels for the columns/items that make up the consequent in the association rule
        """
        top = itemset.support(self,df[antecedentCols + consequentCols])
        bottom = itemset.support(self,df[antecedentCols])
        conf = top/bottom
        print ("The confidence is: ", conf)
        return conf
class model_test:
    def jeffery_interval(self,obs,corr_obs):
        """
        Prints the jeffery interval for a model. 
        obs : observations
        corr_obs : correctly classified observations
        """
        n = obs #observations
        m = corr_obs #correct classified
        alpha = 0.05 #always = 0.05

        a = m + 0.5
        b = n-m + 0.5

        #Confidence Intervals
        CI_L = st.beta.ppf(alpha/2,a,b)
        CI_H = st.beta.ppf(1-alpha/2,a,b)
        theta = a/(a+b)

        #Jeffery Intervals Results
        print(f"a={a}")
        print(f"b={b}")
        print(f"CI_L={CI_L}")
        print(f"CI_H={CI_H}")
        print(f"Theta={theta}")

    def mcnemar_test(self,n1,n2):
        """
        Prints the p value from the McNemar test between 2 classification models 
        n1 : the total number of times that Model 1 is correct and Model 2 is incorrect. Remember to sum, if multiple folds
        n2 : the total number of times that Model 1 is incorrect and Model 2 is correct. Remember to sum, if multiple folds
        """
        N = n1+n2
        m = min(n1,n2)
        theta = 1/2 #always
        
        p_val = 2*st.binom.cdf(m,N,theta)
        
        print(f"p-val={p_val:.5f}")
