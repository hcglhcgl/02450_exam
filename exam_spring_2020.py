from numpy.core.getlimits import _fr0
import toolbox_extended as te
import toolbox_02450 as tb
import numpy as np
import pandas as pd
from exam_toolbox import *
import re
import os
from math import exp
from scipy.integrate import quad


class exam:

    # ----------------------------------------------- OPG 1-----------------------------------------------
    def opg1():
        # One is nominal, since the Region is encoded
        return "B"

    # ----------------------------------------------- OPG 2-----------------------------------------------
    def opg2():
        """
        BirthRT must be Histogram 1
        DeathRT Must be Histogram 2
        InfMort must be Histogram 4
        LEXpM must be HistoGram 3
        So the answer is D
        """
        return "D"

    # ----------------------------------------------- OPG 3-----------------------------------------------
    def opg3():
        pca = pca_calc()
        S = [19.64,6.87,3.26,2.30,1.12]
        pca.draw_curve_from_diagonal_values(S)
        pca.var_explained(S)
        # From the output we see A must be correct
        return "A"

    # ----------------------------------------------- OPG 4-----------------------------------------------
    def opg4():
        # We insert the observations in PCA_proj.mw
        # For A, we see that inserting the values gives a large positive projection
        # On the plot, we see that african countries indeed do have a large positive projection to PCA1
        return "A"

    # ----------------------------------------------- OPG 5-----------------------------------------------
    def opg5():
        GMM_Plot = gmm()
        # We plot different options for first cluster:
        #A 
        mu = [[3.39],[0.74]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[0.1695, 0.0665], [0.0665, 0.1104]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        # B,C & D
        mu = [[-1.9482],[0.6132]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[0.1695, 0.0665], [0.0665, 0.1104]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        # It must be B, C or D
        # We can rule out B because it has such a low weight for this cluster (0.1425)
        # C
        mu = [[3.39],[0.74]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[2.07, 0.1876], [0.1876, 0.1037]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        # D
        mu = [[3.39],[0.74]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[1.2137, -0.0703], [-0.0703, 0.3773]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        # Looks like D, we confirm by doing the last GMM
        mu = [[0.2756],[-0.57]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[2.07, 0.1876], [0.1876, 0.1037]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        # Confirmed, D is the answer
        return "D"

    # ----------------------------------------------- OPG 6-----------------------------------------------
    def opg6():
        data = [[0.0,1.7,1.4,0.4,2.2,3.7,5.2,0.2,4.3,6.8,6.0],
                [1.7,0.0,1.0,2.0,1.3,2.6,4.5,1.8,3.2,5.9,5.2],
                [1.4,1.0,0.0,1.7,0.9,2.4,4.1,1.5,3.0,5.5,4.8],
                [0.4,2.0,1.7,0.0,2.6,4.0,5.5,0.3,4.6,7.1,6.3],
                [2.2,1.3,0.9,2.6,0.0,1.7,3.4,2.4,2.1,4.8,4.1],
                [3.7,2.6,2.4,4.0,1.7,0.0,2.0,3.8,1.6,3.3,2.7],
                [5.2,4.5,4.1,5.5,3.4,2.0,0.0,5.4,2.5,1.6,0.9],
                [0.2,1.8,1.5,0.3,2.4,3.8,5.4,0.0,4.4,6.9,6.1],
                [4.3,3.2,3.0,4.6,2.1,1.6,2.5,4.4,0.0,3.4,2.9],
                [6.8,5.9,5.5,7.1,4.8,3.3,1.6,6.9,3.4,0.0,1.0],
                [6.0,5.2,4.8,6.3,4.1,2.7,0.9,6.1,2.9,1.0,0.0]]
        df = pd.DataFrame(data)
        anom = anomaly()
        anom.ARD(df,2,2)
        #From the function we see D is correct
        return "D"

    # ----------------------------------------------- OPG 7-----------------------------------------------
    def opg7():
        data = [[0.0,1.7,1.4,0.4,2.2,3.7,5.2,0.2,4.3,6.8,6.0],
                [1.7,0.0,1.0,2.0,1.3,2.6,4.5,1.8,3.2,5.9,5.2],
                [1.4,1.0,0.0,1.7,0.9,2.4,4.1,1.5,3.0,5.5,4.8],
                [0.4,2.0,1.7,0.0,2.6,4.0,5.5,0.3,4.6,7.1,6.3],
                [2.2,1.3,0.9,2.6,0.0,1.7,3.4,2.4,2.1,4.8,4.1],
                [3.7,2.6,2.4,4.0,1.7,0.0,2.0,3.8,1.6,3.3,2.7],
                [5.2,4.5,4.1,5.5,3.4,2.0,0.0,5.4,2.5,1.6,0.9],
                [0.2,1.8,1.5,0.3,2.4,3.8,5.4,0.0,4.4,6.9,6.1],
                [4.3,3.2,3.0,4.6,2.1,1.6,2.5,4.4,0.0,3.4,2.9],
                [6.8,5.9,5.5,7.1,4.8,3.3,1.6,6.9,3.4,0.0,1.0],
                [6.0,5.2,4.8,6.3,4.1,2.7,0.9,6.1,2.9,1.0,0.0]]
        df = pd.DataFrame(data)
        clu = cluster()
        clu.dendro_plot(df,"complete")
        # We see Dendrogram 1 is correct = Answer A
        return "A"

    # ----------------------------------------------- OPG 8-----------------------------------------------
    def opg8():
        data = [[0.0,1.7,1.4,0.4,2.2,3.7,5.2,0.2,4.3,6.8,6.0],
                [1.7,0.0,1.0,2.0,1.3,2.6,4.5,1.8,3.2,5.9,5.2],
                [1.4,1.0,0.0,1.7,0.9,2.4,4.1,1.5,3.0,5.5,4.8],
                [0.4,2.0,1.7,0.0,2.6,4.0,5.5,0.3,4.6,7.1,6.3],
                [2.2,1.3,0.9,2.6,0.0,1.7,3.4,2.4,2.1,4.8,4.1],
                [3.7,2.6,2.4,4.0,1.7,0.0,2.0,3.8,1.6,3.3,2.7],
                [5.2,4.5,4.1,5.5,3.4,2.0,0.0,5.4,2.5,1.6,0.9],
                [0.2,1.8,1.5,0.3,2.4,3.8,5.4,0.0,4.4,6.9,6.1],
                [4.3,3.2,3.0,4.6,2.1,1.6,2.5,4.4,0.0,3.4,2.9],
                [6.8,5.9,5.5,7.1,4.8,3.3,1.6,6.9,3.4,0.0,1.0],
                [6.0,5.2,4.8,6.3,4.1,2.7,0.9,6.1,2.9,1.0,0.0]]
        df = pd.DataFrame(data)
        sup = supervised()
        red_classes = [1,2,3,4,5,6,7,8]
        black_classes = [9,10,11]
        pred = sup.knn_dist_pred_2d(df,red_classes,black_classes,1)
        sup.pred_stats(pred["True_label"],pred["Predicted_label"],show=True)
        # We can see this is equal to 4/11 = Answer B
        return "B"

    # ----------------------------------------------- OPG 9-----------------------------------------------
    def opg9():
        obsA=(np.array([1,-0.06,-0.28,0.43,-0.30,-0.36,0,0,0,0,1]))
        weights = np.array([1.41,0.76,1.76,-0.32,-0.96,6.64,-5.13,-2.06,96.73,1.03,-2.74])
        y_a = obsA @ weights
        pA1 = 1 / (1+(exp(-y_a)))
        print (pA1)
        # We can see the likelihood of being in the positive class = rich is 1.66%
        # If we increase the birth-rate of the observation we see an increased probability to be rich
        # This means A is correct.
        return "A"

    # ----------------------------------------------- OPG 10-----------------------------------------------
    def opg10():
        ens = ensemble()
        ens.conf_matrix_stats(34,11,7,39,"all")
        # We see the F-measure is 0.7907
        # So answer B is correct
        return "B"

    # ----------------------------------------------- OPG 11-----------------------------------------------
    def opg11():
        ens = ensemble()
        
        probabilities = [0,0.05,0.05,0.07,0.15,0.2,0.2,0.6,0.7,0.85,0.9]
        
        pred_A = [0,0,0,0,0,0,0,0,1,1,1]
        
        pred_B = [0,0,0,0,0,0,0,1,1,0,1]
        
        pred_C = [0,0,0,0,0,0,1,0,1,0,1]
        
        pred_D = [0,0,0,0,0,0,0,1,0,1,1]
        
        ens.plot_roc_pred(pred_A,probabilities)
        ens.plot_roc_pred(pred_B,probabilities)
        ens.plot_roc_pred(pred_C,probabilities)
        ens.plot_roc_pred(pred_D,probabilities)
        # We see it must be option D
        return "D"

    # ----------------------------------------------- OPG 12-----------------------------------------------
    def opg12():
        # x_3 is chosen in the first round, no combination of x3 with anything gives a smaller error.
        # There forward selection will select attribute x3
        return "A"

    # ----------------------------------------------- OPG 13-----------------------------------------------
    def opg13():
        time_train = 20
        time_test = 1
        
        K1 = 4 # outer fold
        K2 = 7 # inner fold
        S = 3 # number of models tested
        
        training_model_time = K1*(K2*S+1)*time_train
        testing_model_time = K1*(K2*S+1)*time_test
        
        print (training_model_time+testing_model_time)
        # We see the answer is D
        return "E"

    # ----------------------------------------------- OPG 14-----------------------------------------------
    def opg14():
        # Solve it by drawing squares in Paint
        return "C"

    # ----------------------------------------------- OPG 15-----------------------------------------------
    def opg15():
        P_africa = 0.154
        P_GNP_high_africa = 0.286
        P_GNP_high_not_africa = 0.688
        
        P_africa_GNP_high = (P_GNP_high_africa*P_africa)/(P_GNP_high_africa*P_africa+P_GNP_high_not_africa*(1-P_africa))
        print (P_africa_GNP_high)
        # So we see it must be C
        return "C"

    # ----------------------------------------------- OPG 16-----------------------------------------------
    def opg16():
        """
        It is not naive bayes, because it is given a class, what is the probability of the features
        We add an alpha = 1 in the normal bayes equation
        We see zero matches, for f2 = 1, f3 = 1, y=1 (black class) 
        We see 3 matches for y = 1
        """
        alpha = 1
        p = (0+alpha)/(3+2*alpha)
        print (p)
        # So the answer is B
        return "B"

    # ----------------------------------------------- OPG 17-----------------------------------------------
    def opg17():
        data = [[1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [0,1,1,0,0],
                [0,1,0,1,1],
                [1,1,1,0,0],
                [1,0,1,0,0],
                [0,0,0,1,1],
                [0,1,0,1,1]]
        df = pd.DataFrame(data)
        sup = supervised()
        labels = [0,0,0,0,0,0,0,0,1,1,1]
        
        sup.naive_bayes(labels,df,[1,2],[1,0],1)
        # We see the answer is 0.4, must be D
        return "D"

    # ----------------------------------------------- OPG 18-----------------------------------------------
    def opg18():
        dec = decision_trees()
        # We see the median of f1 is 1
        # This will put 6 red & 1 black in one branch
        # and 2 red and 2 black in the other
        dec.purity_gain([8,3],[6,1],[2,2],"gini")
        # We see that the purity gain is 0.059, answer B
        return "E"

    # ----------------------------------------------- OPG 19-----------------------------------------------
    def opg19():
        sim = similarity()
        truth = [0,0,0,0,0,0,0,0,1,1,1]
        f_2_clustering = [1,1,1,1,1,1,1,1,0,0,1]
        sim.measures(truth,f_2_clustering)
        return "E"

    # ----------------------------------------------- OPG 20-----------------------------------------------
    def opg20():
        data = [[1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [0,1,1,0,0],
                [0,1,0,1,1],
                [1,1,1,0,0],
                [1,0,1,0,0],
                [0,0,0,1,1],
                [0,1,0,1,1]]
        df = pd.DataFrame(data)
        it = itemset()
        it.itemsets(df,0.3)
        # We see the answer must be C
        return "C"

    # ----------------------------------------------- OPG 21-----------------------------------------------
    def opg21():
        data = [[1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [1,1,1,0,0],
                [0,1,1,0,0],
                [0,1,0,1,1],
                [1,1,1,0,0],
                [1,0,1,0,0],
                [0,0,0,1,1],
                [0,1,0,1,1]]
        df = pd.DataFrame(data)
        it = itemset()
        it.confidence(df,[0,1],[2])
        # We see the confidence is 1, answer D
        return "D"

    # ----------------------------------------------- OPG 22-----------------------------------------------
    def opg22():
        # A must be correct, because it's furthest point is the closest to the green dot,
        # compared to the other clusters.
        return "A"

    # ----------------------------------------------- OPG 23-----------------------------------------------
    def opg23():
        """
        Read the description in the solved set
        """
        return "A"

    # ----------------------------------------------- OPG 24-----------------------------------------------
    def opg24():
        """
        We integrate the function by cutting it up into blocks:
        Done in Maple
        """
        return "B"

    # ----------------------------------------------- OPG 25-----------------------------------------------
    def opg25():
        """
        Solved in Maple Sheet, IntegralsPnorm
        """
        return "B"

    # ----------------------------------------------- OPG 26-----------------------------------------------
    def opg26():
        """
        The number of observations used for testing is the same 
        when we use K-fold cross-validation for all values of K
        as all observations are used once for testing.
        """
        return "D"

    # ----------------------------------------------- OPG 27-----------------------------------------------
    def opg27():
        ada = adaboost()
        alphas = ada.get_alpha_given_errors([0.417,0.243,0.307,0.534])
        # For point 1, we see it is classified as 0 in round 1,3 & 4
        F0_1 = alphas[0] + alphas[2] + alphas[3]
        # and as 1 in round 2
        F1_1 = alphas[1]
        print("0: ", F0_1, "1: ", F1_1)
        
        # We do that same for point 2, 0 in round 1, 1 in round 2,3,4
        F0_2 = alphas[0]
        F1_2 = alphas[1] + alphas[2] + alphas[3]
        print("0: ", F0_2, "1: ", F1_2)
        
        # We see F is higher for class 1, in the first test
        # and F is also higher for class 1 in second test
        # Both will therefore be classified as 1.
        return "D"

    # -------------------------------- answers dataframe -------------------------------------------------
    def answers(show=True, csv=False, excel=False):
        ans = pd.DataFrame(
            columns=["Student number: s174852"]
        )  # columns = ["OPG", "svar"])

        # ans.loc[0] = ""
        # ans.loc[1] = "Q01: {}".format(exam.opg1())
        # ans.loc[2] = "Q02: {}".format(exam.opg2())
        # ans.loc[3] = "Q03: {}".format(exam.opg3())
        # ans.loc[4] = "Q04: {}".format(exam.opg4())
        # ans.loc[5] = "Q05: {}".format(exam.opg5())
        # ans.loc[6] = "Q06: {}".format(exam.opg6())
        # ans.loc[7] = "Q07: {}".format(exam.opg7())
        # ans.loc[8] = "Q08: {}".format(exam.opg8())
        # ans.loc[9] = "Q09: {}".format(exam.opg9())
        # ans.loc[10] = "Q10: {}".format(exam.opg10())
        # ans.loc[11] = ""

        # ans.loc[12] = "Q11: {}".format(exam.opg11())
        # ans.loc[13] = "Q12: {}".format(exam.opg12())
        # ans.loc[14] = "Q13: {}".format(exam.opg13())
        # ans.loc[15] = "Q14: {}".format(exam.opg14())
        # ans.loc[16] = "Q15: {}".format(exam.opg15())
        # ans.loc[17] = "Q16: {}".format(exam.opg16())
        # ans.loc[18] = "Q17: {}".format(exam.opg17())
        # ans.loc[19] = "Q18: {}".format(exam.opg18())
        # ans.loc[20] = "Q19: {}".format(exam.opg19())
        # ans.loc[21] = "Q20: {}".format(exam.opg20())
        # ans.loc[22] = ""

        # ans.loc[23] = "Q21: {}".format(exam.opg21())
        # ans.loc[24] = "Q22: {}".format(exam.opg22())
        # ans.loc[25] = "Q23: {}".format(exam.opg23())
        # ans.loc[26] = "Q24: {}".format(exam.opg24())
        # ans.loc[27] = "Q25: {}".format(exam.opg25())
        # ans.loc[28] = "Q26: {}".format(exam.opg26())
        # ans.loc[29] = "Q27: {}".format(exam.opg27())

        if excel:
            ans.to_excel(re.sub(".py", "_answers.xlsx", __file__), index=False)
        if csv:
            ans.to_csv(re.sub(".py", "_answers.csv", __file__), index=False)
        if show:
            print(ans)

        return ans


exam.answers()
