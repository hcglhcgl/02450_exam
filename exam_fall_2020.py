from scipy.spatial.distance import directed_hausdorff
import toolbox_extended as te
import toolbox_02450 as tb
import numpy as np
import pandas as pd
from exam_toolbox import *
import re
import os


class exam:

    # ----------------------------------------------- OPG 1-----------------------------------------------
    def opg1():
        """
        For Sex and Species only uniqueness matters, so they are nominal.
        """
        return "C"

    # ----------------------------------------------- OPG 2-----------------------------------------------
    def opg2():
        """
        Simply looking at the mean of each boxplot and the data, we see that it must be Bill length, flipper length,bill depth
        Answer is therefore B
        """
        return "B"

    # ----------------------------------------------- OPG 3-----------------------------------------------
    def opg3():
        """
        The mean has been subtracted from the median, for sex, the median must have been 1, since it is now -1.
        It's 1-2 = -1
        """
        return "E"

    # ----------------------------------------------- OPG 4-----------------------------------------------
    def opg4():
        """
        The correlation coefficient is defined as: p=cov(x,y)/(sigma_x*sigma_y)
        """
        print (9852/(math.sqrt(196)*math.sqrt(648025)))
        # This is equal to the correlation coefficient, so the answer A must be correct
        return "A"

    # ----------------------------------------------- OPG 5-----------------------------------------------
    def opg5():
        pca = pca_calc()
        pca.var_explained([30.19,16.08,11.07,5.98])
        pca.draw_curve_from_diagonal_values([30.19,16.08,11.07,5.98])
        # We can see that only C is correct
        return "C"

    # ----------------------------------------------- OPG 6-----------------------------------------------
    def opg6():
        """
        Negative value = seperates low/short flippers from high/long flippers
        Postive value = seperates high body mass from low body mass.
        This means D is correct
        """
        return "D"

    # ----------------------------------------------- OPG 7-----------------------------------------------
    def opg7():
        data = [[0.45,-0.60,-0.64,0.15],[-0.40,-0.80,0.43,-0.16],[0.58,-0.01,0.24,-0.78],[0.55,-0.08,0.59,0.58]]
        x = [-1,-1,-1,1]
        pca = pca_calc()
        pca.proj(data,x,components=[1,2,3,4],V_full=True)
        # This is only close to a X = chinstrap in the coordinate system PC1/PC4 
        # The answer is therefore A
        return "A"

    # ----------------------------------------------- OPG 8-----------------------------------------------
    def opg8():
        #Solved using the script: "rand_index_occurence.py"
        return "C"

    # ----------------------------------------------- OPG 9-----------------------------------------------
    def opg9():
        #KNN is instance based with no learned
        #parameters when K is ﬁxed.
        return "C"

    # ----------------------------------------------- OPG 10-----------------------------------------------
    def opg10():
        """
        Reducing the amount of training data is
        the only option that will typically increase the amount
        of over-ﬁtting. The other choices will typically decrease
        over-ﬁtting.
        """
        return "B"

    # ----------------------------------------------- OPG 11-----------------------------------------------
    def opg11():
        print ((0.16*0.20)/(0.85*0.36+0.20*0.16))
        return "C"

    # ----------------------------------------------- OPG 12-----------------------------------------------
    def opg12():
        data = [[0,725,800,150,1000,525,600,500,400,850],
                [725,0,75,575,275,1250,1325,226,325,125],
                [800,75,0,650,200,1325,1400,300,400,51],
                [150,575,650,0,850,675,750,350,250,700],
                [1000,275,200,850,0,1525,1600,500,600,150],
                [525,1250,1325,675,1525,0,75,1025,925,1375],
                [600,1325,1400,750,1600,75,0,1100,1000,1450],
                [500,226,300,350,500,1025,1100,0,100,350],
                [400,325,400,250,600,925,1000,100,0,450],
                [850,125,51,700,150,1375,1450,350,450,0]]
        df = pd.DataFrame(data)
        anom = anomaly()
        anom.ARD(df,1,2)
        return "D"

    # ----------------------------------------------- OPG 13-----------------------------------------------
    def opg13():
        data = [[0,725,800,150,1000,525,600,500,400,850],
                [725,0,75,575,275,1250,1325,226,325,125],
                [800,75,0,650,200,1325,1400,300,400,51],
                [150,575,650,0,850,675,750,350,250,700],
                [1000,275,200,850,0,1525,1600,500,600,150],
                [525,1250,1325,675,1525,0,75,1025,925,1375],
                [600,1325,1400,750,1600,75,0,1100,1000,1450],
                [500,226,300,350,500,1025,1100,0,100,350],
                [400,325,400,250,600,925,1000,100,0,450],
                [850,125,51,700,150,1375,1450,350,450,0]]
        cluster1 = [5,6]
        cluster2 = [7,8,9]
        clu = cluster()
        clu.distance_between_clusters(data,cluster1,cluster2)
        return "B"

    # ----------------------------------------------- OPG 14-----------------------------------------------
    def opg14():
        """
        C is impossible because if we want to cut O3 & O10 from O2 at approx. dist = 150 we would also cut O1 from O4
        
        """
        return "C"

    # ----------------------------------------------- OPG 15-----------------------------------------------
    def opg15():
        # Numerator
        p_f1_0_x_1 = 3/6
        p_f2_0_f3_1_x_1 = 3/6
        p_f4_0_x_1 = 4/6
        p_x_1 = 6/10
        numerator = p_f1_0_x_1 * p_f2_0_f3_1_x_1 * p_f4_0_x_1 * p_x_1
        # Denominator
        # K = 1
        p_f1_0_x_1 = 3/6
        p_f2_0_f3_1_x_1 = 3/6
        p_f4_0_x_1 = 4/6
        p_x_1 = 6/10
        denom_1 = p_f1_0_x_1 * p_f2_0_f3_1_x_1 * p_f4_0_x_1 * p_x_1
        # K = 2
        p_f1_0_x_2 = 2/4
        p_f2_0_f3_1_x_2 = 2/4
        p_f4_0_x_2 = 4/4
        p_x_2 = 4/10
        denom_2 = p_f1_0_x_2 * p_f2_0_f3_1_x_2 * p_f4_0_x_2 * p_x_2
        
        print (numerator/(denom_1+denom_2))
        
        # So the answer is 1/2
        return "C"

    # ----------------------------------------------- OPG 16-----------------------------------------------
    def opg16():
        data = [[0,0,1,0],
                [1,0,1,0],
                [0,1,0,0],
                [1,0,1,1],
                [1,0,0,0],
                [0,0,0,1],
                [1,0,1,0],
                [0,1,0,0],
                [0,0,1,0],
                [1,0,0,0]]
        df = pd.DataFrame(data)
        it = itemset()
        it.itemsets(df,0.25)
        # We see the answer is A
        return "A"

    # ----------------------------------------------- OPG 17-----------------------------------------------
    def opg17():
        data = [[0,0,1,0],
                [1,0,1,0],
                [0,1,0,0],
                [1,0,1,1],
                [1,0,0,0],
                [0,0,0,1],
                [1,0,1,0],
                [0,1,0,0],
                [0,0,1,0],
                [1,0,0,0]]
        df = pd.DataFrame(data)
        assc = association_mining()
        assc.rule_stats(df,[0,3],[2])
        # Test with other function
        it = itemset()
        it.confidence(df,[0,3],[2])
        return "D"

    # ----------------------------------------------- OPG 18-----------------------------------------------
    def opg18():
        dec = decision_trees()
        dec.purity_gain([146,119,68],[146,119,0],[0,0,68],"class_error")
        # This gives option B
        return "B"

    # ----------------------------------------------- OPG 19-----------------------------------------------
    def opg19():
        miss = np.zeros(7) 
        miss[:1] = 1 
        te.adaboost(miss, rounds=1)
        # The answer must be A
        return "A"

    # ----------------------------------------------- OPG 20-----------------------------------------------
    def opg20():
        # We see from Sauds scripts "Classification graphing_3classes_3rules", that it must be B
        return "B"

    # ----------------------------------------------- OPG 21-----------------------------------------------
    def opg21():
        # Solved using Mine_Scripts/output_probability.py, and testing for x = 14, 16 & and 18
        #This means the answer must be D
        return "D"

    # ----------------------------------------------- OPG 22-----------------------------------------------
    def opg22():
        # Dunno
        return "E"

    # ----------------------------------------------- OPG 23-----------------------------------------------
    def opg23():
        test = model_test()
        # We apply the mcnemar function. The total number of times (over all folds) where M1 is correct and M2 incorrect is 28
        # Sum of M2 incorrect and M1 correct is 35 times
        test.mcnemar_test(28,35)
        # We see the P-value is 0.45
        return "B"

    # ----------------------------------------------- OPG 24-----------------------------------------------
    def opg24():
        ens = ensemble()
        true = [1,0,0,1,1,0,1]
        pred = [0.01,0.05,0.14,0.3,0.31,0.36,0.91]
        ens.plot_roc_pred(true,pred)
        # We see the answer must be curve 3, answer C
        return "C"

    # ----------------------------------------------- OPG 25-----------------------------------------------
    def opg25():
        # Using the ANN_parameters.py script we see that there must be 51 parameters
        return "E"

    # ----------------------------------------------- OPG 26-----------------------------------------------
    def opg26():
        # Bayes problem:
        P_ID_penguin = 0.97
        P_ID_no_penguin = 0.03
        P_penguin = 0.01
        P_no_penguin = 0.99
        P_penguin_ID = (P_ID_penguin*P_penguin)/(P_ID_penguin*P_penguin+P_ID_no_penguin*P_no_penguin)
        print (P_penguin_ID)
        return "D"

    # ----------------------------------------------- OPG 27-----------------------------------------------
    def opg27():
        # Outer fold, split into 3, 2 for training 1 for testing
        # 222 in training, 111 testing.
        # The observation will be part of training 2 times
        # Inner fold: Leave-one-out, 221 times training.
        # 4 models trained for regularization constant
        # We also train one for optimal choice of lambda on the outer fold set
        obs_training = 2 * (221 * 4 + 1)
        print (obs_training)
        # The answer must be C
        return "C"

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
