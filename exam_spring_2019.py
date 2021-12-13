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
        # Looking at the distribution in the scatterplot, we can see that the only possible answer is D
        return "D"

    # ----------------------------------------------- OPG 2-----------------------------------------------
    def opg2():
        pca = pca_calc()
        S = [14.14,11.41,9.46,4.19,0.17]
        pca.draw_curve_from_diagonal_values(S)
        # We can from the output of the function that C must be the answer
        return "C"

    # ----------------------------------------------- OPG 3-----------------------------------------------
    def opg3():
        # Solved in Maple
        
        return "E"

    # ----------------------------------------------- OPG 4-----------------------------------------------
    def opg4():
        data = [[0.0,2.0,5.7,0.9,2.9,1.8,2.7,3.7,5.3,5.1],
                [2.0,0.0,5.6,2.4,2.5,3.0,3.5,4.3,6.0,6.2],
                [5.7,5.6,0.0,5.0,5.1,4.0,3.3,5.4,1.2,1.8],
                [0.9,2.4,5.0,0.0,2.7,2.1,2.2,3.5,4.6,4.4],
                [2.9,2.5,5.1,2.7,0.0,3.5,3.7,4.0,5.8,5.7],
                [1.8,3.0,4.0,2.1,3.5,0.0,1.7,5.3,3.8,3.7],
                [2.7,3.5,3.3,2.2,3.7,1.7,0.0,4.2,3.1,3.2],
                [3.7,4.3,5.4,3.5,4.0,5.3,4.2,0.0,5.5,6.0],
                [5.3,6.0,1.2,4.6,5.8,3.8,3.1,5.5,0.0,2.1],
                [5.1,6.2,1.8,4.4,5.7,3.7,3.2,6.0,2.1,0.0]]
        df = pd.DataFrame(data)
        anom = anomaly()
        anom.ARD(df,6,2)
        return "E"

    # ----------------------------------------------- OPG 5-----------------------------------------------
    def opg5():
        data = [[0.0,2.0,5.7,0.9,2.9,1.8,2.7,3.7,5.3,5.1],
                [2.0,0.0,5.6,2.4,2.5,3.0,3.5,4.3,6.0,6.2],
                [5.7,5.6,0.0,5.0,5.1,4.0,3.3,5.4,1.2,1.8],
                [0.9,2.4,5.0,0.0,2.7,2.1,2.2,3.5,4.6,4.4],
                [2.9,2.5,5.1,2.7,0.0,3.5,3.7,4.0,5.8,5.7],
                [1.8,3.0,4.0,2.1,3.5,0.0,1.7,5.3,3.8,3.7],
                [2.7,3.5,3.3,2.2,3.7,1.7,0.0,4.2,3.1,3.2],
                [3.7,4.3,5.4,3.5,4.0,5.3,4.2,0.0,5.5,6.0],
                [5.3,6.0,1.2,4.6,5.8,3.8,3.1,5.5,0.0,2.1],
                [5.1,6.2,1.8,4.4,5.7,3.7,3.2,6.0,2.1,0.0]]
        df = pd.DataFrame(data)
        
        super = supervised()

        prediction = super.knn_dist_pred_3d(df,[1,2],[3,4,5],[6,7,8,9,10],3)
        super.pred_stats(prediction["True_label"],prediction["Predicted_label"],show=True)
        # We can see the error rate is 0.6 = Answer C
        return "C"

    # ----------------------------------------------- OPG 6-----------------------------------------------
    def opg6():
        data = [[0.0,2.0,5.7,0.9,2.9,1.8,2.7,3.7,5.3,5.1],
                [2.0,0.0,5.6,2.4,2.5,3.0,3.5,4.3,6.0,6.2],
                [5.7,5.6,0.0,5.0,5.1,4.0,3.3,5.4,1.2,1.8],
                [0.9,2.4,5.0,0.0,2.7,2.1,2.2,3.5,4.6,4.4],
                [2.9,2.5,5.1,2.7,0.0,3.5,3.7,4.0,5.8,5.7],
                [1.8,3.0,4.0,2.1,3.5,0.0,1.7,5.3,3.8,3.7],
                [2.7,3.5,3.3,2.2,3.7,1.7,0.0,4.2,3.1,3.2],
                [3.7,4.3,5.4,3.5,4.0,5.3,4.2,0.0,5.5,6.0],
                [5.3,6.0,1.2,4.6,5.8,3.8,3.1,5.5,0.0,2.1],
                [5.1,6.2,1.8,4.4,5.7,3.7,3.2,6.0,2.1,0.0]]
        df = pd.DataFrame(data)
        dendro = cluster()
        dendro.dendro_plot(df,"complete")
        # We can see it must be Dendro 2, answer B
        return "B"

    # ----------------------------------------------- OPG 7-----------------------------------------------
    def opg7():
        clus = cluster()
        truth = [0,0,1,1,1,2,2,2,2,2]
        cutoff_pred = [0,0,2,0,0,0,0,2,2,3]
        clus.cluster_similarity(truth,cutoff_pred)
        # We see the answer is B
        return "B"

    # ----------------------------------------------- OPG 8-----------------------------------------------
    def opg8():
        dec = decision_trees()
        root = [263,359,358]
        split1_branch1 = [143,137,54]
        split1_branch2 = [263-143,359-137,358-54]
        dec.purity_gain(root,split1_branch1,split1_branch2,"class_error")
        split2_branch1 = [223,251,197]
        split2_branch2 = [263-223,359-251,358-197]
        dec.purity_gain(root,split2_branch1,split2_branch2,"class_error")
        # We see answer B is true
        return "B"

    # ----------------------------------------------- OPG 9-----------------------------------------------
    def opg9():
        dec = decision_trees()
        root = [263,359,358]
        split2_branch1 = [223,251,197]
        split2_branch2 = [263-223,359-251,358-197]
        dec.purity_gain(root,split2_branch1,split2_branch2,"class_error",accuracy=True)
        # We see the Answer is A
        return "A"

    # ----------------------------------------------- OPG 10-----------------------------------------------
    def opg10():
        ann_obj = ann()
        w02 = 2.2
        weights = [-0.3,0.5]
        matrices = [[-1.2,-1.3,0.6],[-1.0,0,0.9]]
        # First we get the "inner" function
        sigmoid_func = ann_obj.get_ann(w02,weights,matrices,activation="logistic")
        y_1 = sigmoid_func([1,3,3])
        print (y_1)
        # So it must be output 4 = Answer D
        return "D"

    # ----------------------------------------------- OPG 11-----------------------------------------------
    def opg11():
        # Imagine a point a 0,-1, according to the exercise there must be a 1 first
        point = np.array([1,0,-1])
        
        # We project it onto the weights given in the exercise. (a point projected onto w3 is always 0)
        w_a_1 = np.array([-0.77,-5.54,0.01])
        w_a_2 = np.array([0.26,-2.09,-0.03])
        print (point@w_a_1,point@w_a_2)
        
        w_b_1 = np.array([0.51,1.65,0.01])
        w_b_2 = np.array([0.1,3.8,0.04])
        print (point@w_b_1,point@w_b_2)
        
        w_c_1 = np.array([-0.9,-4.39,0])
        w_c_2 = np.array([-0.09,-2.45,-0.04])
        print (point@w_c_1,point@w_c_2)
        
        w_d_1 = np.array([-1.22,-9.88,-0.01])
        w_d_2 = np.array([-0.28,-2.9,-0.01])
        print (point@w_d_1,point@w_d_2)
        
        # We can see from the plot, that this point must be class 2. 
        # The observation is assigned to the class with the highest value in the projection.
        # Only A puts it in class 2
        return "A"

    # ----------------------------------------------- OPG 12-----------------------------------------------
    def opg12():
        x = [1.0,1.2,1.8,2.3,2.6,3.4,4.0,4.1,4.2,4.6]
        init_centers = [1.8,3.3,3.6]
        
        clu = cluster()
        clu.kmeans_1d(x,3,init=init_centers)
        # Only C gives the cluster centers
        return "C"

    # ----------------------------------------------- OPG 13-----------------------------------------------
    def opg13():
        data = [[0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1],
                [0,1,1,1,1,1,0,0,0],
                [1,0,0,0,0,0,0,0,0],
                [1,0,0,1,0,0,0,0,0],
                [0,0,1,1,0,0,0,1,0],
                [0,0,1,1,1,0,0,0,0],
                [0,0,0,0,1,0,0,0,0],
                [0,1,1,0,1,0,0,0,0],
                [0,0,1,1,1,0,1,0,0]]
        df = pd.DataFrame(data)
        
        classes = [0,0,1,1,1,2,2,2,2,2]
        super = supervised()
        super.naive_bayes(classes,df,[1,3,4],[0,1,0],1)
        # This is equal to answer A
        return "A"

    # ----------------------------------------------- OPG 14-----------------------------------------------
    def opg14():
        data = [[0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1],
                [0,1,1,1,1,1,0,0,0],
                [1,0,0,0,0,0,0,0,0],
                [1,0,0,1,0,0,0,0,0],
                [0,0,1,1,0,0,0,1,0],
                [0,0,1,1,1,0,0,0,0],
                [0,0,0,0,1,0,0,0,0],
                [0,1,1,0,1,0,0,0,0],
                [0,0,1,1,1,0,1,0,0]]
        df = pd.DataFrame(data)
        
        it = itemset()
        it.itemsets(df,0.15)
        # We can see this corresponds to A
        return "A"

    # ----------------------------------------------- OPG 15-----------------------------------------------
    def opg15():
        data = [[0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1],
                [0,1,1,1,1,1,0,0,0],
                [1,0,0,0,0,0,0,0,0],
                [1,0,0,1,0,0,0,0,0],
                [0,0,1,1,0,0,0,1,0],
                [0,0,1,1,1,0,0,0,0],
                [0,0,0,0,1,0,0,0,0],
                [0,1,1,0,1,0,0,0,0],
                [0,0,1,1,1,0,1,0,0]]
        df = pd.DataFrame(data)
        
        it = itemset()
        it.confidence(df,[1],[2,3,4,5])
        # We see the answer is B
        return "B"

    # ----------------------------------------------- OPG 16-----------------------------------------------
    def opg16():
        # Run through the apriori algorithm
        
        return "B"

    # ----------------------------------------------- OPG 17-----------------------------------------------
    def opg17():
        data = [[0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1],
                [0,1,1,1,1,1,0,0,0],
                [1,0,0,0,0,0,0,0,0],
                [1,0,0,1,0,0,0,0,0],
                [0,0,1,1,0,0,0,1,0],
                [0,0,1,1,1,0,0,0,0],
                [0,0,0,0,1,0,0,0,0],
                [0,1,1,0,1,0,0,0,0],
                [0,0,1,1,1,0,1,0,0]]

        o_1 = [0,0,0,1,0,0,0,0,0]
        o_2 = [0,0,0,0,0,0,0,0,1]
        o_3 = [0,1,1,1,1,1,0,0,0]
        o_4 = [1,0,0,0,0,0,0,0,0]
        
        sim = similarity()
        
        print ("o1 and o3")
        sim.measures(o_1,o_3)
        print ("o2 and o3")
        sim.measures(o_2,o_3)
        print ("o2 and o4")
        sim.measures(o_2,o_4)
        
        # We can see that answer must be B
        return "B"

    # ----------------------------------------------- OPG 18-----------------------------------------------
    def opg18():
        truth = [1,1,0,1,1,1,0]
        probabilities = [0.14,0.15,0.27,0.61,0.71,0.75,0.81]
        
        ens = ensemble()
        ens.plot_roc_pred(truth,probabilities)
        # So it must be curve 4
        # Answer D
        return "D"

    # ----------------------------------------------- OPG 19-----------------------------------------------
    def opg19():
        # Run through the forward selection method
        # Answer is B
        return "B"

    # ----------------------------------------------- OPG 20-----------------------------------------------
    def opg20():
        numerator = 0.17*0.268
        denominator = 0.17*0.268+0.28*0.366+0.33*0.365
        print (numerator/denominator)
        # So the answer is A
        return "A"

    # ----------------------------------------------- OPG 21-----------------------------------------------
    def opg21():
        # From Sauds script - classification graphing 3 class 3 rule we see it is 
        return "E"

    # ----------------------------------------------- OPG 22-----------------------------------------------
    def opg22():
        outer_fold = 4 # Aka K1
        inner_fold = 5 # Aka K2
        no_tested_values = 5
        models = outer_fold * (inner_fold*no_tested_values+1)
        # We are running 2 models - neural network and regression
        models = models * 2
        print (models)
        return "E"

    # ----------------------------------------------- OPG 23-----------------------------------------------
    def opg23():
        gm = gmm()
        weights = [0.19,0.34,0.48]
        means = [3.177,3.181,3.184]
        std_dev = [0.0062,0.0076,0.0075]
        gm.prob_gmm(3.19,weights,means,std_dev)
        # We see it must be B
        return "B"

    # ----------------------------------------------- OPG 24-----------------------------------------------
    def opg24():
        ada  = adaboost()
        errors = [0,1,1,1,0,1,1]
        
        ada.adaboost(errors,1)
        # We see it must be A
        return "A"

    # ----------------------------------------------- OPG 25-----------------------------------------------
    def opg25():
        # Not sure, calculate the stuff ?
        return "E"

    # ----------------------------------------------- OPG 26-----------------------------------------------
    def opg26():
        # We can see that the it must be Sigma 1 that is right matrix, because x1/x2 are positively correlated:
        # So we see the matrix between x1 and x2 is 0.5     0.56
        #                                           0.56    1.5
        # The correlation coefficient is defined as: p=cov(x,y)/(sigma_x*sigma_y)
        p = 0.56/(0.5*1.5)
        print (p)
        
        # Using function:
        sim_obj = similarity()
        cov = [[0.5, 0.56], [0.56, 1.5]]
        sim_obj.correlation_from_covariance(cov)
        return "A"

    # ----------------------------------------------- OPG 27-----------------------------------------------
    def opg27():
        GMM_Plot = gmm()
        # We plot different options for first cluster:
        #A & D
        mu = [[-7.2],[10.0]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[2.4, -0.4], [-0.4, 1.7]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        #B & C
        mu = [[-7.2],[10.0]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[1.6, 0.9], [0.9, 1.5]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        #Looks like it must A or D
        # We plot the second cluster
        #A
        mu = [[-13.8],[-0.8]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[1.7, -0.3], [-0.3, 2.3]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        #D
        mu = [[-13.8],[-0.8]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[1.6, 0.9], [0.9, 1.5]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        # definitely looks like A, but we check the last one to be sure
        #A
        mu = [[-6.8],[6.4]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[1.6, 0.9], [0.9, 1.5]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        #D
        mu = [[-6.8],[6.4]] # defining the mean/center of the Gaussian (mX, mY)
        cov = [[1.7, -0.3], [-0.3, 2.3]]   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        # It's A
        return "A"

    # -------------------------------- answers dataframe -------------------------------------------------
    def answers(show=True, csv=False, excel=False):
        ans = pd.DataFrame(
            columns=["Student number: s174852"]
        )  # columns = ["OPG", "svar"])

        ans.loc[0] = ""
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
        ans.loc[28] = "Q26: {}".format(exam.opg26())
        # ans.loc[29] = "Q27: {}".format(exam.opg27())

        if excel:
            ans.to_excel(re.sub(".py", "_answers.xlsx", __file__), index=False)
        if csv:
            ans.to_csv(re.sub(".py", "_answers.csv", __file__), index=False)
        if show:
            print(ans)

        return ans


exam.answers()
