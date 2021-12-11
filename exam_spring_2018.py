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
    # Solve using attribute type .pdf
    # Answer is C, because for each attribute the number 0 has a specific physical meaning
        return "C"

    # ----------------------------------------------- OPG 2-----------------------------------------------
    pca = pca_calc()
    def opg2():
        # We insert the the diagonal numbers from the S matrix
        diag_values = [13.5,7.6,6.5,5.8,3.5,2.0]
        pca.draw_curve_from_diagonal_values(diag_values)
        # Can also be done by:
        pca.var_explained(diag_values)
        return "A"

    # ----------------------------------------------- OPG 3-----------------------------------------------
    def opg3():
        # We define the V matrix:
        v_matrix = [[0.38,-0.51,0.23,0.47,-0.55,0.11],[0.41,0.41,-0.53,0.24,0.00,0.58],[0.50,0.34,-0.13,0.15,-0.05,-0.77],[0.29,0.48,0.78,-0.17,0.00,0.23],[0.45,-0.42,0.09,0.03,0.78,0.04],[0.39,-0.23,-0.20,-0.82,-0.30,0.04]]
        pca.proj(v_matrix,[-0.1,0.2,0.1,-0.3,1,0.5],[1,2],V_full=True)
        # From this we see the answer is C
        return "C"

    # ----------------------------------------------- OPG 4-----------------------------------------------
    def opg4():
        # According to text on the end of page 331 in the book, the answer is B

        return "B"

    # ----------------------------------------------- OPG 5-----------------------------------------------
    def opg5():
        GMM_Plot = gmm()
        # We plot different options for first cluster:
        mu = np.array([[1.84],[2.43]])  # defining the mean/center of the Gaussian (mX, mY)
        cov = np.array([[0.2639, 0.0803], [0.0803, 0.0615]])   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
                
        mu = np.array([[1.84],[2.43]])  # defining the mean/center of the Gaussian (mX, mY)
        cov = np.array([[3.82, 1.71], [1.71, 0.7672]])   # defining the covariance matrix
        GMM_Plot.plot_gmm(mu,cov)
        
        #It's not A 
        
        # We plot different options for third cluster:
        mu = np.array([[-0.6687],[-0.7343]])  # defining the mean/center of the Gaussian (mX, mY)
        cov = np.array([[4.0475, -1.5818], [-1.5818, 1.1146]])   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
                
        mu = np.array([[-0.6687],[-0.7343]])  # defining the mean/center of the Gaussian (mX, mY)
        cov = np.array([[0.1166, -0.0771], [-0.0771, 0.1729]])   # defining the covariance matrix left-right, row by row
        GMM_Plot.plot_gmm(mu,cov)
        
        #It's not B
        
        # The number infront of N is the "mixing proportion" correlated to the amount of observations.
        # Since it must be lowest for the first cluster, it must mean the answer D is correct.
        return "D"

    # ----------------------------------------------- OPG 6-----------------------------------------------
    def opg6():
        # Use Method described in "Eksamensforberedelse" or Mikkels Script
        return "A"

    # ----------------------------------------------- OPG 7-----------------------------------------------
    def opg7():

        return "E"

    # ----------------------------------------------- OPG 8-----------------------------------------------
    def opg8():
     #  HowTO: 
     # ann = get_ann(2.84, [3.25, 3.46], [[21.78, -1.65, 0, -13.26, -8.46], [-9.6, -0.44, 0.01, 14.54, 9.5]], "rect")
     #   y = ann([1, 6.8, 225, 0.44, 0.68])
        ann_obj = ann()
        an_model = ann_obj.get_ann(0.3799E-6, [-0.3440E-6, 0.0429E-6], [[0.0189, 0.9159, -0.4256], [3.7336, -0.8003, 5.0741]], "logistic")
     #  Always put "1" as first for some reason 
        y_1 = an_model([1,0, 3])
        y_2 = an_model([1,24, 0])
        print ("y1:")
        print (y_1)
        print ("y2:")
        print (y_2)
    # We can see the only possible answer is D
        return "D"

    # ----------------------------------------------- OPG 9-----------------------------------------------
    def opg9():
        outer_fold = 5
        inner_fold = 10
        models = 3
        
        H = 6
        
        models_trained = outer_fold * (inner_fold+1)*(models*H)
        
        print (models_trained)
    # This gives 990 models trained - 19 units gives more than 3000. So maximum is 6
        return "A"

    # ----------------------------------------------- OPG 10-----------------------------------------------
    def opg10():
        # Solved in Maple

        return "C"

    # ----------------------------------------------- OPG 11-----------------------------------------------
    def opg11():
        tree = decision_trees()
        tree.purity_gain([32,24],[23,8],[9,16],"class_error")
        return "B"

    # ----------------------------------------------- OPG 12-----------------------------------------------
    def opg12():
    # We can see the true positive rate is 23, false positive 8
    # And the False Negative is 9, True Negative 16
    # A confusion matrix is in the form: 
    # TP FN
    # FP TN
    # So the answer must be Confusion Matrix 2 = Answer B
        return "B"

    # ----------------------------------------------- OPG 13-----------------------------------------------
    def opg13():
        # Softmax gives the probability
        return "D"

    # ----------------------------------------------- OPG 14-----------------------------------------------
    def opg14():
        TP=14
        FN=18
        FP=10
        TN=14
        conf = ensemble()
        conf.conf_matrix_stats(TP,FN,FP,TN,"all")
        # We can see the precision is 7/12 = 0.58333, so the answer is A
        return "A"

    # ----------------------------------------------- OPG 15-----------------------------------------------
    def opg15():
        # Using Mathpix "ASCII Math"
        data =[[0,8.55,0.43,1.25,1.14,3.73,2.72,1.63,1.68,1.28],
               [8.55,0,8.23,8.13,8.49,6.84,8.23,8.28,8.13,7.66],
               [0.43,8.23,0,1.09,1.10,3.55,2.68,1.50,1.52,1.05],
               [1.25,8.13,1.09,0,1.23,3.21,2.17,1.29,1.33,0.56],
               [1.14,8.49,1.10,1.23,0,3.20,2.68,1.56,1.50,1.28],
               [3.73,6.84,3.55,3.21,3.20,0,2.98,2.66,2.50,3.00],
               [2.72,8.23,2.68,2.17,2.68,2.98,0,2.28,2.30,2.31],
               [1.63,8.28,1.50,1.29,1.56,2.66,2.28,0,0.25,1.46],
               [1.68,8.13,1.52,1.33,1.50,2.50,2.30,0.25,0,1.44],
               [1.28,7.66,1.05,0.56,1.28,3.00,2.31,1.46,1.44,0]]
        red_Os = [2,6,7,8,9]
        black_Os = [1,3,4,5,10]
        df = pd.DataFrame(data)
        super = supervised()
        prediction = super.knn_dist_pred_2d(df,red_Os,black_Os,1,show=True)
        super.pred_stats(prediction["True_label"],prediction["Predicted_label"],show=True)
        # We see the answer must be 10%
        return "B"

    # ----------------------------------------------- OPG 16-----------------------------------------------
    def opg16():
        data =[[0,8.55,0.43,1.25,1.14,3.73,2.72,1.63,1.68,1.28],
               [8.55,0,8.23,8.13,8.49,6.84,8.23,8.28,8.13,7.66],
               [0.43,8.23,0,1.09,1.10,3.55,2.68,1.50,1.52,1.05],
               [1.25,8.13,1.09,0,1.23,3.21,2.17,1.29,1.33,0.56],
               [1.14,8.49,1.10,1.23,0,3.20,2.68,1.56,1.50,1.28],
               [3.73,6.84,3.55,3.21,3.20,0,2.98,2.66,2.50,3.00],
               [2.72,8.23,2.68,2.17,2.68,2.98,0,2.28,2.30,2.31],
               [1.63,8.28,1.50,1.29,1.56,2.66,2.28,0,0.25,1.46],
               [1.68,8.13,1.52,1.33,1.50,2.50,2.30,0.25,0,1.44],
               [1.28,7.66,1.05,0.56,1.28,3.00,2.31,1.46,1.44,0]]
        df = pd.DataFrame(data)
        clu=cluster()
        clu.dendro_plot(df,"average",sort="descending")
        # We can see  it must be Dendro 4
        return "D"

    # ----------------------------------------------- OPG 17-----------------------------------------------
    def opg17():
        # LOOK AT DENDOGRAM 1
        clu = cluster()
        #Cutoff at the level of 3 clusters = 3 vertical lines
        # Here we see O2 has been seperated, and O6 as well. 
        # Since O2 was seperated in the first cluster, we give it 2
        # And O6 we give 3
        # The rest have not been seperated yet, but majority black so we give it 1
        clusters_1=[1,2,1,1,1,3,1,1,1,1]
        
        # The true class labels are just O1 - O10 where black = 1, red = 2
        clusters_2=[1,2,1,1,1,2,2,2,2,1]
        clu.cluster_similarity(clusters_1,clusters_2)
        #Which gives Rand = 0.511111
        return "C"

    # ----------------------------------------------- OPG 18-----------------------------------------------
    def opg18():
        #Same dataset
        data =[[0,8.55,0.43,1.25,1.14,3.73,2.72,1.63,1.68,1.28],
               [8.55,0,8.23,8.13,8.49,6.84,8.23,8.28,8.13,7.66],
               [0.43,8.23,0,1.09,1.10,3.55,2.68,1.50,1.52,1.05],
               [1.25,8.13,1.09,0,1.23,3.21,2.17,1.29,1.33,0.56],
               [1.14,8.49,1.10,1.23,0,3.20,2.68,1.56,1.50,1.28],
               [3.73,6.84,3.55,3.21,3.20,0,2.98,2.66,2.50,3.00],
               [2.72,8.23,2.68,2.17,2.68,2.98,0,2.28,2.30,2.31],
               [1.63,8.28,1.50,1.29,1.56,2.66,2.28,0,0.25,1.46],
               [1.68,8.13,1.52,1.33,1.50,2.50,2.30,0.25,0,1.44],
               [1.28,7.66,1.05,0.56,1.28,3.00,2.31,1.46,1.44,0]]
        df = pd.DataFrame(data)
        anom = anomaly()
        anom.ARD(df,1,2)
        # The ARD is 0.169
        return "C"

    # ----------------------------------------------- OPG 19-----------------------------------------------
    def opg19():
        # We add the binary table
        data =[[1,0,1,0,1,0,1,0,1,0,1,0],
               [0,1,0,1,0,1,0,1,0,1,0,1],
               [1,0,0,1,1,0,1,0,1,0,1,0],
               [1,0,1,0,1,0,0,1,0,1,1,0],
               [0,1,1,0,1,0,1,0,1,0,1,0],
               [0,1,0,1,0,1,0,1,0,1,0,1],
               [0,1,1,0,1,0,0,1,0,1,0,1],
               [1,0,1,0,1,0,1,0,0,1,0,1],
               [0,1,0,1,1,0,1,0,0,1,0,1],
               [1,0,0,1,0,1,0,1,0,1,1,0]]
        df = pd.DataFrame(data)
        assc = association_mining()
        # We add the indexes of the assocation rule elements
        assc.rule_stats(df,[3,5,7,9],[11])
        # Rule = 0.2
        # Confidence = 0.666
        return "B"

    # ----------------------------------------------- OPG 20-----------------------------------------------
    def opg20():
        # We got in the opg19
        return "C"

    # ----------------------------------------------- OPG 21-----------------------------------------------
    def opg21():
        
        # We add the binary table
        data =[[1,0,1,0,1,0,1,0,1,0,1,0],
               [0,1,0,1,0,1,0,1,0,1,0,1],
               [1,0,0,1,1,0,1,0,1,0,1,0],
               [1,0,1,0,1,0,0,1,0,1,1,0],
               [0,1,1,0,1,0,1,0,1,0,1,0],
               [0,1,0,1,0,1,0,1,0,1,0,1],
               [0,1,1,0,1,0,0,1,0,1,0,1],
               [1,0,1,0,1,0,1,0,0,1,0,1],
               [0,1,0,1,1,0,1,0,0,1,0,1],
               [1,0,0,1,0,1,0,1,0,1,1,0]]
        df = pd.DataFrame(data)
        super = supervised()
        super.naive_bayes_2class([1,0,1,1,1,0,0,0,0,1],df,[3,5,7,9],[1,1,1,1],1)
        # We see the answer is 0.0816 = 4/49. Answer C
        return "C"

    # ----------------------------------------------- OPG 22-----------------------------------------------
    def opg22():
        ens = ensemble()
        true=[1,0,1,1,1,0,0,0,0,1]
        pred=[1,0,1,0,1,0,0,0,0,0]
        ens.plot_roc(true,pred)
        # We can see this corresponds to ROC Curve 1.
        # The answer is A
        return "A"

    # ----------------------------------------------- OPG 23-----------------------------------------------
    def opg23():
        weights = np.array([[0.1000,0.0714,0.0469,0.0319],
                  [0.1000,0.0714,0.0469,0.0319],
                  [0.1000,0.1667,0.1094,0.2059],
                  [0.1000,0.0714,0.0469,0.0319],
                  [0.1000,0.1667,0.1094,0.2059],
                  [0.1000,0.0714,0.0469,0.0882],
                  [0.1000,0.0714,0.0469,0.0319],
                  [0.1000,0.1667,0.3500,0.2383],
                  [0.1000,0.0714,0.1500,0.1021],
                  [0.1000,0.0714,0.0469,0.0319]])
        # We can see from the figure, that in the last round O3 and O6 are wrong 
        last_wrong = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        
        ada = adaboost()
        alpha = ada.get_alpha_given_w(weights,last_wrong)
        
        # Now we look at O5 in the boosting round figures.
        # It is correctly classified in Round 2 and 4
        O5_correctly = alpha[1] + alpha[3]
        O5_wrong = alpha[0] + alpha[2]
        print ("O5_correct: ",O5_correctly,"O5_wrong: ",O5_wrong)
        # Now we look at O6.
        # It is correctly classified in Round 1 and 2
        O6_correctly = alpha[0] + alpha[1]
        O6_wrong = alpha[2] + alpha[3]
        print ("O6_correct: ",O6_correctly,"O6_wrong: ",O6_wrong)
        # For both observations, alpha is higher for the correct answer, so they will be correctly classified 
        return "B"

    # ----------------------------------------------- OPG 24-----------------------------------------------
    def opg24():
        # The Classifier 2 looks special, with a smooth line. It cannot be a nearest neighbor classifier
        # So the answer must be D
        return "D"

    # ----------------------------------------------- OPG 25-----------------------------------------------
    def opg25():
        observations = [1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 3.0, 3.2]
        clus = cluster()
        clus.kmeans_1d(observations,3,init=[1.0,1.2,1.5])
        # We take the average of each cluster in the possible answers
        # We see that only answer A correspond to the cluster centers of the converged k-means algortihm
        return "A"

    # ----------------------------------------------- OPG 26-----------------------------------------------
    def opg26():
        # We use the maple sheet vector_norms
        return "D"

    # ----------------------------------------------- OPG 27-----------------------------------------------
    def opg27():

        return "E"

    # -------------------------------- answers dataframe -------------------------------------------------
    def answers(show=True, csv=False, excel=False):
        ans = pd.DataFrame(
            columns=["Student number: s174852"]
        )  # columns = ["OPG", "svar"])

#        ans.loc[0] = ""
#        ans.loc[1] = "Q01: {}".format(exam.opg1())
#        ans.loc[2] = "Q02: {}".format(exam.opg2())
#        ans.loc[3] = "Q03: {}".format(exam.opg3())
#        ans.loc[4] = "Q04: {}".format(exam.opg4())
#        ans.loc[5] = "Q05: {}".format(exam.opg5())
#        ans.loc[6] = "Q06: {}".format(exam.opg6())
#        ans.loc[7] = "Q07: {}".format(exam.opg7())
#        ans.loc[8] = "Q08: {}".format(exam.opg8())
#        ans.loc[9] = "Q09: {}".format(exam.opg9())
#        ans.loc[10] = "Q10: {}".format(exam.opg10())
#        ans.loc[11] = ""

#        ans.loc[12] = "Q11: {}".format(exam.opg11())
#        ans.loc[13] = "Q12: {}".format(exam.opg12())
#        ans.loc[14] = "Q13: {}".format(exam.opg13())
#        ans.loc[15] = "Q14: {}".format(exam.opg14())
#        ans.loc[16] = "Q15: {}".format(exam.opg15())
#        ans.loc[17] = "Q16: {}".format(exam.opg16())
#        ans.loc[18] = "Q17: {}".format(exam.opg17())
#        ans.loc[19] = "Q18: {}".format(exam.opg18())
#        ans.loc[20] = "Q19: {}".format(exam.opg19())
#        ans.loc[21] = "Q20: {}".format(exam.opg20())
#        ans.loc[22] = ""

#        ans.loc[23] = "Q21: {}".format(exam.opg21())
#        ans.loc[24] = "Q22: {}".format(exam.opg22())
#        ans.loc[25] = "Q23: {}".format(exam.opg23())
#        ans.loc[26] = "Q24: {}".format(exam.opg24())
#        ans.loc[27] = "Q25: {}".format(exam.opg25())
#        ans.loc[28] = "Q26: {}".format(exam.opg26())
#        ans.loc[29] = "Q27: {}".format(exam.opg27())

        if excel:
            ans.to_excel(re.sub(".py", "_answers.xlsx", __file__), index=False)
        if csv:
            ans.to_csv(re.sub(".py", "_answers.csv", __file__), index=False)
        if show:
            print(ans)

        return ans


exam.answers()
