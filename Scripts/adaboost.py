import numpy as np
from basic_operations import choose_mode

string_table = """Round 1 Round 2 Round 3 Round 4
O1 0.1000 0.0714 0.0469 0.0319
O2 0.1000 0.0714 0.0469 0.0319
O3 0.1000 0.1667 0.1094 0.2059
O4 0.1000 0.0714 0.0469 0.0319
O5 0.1000 0.1667 0.1094 0.2059
O6 0.1000 0.0714 0.0469 0.0882
O7 0.1000 0.0714 0.0469 0.0319
O8 0.1000 0.1667 0.3500 0.2383
O9 0.1000 0.0714 0.1500 0.1021
O10 0.1000 0.0714 0.0469 0.0319"""

last_wrong = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
M = choose_mode(string_table)


def get_alpha_given_w(M, last_wrong):
    """

    :param M: Matrix of weights in dim. Number of Obs.x Number of Rounds
    :param last_wrong: Binary vector of wrong classified obs. at the last round.
    :return:
    """
    alpha = []
    for i in range(M.shape[1] - 1):
        er = np.dot(M[:, i], (M[:, i + 1] > M[:, i]))
        alpha.append(1 / 2 * np.log((1 - er) / er))
    er = np.dot(M[:, M.shape[1] - 1], last_wrong)
    alpha.append(1 / 2 * np.log((1 - er) / er))
    print(alpha)


def get_weights(w_init, false_ones):
    """
    :param w_init: initial weights
    :param false_ones: binary vector of false classified observations
    :return:
    """
    er = np.dot(w_init, false_ones)
    alpha = (1 / 2)* np.log((1 - er) / er)
    w_init[false_ones > 0] = w_init[false_ones> 0] * np.exp(alpha)
    w_init[false_ones < 1] = w_init[false_ones < 1] * np.exp(-alpha)
    w_init = w_init/np.sum(w_init)
    print(w_init)
    return w_init


# get_alpha_given_w(M, last_wrong)

get_weights(M[:, 1], np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0]))
