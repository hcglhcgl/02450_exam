import numpy as np
import matplotlib.pyplot as plt



def draw_curve_from_diagonal_values(values):
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
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()
