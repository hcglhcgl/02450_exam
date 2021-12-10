import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


def rect(x):
    return np.max([x, 0])


def tanh(x):
    return np.sinh(x) / np.cosh(x)


def get_ann(w02, weights, matrices, activation='logistic'):
    """

    ann = get_ann(2.84, [3.25, 3.46], [[21.78, -1.65, 0, -13.26, -8.46], [-9.6, -0.44, 0.01, 14.54, 9.5]], "rect")
    y = ann([1, 6.8, 225, 0.44, 0.68])

    REMEMBER TO PUT "1" AT THE START OF ann([..])
    """
    matrices = np.array([np.matrix(m).T for m in matrices])
    weights = np.array(weights)

    activation_func = {
        "logistic": logistic,
        "rect": rect,
        "tanh": tanh
    }[activation]

    def predict_y(x):
        x = np.matrix(x)
        activated_matrices = np.array([activation_func(x * m) for m in matrices])
        ann_sum = 0
        for (i, _) in enumerate(activated_matrices):
            ann_sum = ann_sum + (activated_matrices[i] * weights[i])
        return w02 + ann_sum

    return predict_y
