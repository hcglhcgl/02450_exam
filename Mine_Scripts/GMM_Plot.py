import numpy as np
import matplotlib.pyplot as plt


def plot_gmm(m,cov):
    N = 1000

    cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
    cov_det = np.linalg.det(cov)  # determinant of covariance matrix
    # Plotting
    x = np.linspace(-2, 6, N) # Size of coordinate system
    y = np.linspace(-2, 4, N)
    X,Y = np.meshgrid(x,y)
    coe = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
    Z = coe * np.e ** (-0.5 * (cov_inv[0,0]*(X-m[0])**2 + (cov_inv[0,1] + cov_inv[1,0])*(X-m[0])*(Y-m[1]) + cov_inv[1,1]*(Y-m[1])**2))
    plt.contour(X,Y,Z)
    plt.show()