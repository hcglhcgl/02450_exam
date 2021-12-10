from sklearn.cluster import KMeans
import numpy as np

x_data = [5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4]
x = np.array(x_data).reshape(-1, 1)

"""
k-means without initialization 
"""
kmeans = KMeans(n_clusters=2).fit(x)
print(kmeans.predict(x))

"""
k-means with initialization
"""
init_points = np.array([5.7, 6.0, 6.2]).reshape(-1, 1)
kmeans = KMeans(n_clusters=3, init=init_points).fit(x)
print(kmeans.predict(x))
