# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Kmeans.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/27 19:51:53 by archid-           #+#    #+#              #
#    Updated: 2023/04/28 21:36:19 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from sys import argv
import pandas as pd
import numpy as np


class KmeansClustering:
    def __init__(self, max_iter=20, ncentroid=5):
        self.ncentroid = ncentroid  # number of centroids
        self.max_iter = max_iter  # number of max iterations to update the centroids

    def centroids_init(self, X: np.ndarray):
        np.random.RandomState()
        idx = np.random.permutation(X.shape[0])
        return X[idx[:self.ncentroid]]

    def compute_distance(self, X: np.ndarray, centroids: np.ndarray):
        distance = np.zeros((X.shape[0], self.ncentroid))
        for k in range(self.ncentroid):
            distance[:, k] = np.square( #eclidian distance
                np.linalg.norm(X - centroids[k, :], axis=1))
        return distance

    def fit(self, X: np.ndarray):
        print(X)
        centroids = self.centroids_init(X)
        for i in range(self.max_iter):
            old_centroids = centroids
            distance = self.compute_distance(X, centroids)
            # print()
        return self

    def predict(self, X: np.ndarray):
        return X


class Args:
    def __init__(self, argv):
        for arg in argv:
            arg = arg.split('=')
            if len(arg) != 2:
                exit(1)


if __name__ == '__main__':

    print(argv)
    km = KmeansClustering()
    df = pd.read_csv('solar_system_census.csv')

    X = np.asarray(df[['height', 'weight', 'bone_density']])

    km.fit(X)
