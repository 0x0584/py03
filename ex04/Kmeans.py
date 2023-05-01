# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Kmeans.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/27 19:51:53 by archid-           #+#    #+#              #
#    Updated: 2023/05/01 06:06:34 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from sys import argv
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from pprint import pprint
import matplotlib.pyplot as plt


class KmeansClustering:
    def __init__(self, max_iter=20, ncentroid=5):
        self.ncentroid = ncentroid  # number of centroids
        self.max_iter = max_iter  # number of max iterations to update the centroids

    def compute_distance(X, centroids, ncentroid):
        distance = np.zeros((X.shape[0], ncentroid))
        for k in range(ncentroid):
            distance[:, k] = np.square(  # eclidian distance
                np.linalg.norm(X - centroids[k, :], axis=1))
        return distance

    def label_centroids(distance):
        return np.argmin(distance, axis=1)

    def fit(self, X: np.ndarray):
        np.random.RandomState()
        idx = np.random.permutation(X.shape[0])
        self.centroids = X[idx[:self.ncentroid]]  # selecting clusters

        for _ in range(self.max_iter):
            old_centroids = self.centroids

            # computing distance bewteen each point and all clusters
            distance = KmeansClustering.compute_distance(
                X, old_centroids, self.ncentroid)
            # picking clusters
            self.labels = KmeansClustering.label_centroids(distance)
            # computing cluster centers
            for k in range(self.ncentroid):
                self.centroids[k, :] = np.mean(X[self.labels == k], axis=0)

            if np.all(old_centroids == self.centroids):
                break  # clusters have remained constant

        return self

    def predict(self, X: np.ndarray):
        distance = KmeansClustering.compute_distance(
            X, self.centroids, self.ncentroid)
        return KmeansClustering.label_centroids(distance)

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)


class Args:
    def __init__(self, argv):
        for arg in argv:
            arg = arg.split('=')
            if len(arg) != 2:
                exit(1)


if __name__ == '__main__':

    km = KmeansClustering()
    df = pd.read_csv('solar_system_census.csv').dropna()

    X = np.asarray(df[['height', 'weight']])

    plt.scatter(x=X[:, 0], y=X[:, 1])
    plt.show()

    labels = km.fit_predict(X)

    for cluster in range(km.ncentroid):
        foo = plt.scatter(x=X[labels == cluster, 0], y=X[labels == cluster, 1], s=5)
        points = np.asarray([[x, y] for x, y in zip(
            X[labels == cluster, 0], X[labels == cluster, 1])])
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], c=foo.get_facecolor())

    plt.scatter(x=km.centroids[:, 0],
                y=km.centroids[:, 1], marker="+", c='magenta')
    plt.show()
