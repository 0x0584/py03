# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Kmeans.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/27 19:51:53 by archid-           #+#    #+#              #
#    Updated: 2023/05/01 08:49:02 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from sys import argv
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from pprint import pprint
import matplotlib.pyplot as plt


class KmeansClustering:
    def __init__(self, max_iter=20, ncentroid=4, random_state=123):
        self.ncentroid = ncentroid
        self.max_iter = max_iter
        self.random_state = random_state

    def compute_distance(X, centroids, ncentroid):
        distance = np.zeros((X.shape[0], ncentroid))
        for k in range(ncentroid):
            distance[:, k] = np.square(
                np.linalg.norm(X - centroids[k, :], axis=1))
        return distance

    def label_centroids(distance):
        return np.argmin(distance, axis=1)

    def fit(self, X: np.ndarray):
        np.random.RandomState(self.random_state)
        idx = np.random.permutation(X.shape[0])
        self.centroids = X[idx[:self.ncentroid]]

        for _ in range(self.max_iter):
            old_centroids = self.centroids
            distance = KmeansClustering.compute_distance(
                X, old_centroids, self.ncentroid)
            self.labels = KmeansClustering.label_centroids(distance)
            for k in range(self.ncentroid):
                self.centroids[k, :] = np.mean(X[self.labels == k], axis=0)

            if np.all(old_centroids == self.centroids):
                break

        return self

    def predict(self, X: np.ndarray):
        distance = KmeansClustering.compute_distance(
            X, self.centroids, self.ncentroid)
        return KmeansClustering.label_centroids(distance)

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)


class Parser:
    def __init__(self, argv):
        self.args = dict()
        self.filepath = None
        self.columns = None
        for arg in argv[1:]:
            arg = arg.split('=')
            if len(arg) != 2:
                print("invalid commandline args ")
                exit(1)
            if arg[0] == "filepath":
                self.filepath = arg[1]
            elif arg[0] == "ncentroid":
                self.args["ncentroid"] = int(arg[1])
            elif arg[0] == "max_iter":
                self.args["max_iter"] = int(arg[1])
            elif arg[0] == "random_state":
                self.args["random_state"] = int(arg[1])
            elif arg[0] == "columns":
                self.columns = arg[1].split(",")
            else:
                print("invalid commandline args ")
                exit(1)
        if self.filepath is None:
            print("invalid commandline args ")
            exit(1)


if __name__ == '__main__':
    parser = Parser(argv)
    km = KmeansClustering(**parser.args)

    df = pd.read_csv(parser.filepath).dropna()
    X = np.asarray(df if parser.columns is None else df[parser.columns])
    print(X.shape)
    labels = km.fit_predict(X)

    print(km.centroids)

    count = dict()
    for label in labels:
        if label in count.keys():
            count[label] += 1
        else:
            count[label] = 1

    for label in np.unique(labels):
        print("label {}: {}".format(
            label, count[label] if label in count.keys() else 0))

    plt.scatter(x=X[:, 0], y=X[:, 1])
    plt.show()

    def plot_centroids(cluster, x_axis, y_axis):
        foo = plt.scatter(x=X[labels == cluster, x_axis],
                          y=X[labels == cluster, y_axis], s=5)
        points = np.asarray([[x, y] for x, y in zip(
            X[labels == cluster, x_axis], X[labels == cluster, y_axis])])
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1],
                     c=foo.get_facecolor())
        plt.scatter(x=km.centroids[:, x_axis],
                    y=km.centroids[:, y_axis], marker="+", c='magenta')
        plt.xlabel(parser.columns[x_axis])
        plt.ylabel(parser.columns[y_axis])

    for cluster in range(km.ncentroid):
        plot_centroids(cluster, 0, 1)
    plt.show()

    for cluster in range(km.ncentroid):
        plot_centroids(cluster, 0, 2)
    plt.show()

    for cluster in range(km.ncentroid):
        plot_centroids(cluster, 2, 1)
    plt.show()