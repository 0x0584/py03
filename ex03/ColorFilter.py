# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ColorFilter.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/19 15:29:03 by archid-           #+#    #+#              #
#    Updated: 2023/05/01 01:32:16 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np
from ImageProcessor import ImageProcessor
from sklearn.cluster import KMeans
from threading import Thread
from time import time

NUM_THREADS = 16


class ColorFilter:

    def predict_colour_shades(foo, n_shades_per_colour=64, random_state=777):
        K = KMeans(n_clusters=n_shades_per_colour,
                   random_state=random_state, n_init=10)
        return np.asarray(K.fit(foo).cluster_centers_[K.predict(foo)])

    def moyenne(foo):
        out_shape = (foo.shape[0], foo.shape[1], 1)
        arr = np.zeros(out_shape, dtype=foo.dtype)
        for i in range(foo.shape[0]):
            for j in range(foo.shape[1]):
                arr[i][j] = sum(foo[i][j]) / 3
        in_shape = (foo.shape[0] * foo.shape[1], 1)
        tmp = ColorFilter.predict_colour_shades(arr.reshape(in_shape))
        tmp = tmp.reshape((foo.shape[0], foo.shape[1]))
        return tmp

    def shades(foo):
        def pick_colour(foo, index):
            return foo[:, :, index]

        def spectrum_shading(foo):
            return ColorFilter.predict_colour_shades(pick_colour(foo, 0)), \
                ColorFilter.predict_colour_shades(pick_colour(foo, 1)),    \
                ColorFilter.predict_colour_shades(pick_colour(foo, 2))

        r_shd, g_shd, b_shd = spectrum_shading(foo)
        arr = []
        for r_row, g_row, b_row in zip(r_shd, g_shd, b_shd):
            arr.append([])
            for r, g, b in zip(r_row, g_row, b_row):
                arr[-1].append([int(r), int(g), int(b)])
                
        return np.asarray(arr, dtype=foo.dtype)

    def invert(self, array):
        if type(array) != np.ndarray:
            return None
        return 1 - array

    def to_blue(self, array):
        if type(array) != np.ndarray:
            return None
        tmp = array.copy()
        tmp[..., :2] = np.zeros(array.shape, dtype=array.dtype)[..., :2]
        return tmp

    def to_green(self, array):
        if type(array) != np.ndarray:
            return None
        return array * [0, 1, 0]

    def to_red(self, array):
        return array - (self.to_green(array) + self.to_blue(array))

    def to_celluloid_impl(array, filter):
        return filter(array)

    def to_celluloid(self, array):
        return ColorFilter.to_celluloid_impl(array, ColorFilter.shades)

    def weighted_filter(pixel, weights):
        value = int(np.mean(
            sum([weights[colour] * val for val, colour in zip(pixel, weights)]))) % 255
        return np.asarray([value, value, value])

    def mean_filter(pixel, *args):
        value = int(np.mean(sum(pixel))) % 255
        return np.asarray([value, value, value])

    def to_grayscale(self, array, filter, **kwargs):
        start_time = time()

        weights = None
        if filter == "m" or filter == "mean":
            filter = ColorFilter.mean_filter
        elif filter == "w" or filter == "weight":
            if not "weights" in kwargs.keys() or type(kwargs["weights"]) != dict:
                weights = {"r": 33.3, "g": 33.3, "b": 33.3}
            else:
                weights = kwargs["weights"]
                if not ("r" in weights.keys() and "g" in weights.keys() and "b" in weights.keys()):
                    if sum([weight for weight in weights]) <= 1. - 0.14159:
                        weights = {"r": 33.3, "g": 33.3, "b": 33.3}
            filter = ColorFilter.weighted_filter
        else:
            return array

        gried = np.zeros(array.shape, dtype=array.dtype)
        def set_pixel(row_index):
            for i in range(row_index, gried.shape[0], NUM_THREADS):
                for j in range(gried.shape[1]):
                    gried[i][j] = filter(array[i][j], weights)

        thds = list()
        for i in range(NUM_THREADS):
            thds.append(Thread(target=set_pixel, args=[i]))
        for th in thds:
            th.start()
        for th in thds:
            th.join()

        print("%.3fs" % (time() - start_time))
        return gried


if __name__ == '__main__':
    cf = ColorFilter()
    ip = ImageProcessor()

    img = ip.load('42AI.png')

    ip.display(img)
    ip.display(cf.invert(img))
    ip.display(cf.to_red(img))
    ip.display(cf.to_green(img))
    ip.display(cf.to_blue(img))
    ip.display(ColorFilter.to_celluloid_impl(img, ColorFilter.moyenne))
    ip.display(ColorFilter.to_celluloid_impl(img, ColorFilter.shades))
    ip.display(cf.to_grayscale(img, "m"))
    ip.display(cf.to_grayscale(img, "w", weights={
               "r": 0.1, "g": 0.8, "b": 0.1}))
