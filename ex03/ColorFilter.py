# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ColorFilter.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/19 15:29:03 by archid-           #+#    #+#              #
#    Updated: 2023/04/30 16:21:15 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np
from ImageProcessor import ImageProcessor
from sklearn.cluster import KMeans

# R G B colour, invert return the complementary, to apply colour filter remove the tuple


class ColorFilter:

    def predict_colour_shades(foo, n_shades_per_colour=64, random_state=777):
        K = KMeans(n_clusters=n_shades_per_colour,
                   random_state=random_state)
        return np.asarray(K.fit(foo).cluster_centers_[K.predict(foo)])

    def moyenne(foo):
        out_shape = (foo.shape[0], foo.shape[1], 1)
        arr = np.zeros(out_shape)
        for i in range(foo.shape[0]):
            for j in range(foo.shape[1]):
                arr[i][j] = sum(foo[i][j]) / 3
        in_shape = (foo.shape[0] * foo.shape[1], 1)
        tmp = ColorFilter.predict_colour_shades(arr.reshape(in_shape))
        tmp = tmp.reshape((foo.shape[0], foo.shape[1]))
        print(type(tmp))
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
        return np.asarray(arr)

    def invert(self, array):
        if type(array) != np.ndarray:
            return None
        return 1 - array

    def to_blue(self, array):
        if type(array) != np.ndarray:
            return None
        tmp = array.copy()
        tmp[..., :2] = np.zeros(array.shape)[..., :2]
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

    def to_grayscale(self, array, filter, **kwargs):
        pass

if __name__ == '__main__':
    cf = ColorFilter()
    img = ImageProcessor()

    foo = img.load('42AI.png')

    img.display(foo)
    img.display(cf.invert(foo))
    img.display(cf.to_red(foo))
    img.display(cf.to_green(foo))
    img.display(cf.to_blue(foo))
    img.display(ColorFilter.to_celluloid_impl(foo, ColorFilter.moyenne))
    img.display(ColorFilter.to_celluloid_impl(foo, ColorFilter.shades))
