# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ScrapBooker.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/19 12:16:25 by archid-           #+#    #+#              #
#    Updated: 2023/04/27 18:14:39 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from ImageProcessor import ImageProcessor


class ScrapBooker:
    def crop(self, array, dim, position=(0, 0)):
        if type(array) != np.ndarray or type(dim) != tuple or type(position) != tuple:
            return None
        if len(dim) != 2 or len(position) != 2:
            return None
        if dim[0] > array.shape[0] or dim[0] <= 0:
            return None
        if position[0] > array.shape[0] or position[0] < 0:
            return None
        if dim[1] > array.shape[1] or dim[1] <= 0:
            return None
        if position[1] > array.shape[1] or position[1] < 0:
            return None
        y_start, y_end = position[1], position[1]+dim[1]
        x_start, x_end = position[0], position[0]+dim[0]
        if y_end > array.shape[1]:
            return None
        if x_end > array.shape[0]:
            return None
        return array[y_start:y_end, x_start:x_end]

    def thin(self, array, n, axis=0):
        if type(array) != np.ndarray or type(axis) != int or (axis != 0 and axis != 1) or type(n) != int or n <= 1:
            return None
        return np.delete(array, range(n, array.shape[axis], n), axis=axis)

    def juxtapose(self, array, n, axis=0):
        if type(array) != np.ndarray or type(axis) != int or (axis != 0 and axis != 1) or type(n) != int or n <= 0:
            return None
        tmp = array.copy()
        for _ in range(n):
            tmp = np.concatenate((tmp, array), axis=axis)
        return tmp

    def mosaic(self, array, dim):
        if type(array) != np.ndarray:
            return None
        if type(dim) != tuple or len(dim) != 2:
            return None
        res = self.juxtapose(array, dim[0], 0)
        res = self.juxtapose(res, dim[1], 1)
        return res


if __name__ == '__main__':
    spb = ScrapBooker()
    arr1 = np.arange(0, 25).reshape(5, 5)
    print(spb.crop(arr1, (3, 1), (1, 0)))
    arr2 = np.array("A B C D E F G H I".split() * 6).reshape(-1, 9)
    print(spb.thin(arr2, 3, 0))
    arr3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    print(spb.juxtapose(arr3, 3, 1))
    arr4 = np.array([[1, 2], [1, 2]])
    print(spb.mosaic(arr4, (2, 2)))

    ip = ImageProcessor()
    img = ip.load("../beavers.jpg")

    ip.display(spb.crop(img, (1200, 720), (0, 0)))
    for i in range(1, 720, 100):
        print(spb.thin(img, i, axis=0))
        # ip.display()
        # ip.display(spb.thin(img, i, axis=1))
