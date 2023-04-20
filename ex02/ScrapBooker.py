# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ScrapBooker.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/19 12:16:25 by archid-           #+#    #+#              #
#    Updated: 2023/04/19 15:17:03 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from ImageProcessor import ImageProcessor

class ScrapBooker:
    def crop(self, array, dim, position=(0,0)):
        """
        Crops the image as a rectangle via dim arguments (being the new height
        and width of the image) from the coordinates given by position arguments.
        Args:
        -----
        array: numpy.ndarray
        dim: tuple of 2 integers.
        position: tuple of 2 integers.
        Return:
        -------
        new_arr: the cropped numpy.ndarray.
        None (if combinaison of parameters not compatible).
        Raise:
        ------
        This function should not raise any Exception.
        """
        if type(array) != np.ndarray or type(dim) != tuple or type(position) != tuple:
            raise ValueError()
        if len(dim) != 2 or len(position) != 2:
            raise ValueError()
        if dim[0] > array.shape[0] or dim[0] <= 0:
            raise ValueError()
        if position[0] > array.shape[0] or position[0] < 0:
            raise ValueError()
        if dim[1] > array.shape[1] or dim[1] <= 0:
            raise ValueError()
        if position[1] > array.shape[1] or position[1] < 0:
            raise ValueError()
        y_start, y_end = position[1], position[1]+dim[1]
        x_start, x_end = position[0], position[0]+dim[0]
        if y_end > array.shape[1]:
            raise ValueError()
        if x_end > array.shape[0]:
            raise ValueError()
        return array[y_start:y_end, x_start:x_end]

    def thin(self, array, n, axis=0):
        """
        Deletes every n-th line pixels along the specified axis (0: Horizontal, 1: Vertical)
        Args:
        -----
        array: numpy.ndarray.
        n: non null positive integer lower than the number of row/column of the array
        (depending of axis value).
        axis: positive non null integer.
        Return:
        -------
        new_arr: thined numpy.ndarray.
        None (if combinaison of parameters not compatible).
        Raise:
        ------
        This function should not raise any Exception.
        """
        if type(array) != np.ndarray or type(axis) != int or (axis != 0 and axis != 1) or type(n) != int or n < 0:
            raise ValueError()
        return np.delete(array, range(0, array.shape[axis], n))

    def juxtapose(self, array, n, axis=0):
        """
        Juxtaposes n copies of the image along the specified axis.
        Args:
        -----
        array: numpy.ndarray.
        n: positive non null integer.
        axis: integer of value 0 or 1.
        Return:
        -------
        new_arr: juxtaposed numpy.ndarray.
        None (combinaison of parameters not compatible).
        Raises:
        -------
        This function should not raise any Exception.
        """
        if type(array) != np.ndarray or type(axis) != int or (axis != 0 and axis != 1) or type(n) != int or n < 0:
            raise ValueError()
        tmp = array.copy()
        for _ in range(n):
            tmp = np.concatenate((tmp, array), axis=axis)
        return tmp

    def mosaic(self, array, dim):
        """
        Makes a grid with multiple copies of the array. The dim argument specifies
        the number of repetition along each dimensions.
        Args:
        -----
        array: numpy.ndarray.
        dim: tuple of 2 integers.
        Return:
        -------
        new_arr: mosaic numpy.ndarray.
        None (combinaison of parameters not compatible).
        Raises:
        -------
        This function should not raise any Exception.
        """
        if type(array) != np.ndarray:
            raise ValueError()
        if type(dim) != tuple or len(dim) != 2:
            raise ValueError()
        res = self.juxtapose(array, dim[0], 0)
        res = self.juxtapose(res, dim[1], 1)
        return res
    
if __name__ == '__main__':
    spb = ScrapBooker()
    arr1 = np.arange(0,25).reshape(5,5)
    print(spb.crop(arr1, (3,1),(1,0)))
    arr2 = np.array("A B C D E F G H I".split() * 6).reshape(-1,9)
    print(spb.thin(arr2,3,0))
    arr3 = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
    print(spb.juxtapose(arr3, 3, 1))
    arr4 = np.array([[1, 2], [1, 2]])
    print(spb.mosaic(arr4,))
