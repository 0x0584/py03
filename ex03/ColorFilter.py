# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ColorFilter.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/19 15:29:03 by archid-           #+#    #+#              #
#    Updated: 2023/04/20 08:37:50 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np
from ImageProcessor import ImageProcessor

class ColorFilter:
    def invert(self, array):
        """
        Inverts the color of the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        if type(array) != np.ndarray:
            return None
        return 1 - array

    def to_blue(self, array):
        """
        Applies a blue filter to the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        if type(array) != np.ndarray:
            return None
        tmp = array.copy()
        tmp[..., :2] = np.zeros(array.shape)[..., :2]
        return tmp
    
    def to_green(self, array):
        """
        Applies a green filter to the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        if type(array) != np.ndarray:
            return None
        return array * [0, 1, 0]

    def to_red(self, array):
        """
        Applies a red filter to the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        return array - (self.to_green(array) + self.to_blue(array))

    def to_celluloid(self, array):
        """
        Applies a celluloid filter to the image received as a numpy array.
        Celluloid filter must display at least four thresholds of shades.
        Be careful! You are not asked to apply black contour on the object,
        you only have to work on the shades of your images.
        Remarks:
        celluloid filter is also known as cel-shading or toon-shading.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        pass
    
    def to_grayscale(self, array, filter, **kwargs):
        """
        Applies a grayscale filter to the image received as a numpy array.
        For filter = ’mean’/’m’: performs the mean of RBG channels.
        For filter = ’weight’/’w’: performs a weighted mean of RBG channels.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        filter: string with accepted values in [’m’,’mean’,’w’,’weight’]
        weights: [kwargs] list of 3 floats where the sum equals to 1,
        corresponding to the weights of each RBG channels.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        pass
    
if __name__ == '__main__':
    cf = ColorFilter()
    img = ImageProcessor()
    
    foo = img.load('../beavers.jpg')
    # img.display(foo)
    # img.display(cf.invert(foo))
    # img.display(cf.to_red(foo))
    # img.display(cf.to_green(foo))
    # img.display(cf.to_blue(foo))
    img.display(cf.to_celluloid(foo))
    