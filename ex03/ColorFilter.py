# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ColorFilter.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/19 15:29:03 by archid-           #+#    #+#              #
#    Updated: 2023/04/28 22:06:25 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np
from ImageProcessor import ImageProcessor

celluoid_n_colours = 64

# R G B colour, invert return the complementary, to apply colour filter remove the tuple
class ColorFilter:
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

    def to_celluloid(self, array):
        pass
    
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
    

    