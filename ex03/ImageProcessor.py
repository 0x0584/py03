# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ImageProcessor.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/19 10:21:27 by archid-           #+#    #+#              #
#    Updated: 2023/04/19 15:34:30 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class ImageProcessor:
    def load(self, path):
        try:
            arr = np.asarray(Image.open(path))
            print("Loading image of dimensions {} x {}".format(arr.shape[0], arr.shape[1]))
            return arr
        except Exception as e:
            print(f"Exception: {e.__class__.__name__} -- {e.strerror if hasattr(e, 'strerror') else e }")
            return None
    
    def display(self, array):
        plt.imshow(array)
        plt.show()
    
if __name__ == '__main__':
    ImageProcessor.load('foo')
    arr = ImageProcessor.load('beavers.jpg')
    ImageProcessor.display(arr)