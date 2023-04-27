# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    NumpyCreator.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: archid- <archid-@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/18 14:17:52 by archid-           #+#    #+#              #
#    Updated: 2023/04/27 16:48:08 by archid-          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class NumpyCreator:
    def from_list(self, lst, dtype=object):
        if lst is None or type(lst) != list:
            return None
        return np.asarray(lst, dtype=dtype)
    
    def from_tuple(self, lst, dtype=object):
        if lst is None or type(lst) != tuple:
            return None
        return np.asarray(lst, dtype=dtype)
    
    def from_iterable(self, lst, dtype=object):
        try:
            return np.asarray([x for x in iter(lst)], dtype=dtype)
        except:
            return None
    
    def from_shape(self, shape, value=0, dtype=object):
        if type(shape) != tuple:
            return None
        arr = np.empty(shape, dtype=dtype)
        arr.fill(value)
        return arr

    def random(self, shape, dtype=object):
        if type(shape) != tuple:
            return None
        arr = np.empty(shape, dtype=dtype)
        for i in range(len(arr)):
            arr[i] = np.random.rand()
        return arr
    
    def identity(self, n, dtype=object):
        if type(n) != int:
            return None
        arr = self.from_shape((n, n), dtype=dtype)
        for i in range(n):
            arr[i, i] = 1
        return arr
    
if __name__ == '__main__':
    npc = NumpyCreator()
    npc.from_list([[1,2,3],[6,3,4]])
    npc.from_list([[1,2,3],[6,4]])
    npc.from_list([[1,2,3],['a','b','c'],[6,4,7]])
    npc.from_list(((1,2),(3,4)))

    npc.from_tuple(("a", "b", "c"))
    npc.from_tuple(["a", "b", "c"])

    npc.from_iterable(range(5))

    shape=(3,5)
    npc.from_shape(shape)

    npc.random(shape)

    npc.identity(4)
