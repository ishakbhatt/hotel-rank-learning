# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:49:15 2021

@author: Bo Xian Ye
"""
import numpy as np

def star_onehot_encode(stars):
    """

    :param stars: 1D array
    :return: one-hot encoded star ratings
    """
    # one hot encode
    num_class = 5 #from 1 star to 5 stars
    onehot_encoded = list()
    for star in stars:
        encoded = np.zeros(num_class)
        encoded[star-1] = 1
        onehot_encoded.append(encoded)

    return np.array(onehot_encoded)
