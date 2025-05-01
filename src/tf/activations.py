# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:16:25 2020
@author: Alex Vinogradov
"""
import tensorflow as tf
from tensorflow.keras import backend as K

class CustomSigmoid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomSigmoid, self).__init__(**kwargs)
        # self.L = self.add_weight(name='L',
        #                           initializer='zeros',
        #                           trainable=True)
        
        # self.U = self.add_weight(name='U',
        #                           initializer='ones',
        #                           trainable=True)
    
    
    def call(self, inputs):
        return 5.864388972875728 / (1 + K.exp(-inputs))