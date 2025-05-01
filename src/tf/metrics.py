# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:16:25 2020
@author: Alex Vinogradov
"""
import tensorflow as tf
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name='ssr', initializer='zeros')
        self.sst = self.add_weight(name='sst', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        #Residual Sum of Squares (SSR)
        residuals = y_true - y_pred
        ssr = tf.reduce_sum(tf.square(residuals))

        #Total Sum of Squares (SST)
        mean_y_true = tf.reduce_mean(y_true)
        sst = tf.reduce_sum(tf.square(y_true - mean_y_true))

        #Update state
        self.ssr.assign_add(ssr)
        self.sst.assign_add(sst)

    def result(self):
        return 1 - (self.ssr / (self.sst + tf.keras.backend.epsilon()))  # Add epsilon to avoid division by zero

    def reset_states(self):
        self.ssr.assign(0)
        self.sst.assign(0)