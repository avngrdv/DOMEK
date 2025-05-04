# -*- coding: utf-8 -*-
"""
Created on Thu May 1 00:03:23 2025
@author: Alex Vinogradov
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tf.metrics import R2Score

def initialize_RFA_model(inp_dim, 
                         max_y=0, 
                         L1_reg=0,
                         LR_schedule=None,
                         optimizer=None
                         ):

    '''
    Initialize and compile a simple RFA model. This is essentially a linear
    regression model implemented in tensorflow to access GPU acceleration for
    large datasets. This tensorflow architecture is order-agnostic and can be
    used to fit first-, second- or higher-order RFA decompositions.
    
    This RFA implementation does not feature non-specific epistasis modelling
    as described in the original manuscript (Nat. Commun. 2024, 15, 7953)

    Parameters:
            inp_dim: int, the number of features in the design matrix
                     for the data to be fitted. Features must be rolled
                     out into a single vector of length (inp_dim)
                     
              max_y: float, maximum dG (activation energy) value in the dataset
                     in kcat/mol
                 
             L1_reg: float, the strength of L1 regularization lambda
              
        LR_schedule: learning rate schedule, must be an instance of 
                     tf.keras.optimizers.schedules; if None, the manuscript
                     schedule is used
                     
          optimizer: training optimizer for model fitting; must be an
                     instance of tf.keras.optimizers; if None, the manuscript
                     optimuzer is used (Adam)
                               
    Returns:
                     a compiled tensorflow model ready for fitting
    '''         
    model = Sequential([
        
        Dense(1, 
              input_shape=(inp_dim,),
              use_bias=True,
              activation=None,
              kernel_regularizer=l1(L1_reg),
),
        ReLU(max_value=max_y)
]
)
    if LR_schedule is None:
        LR_schedule = ExponentialDecay(
                                       0.001,
                                       decay_steps=5000,
                                       decay_rate=0.95,
                                       staircase=True
)
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(
                                             learning_rate=LR_schedule,
                                             beta_1=0.9,
                                             beta_2=0.98, 
                                             epsilon=1e-9
)
    
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=[R2Score()]
)
    
    return model

def parse_W_matrix(W, n_pos, n_aas):
    '''
    #TODO: generalize in the future to any RFA order
    Parse the weight matrix for a second order RFA model.

    Parameters:
                  W: model weights directly from model.get_weights()

              n_pos: peptide length (number of amino acids)
                 
              n_aas: number of amino acids in the genetic alphabet
                                
    Returns:
              e0, e1, e2: zeroth-, first-, and second-order weight matrices
    '''        
    
    #extract model weights
    e0 = W[1][0]
    w = W[0].ravel()
    
    #e1 extraction and reshaping
    e1 = w[:n_pos * n_aas].reshape((n_pos, n_aas))

    #e2 extraction and reshaping
    e2 = np.zeros((n_pos, n_pos, n_aas, n_aas))
    i, j = np.triu_indices(n_pos, k=1, m=n_pos)
    e2[i, j] = w[n_pos * n_aas:].reshape((-1, n_aas, n_aas))
    
    #symmetrize the hypercube: helpful for 
    #the e1 renormalization downstream
    #this is a weird symmetrization, we need to make sure that 
    #e2[i, j, aa1, aa2] = e2[j, i, aa2, aa1]
    e2[j, i] = np.transpose(e2[i, j], axes=(0, 2, 1))
    
    return e0, e1, e2


def weight_renormalization(W, n_pos, n_aas):
    '''
    #TODO: generalize in the future to any RFA order
    Renormalize RFA model weights. 

    Parameters:
                  W: model weights directly from model.get_weights()

              n_pos: peptide length (number of amino acids)
                 
              n_aas: number of amino acids in the genetic alphabet
                                
    Returns:
              renormalized W
    '''      
    e0, e1, e2 = parse_W_matrix(W, n_pos, n_aas)
    i, j = np.triu_indices(n_pos, k=1, m=n_pos)
    aas = np.arange(n_aas)
    
    #e0 renormalization
    s0 = e0  + e1.mean(axis=1).sum() + e2[i, j].mean(axis=(1, 2)).sum()
        
    #e1 renormalization: there has to be a vectorized way to do it
    #but w/e, it works
    s1 = np.empty(e1.shape)
    for pos1 in range(n_pos): 
        for aa in range(aas.size):
            s1[pos1, aa] = e1[pos1, aa] - e1[pos1, :].mean()
            
            t = 0
            for pos2 in range(n_pos):
                t += e2[pos1, pos2, aa].mean() - e2[pos1, pos2].mean()        
            
            s1[pos1, aa] += t

    #e2 renormalization
    mean_ij_aa1 = np.nanmean(e2, axis=2, keepdims=True)
    mean_ij_aa2 = np.nanmean(e2, axis=3, keepdims=True)
    mean_ij = np.nanmean(e2, axis=(2, 3), keepdims=True)        
    s2 = e2 - mean_ij_aa1 - mean_ij_aa2 + mean_ij

    #reassemble the weights array
    new_w = np.hstack((s1.ravel(), s2[i, j].ravel())).reshape(-1, 1)
    new_W = [new_w.astype(np.float32),
             np.array(s0.astype(np.float32)).reshape(1),
            ]
    
    return new_W


def pep_aa_contributions(pep, W, n_pos=None, n_aas=None, alphabet=None):
    '''
    #TODO: generalize in the future
    Plot first- and second-order RFA contributions to the overall peptide
    fitness for peptide 'pep'

    Parameters:
                pep: str, primary sequence for the peptide of interest
                
                  W: model weights directly from model.get_weights()
                  
              n_pos: peptide length (number of amino acids)
                 
              n_aas: number of amino acids in the genetic alphabet     

           alphabet: the overall amino acid alphabet
                           
    Returns:
              e1_contr, e2_contr: numpy arrays carrying absolute first-
                                  and second-order contributions, respectively
    '''            
    pep = list(i for i in pep if i != 'd')
    e0, e1, e2 = parse_W_matrix(W, n_pos, n_aas)
    
    e1_contr = []
    e2_contr = []
    
    for i,aa in enumerate(pep):
        aa_id = np.where(alphabet == (aa))[0][0]
        e1_contr.append(e1[i,aa_id])
        
        aa_e2 = []
        for j,aa2 in enumerate(pep):
            aa2_id = np.where(alphabet == (aa2))[0][0]
            
            #this may be controversial, but because every second-order
            #contribution comes from two amino acids, we split its overall
            #magnitude equally between both of them; this is to avoid
            #double counting second-order contributions.
            aa_e2.append(e2[i,j,aa_id,aa2_id] / 2)
            
        e2_contr.append(sum(aa_e2))
    
    return np.array(e1_contr), np.array(e2_contr)

def variance_at_pos_x(e1, e2, x, n_aas):
    '''
    Compute total RFA model variance at position x

    Parameters:
                 e1: np.array, the matrix of first-order RFA terms
                
                e2: np.array, the matrix of second-order RFA terms
           
                 x: int, amino acid position within a substrate
         
            n_aas: number of amino acids in the genetic alphabet         
    Returns:
              total position x variance
    '''         
    e1v = ((e1[x] ** 2) / n_aas).sum()
    e2v = (((e2[x] / 2) ** 2) / (n_aas ** 2)).sum()
    return e1v + e2v






