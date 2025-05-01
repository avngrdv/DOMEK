# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:39:30 2020
@author: Alex Vinogradov
"""
import numpy as np

def get_freqs(arr, alphabet):
    '''
    Compute positional frequency of tokens in the dataset.
       
        Parameters:
                arr:    dataset as ndim=2 ndarray
           alphabet:    tokens to iterate over. should
                        have type(tokens) == arr.dtype
    
        Returns:
                frequency matrix; dims = (num_tokens, arr.shape[-1])
    '''    
    
    assert len(set(alphabet)) == len(alphabet), "Token alphabet should not \
contain duplicated tokens!"    
    
    #C: count matrix for tokens over positions in arr
    C = np.zeros((len(alphabet), arr.shape[1]))
    
    #iteratively fill it
    for i, x in enumerate(alphabet):
        C[i] = np.sum(arr == x, axis=0)
        
    with np.errstate(divide='ignore', invalid='ignore'):
        freq = np.divide(C, arr.shape[0])
        
    return freq

def get_Y_star(f1, f2, alphabet, f_out=None):
    '''
    Load two P datasets (positive/negative) and compute Y scores.
    Write Y score to a file if f_out is specified
       
        Parameters:
                f1:    full path to the positive P dataset
                f2:    full path to the negative P dataset
                f_out: full path to the output file. if left None,
                       no file will be written.
    
        Returns:
                Y score matrix; dims = (num_aa, num_pos)
    '''    

    #load positive and negative P matrices
    pos = np.load(f1).astype(str)
    neg = np.load(f2).astype(str)
    
    freq_pos = get_freqs(pos, alphabet)
    freq_neg = get_freqs(neg, alphabet)
    
    #calculate Y matrix from it and save it
    Y = np.log(np.divide(freq_pos, freq_neg))
    if f_out is not None:
        np.save(f_out, Y)
    
    return Y

def positional_conservation(freq):
    '''
    Compute position-wise sequence conservation for a dataset from it's
    token-wise frequency matrix.
    
        Parameters:
                freq:   frequency matrix; dims = (num_tokens, arr.shape[-1])
                        the product of get_freqs op above.
                            
        Returns:                  
                        conservation as a 1D vector of position-wise conservation 
                        values shape = freq.shape[-1]
    '''
    
    with np.errstate(divide='ignore', invalid='ignore'):
        E = np.nan_to_num(np.multiply(freq, np.log2(freq)))

    n = np.log2(freq.shape[0])
    return np.divide(np.sum(E, axis=0) + n, n)

def arr_purity(arr, alphabet):
    '''
    Computes average positional conservation for array arr filled with
    tokens from user-supplied alphabet. 

        Parameters:
                arr:    dataset as ndim=2 ndarray
           alphabet:    tokens to iterate over. should
                        have type(tokens) == arr.dtype
    
        Returns:
                        array purity, dtype=np.float32

    '''
    freq = get_freqs(arr, alphabet)
    conservation = positional_conservation(freq)
    return np.divide(np.sum(conservation), arr.shape[1])

def naive_clustering_score(X, labels, alphabet=None, return_mean=False):
    '''
    Compute the goodness of clustering for dataset X filled with tokens 
    from alphabet, where every entry is assigned a cluster label as 
    specified in labels array.

        Parameters:
                X:    dataset as ndim=2 ndarray
         alphabet:    tokens filling X
           labels:    array of labels assigned to entries in X (ndim=1)
                      one and only one label per entry in X
    
        Returns:
                      naive_clustering_score cl, dtype=np.float32
    '''    
    
    starting_purity = arr_purity(X, alphabet)
    n_labels = np.unique(labels)
    
    cluster_purities = np.zeros(n_labels.size)
    cluster_sizes = np.zeros(n_labels.size)
    
    #compute purities for every cluster
    for i,label in enumerate(n_labels):
        
        #everything labelled as noise, receives the score of 0 automatically
        if label == -1:
            cluster_purities[i] = 0
            cluster_sizes[i] = X[labels == label].shape[0]
            
        else:
            cluster_purities[i] = arr_purity(X[labels == label], alphabet)
            cluster_sizes[i] = X[labels == label].shape[0]
    
    #compute individual cluster scores
    weighed_added_purity = np.multiply(cluster_purities - starting_purity, 
                                       np.log2(cluster_sizes)
                                      )
    
    if return_mean:
        #the final score is the average of the individual scores
        return weighed_added_purity.mean()
    else:
        return weighed_added_purity


def hamming_distance(P, pep, 
                     h=0,
                     cum=False,
                     return_count=False, 
                     return_index=False,
                     return_distance=False):

    '''
    A flexible Hamming distance calculator.
       
        Parameters:
                P:     P dataset subject to the computation (2D np array)
                pep:   peptide to compare against (1D np array)
                       pep.dtype should be the same as P.dtype
                     
                h:     int; Hamming distance spec. The op will return a view
                       of the original P dataset where for every peptide x in
                       the resulting dataset Hamming_distance(x, pep) = h
                 
                cum:   True/False; if True, all peptides from P at a Hamming
                       distance h or less from pep will be returned
                       
       return_count:   True/False; if True, return the number of peptides in
                       P which are at Hamming_distance=h from pep
                       
       return_index:   True/False; if True, return the indices of peptides in
                       P which are at Hamming_distance=h from pep                       

    return_distance:   True/False; if True, return an array of distances between
                       peptides in P and pep
    
        Returns:
                  H:   a slice of the original P array    
    '''    

    D = P == pep
    
    if return_distance:
        return np.sum(~D, axis=1)
    
    match = pep.size - h
    if cum:
        ind = np.sum(D, axis=1) >= match
    else:
        ind = np.sum(D, axis=1) == match
        
    H = P[ind]
    
    if return_count:
        return H.shape[0]
        
    elif return_index:
        return np.where(ind)[0]

    return H

def shannon_entropy(arr, norm=True, return_counts=True):
    '''
    Compute Shannon entropy for a dataset in arr.  Note that unless 
    norm is set to True, the  resulting value scales with the dataset size.
    log2 entropy computation is used.
       
        Parameters:
                arr:   array holding data, any representation is OK
                norm:  bool; if set to True, the op will calculate 
                       "normalized entropy" (aka efficiency)
                         
        Returns:
                  (Normalized) Shannon Entropy as float32
    '''        
    
    #C - counts; n - dataset size
    C = np.unique(arr, return_counts=True, axis=0)[1]
    n = C.sum()
    normC = np.divide(C, n)
    
    #E - a vector of individual entropy values
    E = -normC * np.log2(normC)
    if norm == True:
        E = np.divide(E.sum(), np.log2(n))
    
    else:
        E = E.sum()
        
    if return_counts:
        return E, C
    else:
        return E
    
def sample_random_peptides(n, y, monomers):
    '''
    Generate an array of random peptide sequences. shape = (n_peptides, pep_len)
       
        Parameters:
                    n:     int, number of peptides to generate (n_peptides)
                    y:     int, peptide length (pep_len) 
             monomers:     list, amino acids to sample from
                         
        Returns:
                           np.ndarray shape=(n, y) filled with 
                           randomly sampled monomers
    '''   
    
    P = np.random.choice(monomers, size=(n, y), replace=True)    
    return P     
    
    
def sample_from_template(template, n, monomers):
    '''
    Generate an array of partially random peptides as specified by
    template; shape = (n_peptides, template_len)
       
        Parameters:
             template:     1D np.ndarray dtype='<U1'; the template sequence used
                           for modelling. 'X' is an amino acid used for randomization.
                           (it encodes any amino acid from the monomer set).
                           template.size is template_len. Other template amino acids
                           are not subject to randomization.
                           
                    n:     int, number of peptides to generate (n_peptides)
             monomers:     list, amino acids to sample from
                         
        Returns:
                           np.ndarray shape=(n, template.size) filled with 
                           partially randomized peptides
    '''   
    
    P = sample_random_peptides(n, len(template), monomers)
    for i,aa in enumerate(template):
        
        if aa != 'X':
            P[:,i] = [aa] * P.shape[0]
            
    return P    

def sample_from_template_improved(template, n, monomers):
    '''
    write
    '''   
    
    P = np.zeros((n, template.size), dtype='<U1')
    for pos in range(template.size):
        if not template[pos].isdigit():
            P[:,pos] = template[pos]
        else:
            P[:,pos] = np.random.choice(monomers[template[pos]], 
                                        size=n, 
                                        replace=True
                                       )    
    return P    
    
def sorted_count(arr, top_n=None, return_index=False):
    '''
    Handy utility to quickly count top_n most abundant entries in array arr.

        Parameters:
                arr:    dataset as ndarray; any dimensionality/repr is OK
              top_n:    top_n entries in arr by count to return.
                        if None, the entire array with its counts is returned.
    
       return_index:    if True, also return an array of indeces to reconstruct
                        the original array
    
        Returns:
            (arr, C):   a slice of the original array with its top_n most
                        common entries and the C array of corresponding counts
    '''
    
    if arr.ndim > 1:
        X, og_ind, C = np.unique(arr, return_counts=True, return_index=True, axis=0)
    else:
        X, og_ind, C = np.unique(arr, return_counts=True, return_index=True)
       
    ind = np.argsort(C)[::-1][:top_n]
    if return_index:
        return (X[ind], og_ind[ind], C[ind])
    
    return (X[ind], C[ind])



def get_S(P, Y):
    '''
    Compute S scores for a list of peptides P according to enrichment scores Y.
       
        Parameters:
                P:    P dataset subject to the computation (2D np array, dtype=int)
                Y:    a matrix of Y scores as calculated by get_Y (see above)
    
        Returns:
                S scores (an np ndarray vector)
                S = sum(Y scores for aa in pos x) for all x
    '''    
    
    def F(pep):
        return sum([Y[x,i] for i,x in enumerate(pep)])
        
    return np.array(list(map(F, P)))
