import numpy as np
import pandas as pd
from numba import njit

def compute_pep_freq(P, kind='pos'):
    '''
    Take a standard P-matrix and summarize the frequencies of the
    peptides comprising it in a pandas dataframe
          
    Parameters:
            P: an NGS-derived peptide sample list
               containing peptide sequences in each row; 
               shape: (num_peptides x #longest_seq_length)
                 
          pos: str, 'pos' or 'neg'
               a string specifier to denote whether freq_pos
               or 'freq_neg' values are computed, i. e., whether
               the P dataset is derived from a positive NGS file
               or a negative one.                   
                               
    Returns:
               pandas dataframe holding peptide 
               sequences and the associated frequences
    '''
   
    n = P.shape[0]
    unique, counts = np.unique(P, axis=0, return_counts=True)
    freq = counts / n
    df = pd.DataFrame(columns=['seq', f'{kind}_freq', f'{kind}_count'])
    
    df['seq'] = [''.join(x) for x in unique]
    df[f'{kind}_freq'] = freq
    df[f'{kind}_count'] = counts
    return df

def compute_yields(f_pos, f_neg, r):
    '''
    Compute yields for each peptide in a dataframe using
    peptide frequencies and the macroscopic (library-wide) yield
           
    Parameters:
        f_pos: pd.Series instance (a dataframe column) holding
               peptide frequencies in the positive sample for
               a given experiment (c, t)
                 
        f_neg: pd.Series instance (a dataframe column) holding
               peptide frequencies in the negative sample    
               a given experiment (c, t)
                
            r: qPCR-derived library-wide yeild (recovery)
               can be passed as a pd.Series instance (a dataframe column)
                               
    Returns:
               pd.Series instance (a dataframe column) holding
               computed yields
    ''' 
    p = r * f_pos
    n = (1 - r) * f_neg
    return p / (p + n)    

def compute_yield_SEMS_v3(pos_freq, 
                          neg_freq, 
                          n_pos, 
                          n_neg, 
                          r, 
                          dr
):   
    '''
    Compute yield SEMs by first computing the sampling errors associated
    with the peptide frequency estimates, and then by propagating 
    the resulting errors.
    
    Works either for individual numbers or for entire datasets (arrays)
           
    Parameters:
        
         pos_freq: pd.Series instance (a dataframe column) holding
                   peptide frequencies in the positive sample for
                   a given experiment (c, t)
                 
         neg_freq: pd.Series instance (a dataframe column) holding
                   peptide frequencies in the negative sample    
                   a given experiment (c, t)
                
            n_pos: total number of reads in the positive (c, t) NGS sample
            
            n_neg: total number of reads in the positive (c, t) NGS sample           
                
                r: qPCR-derived library-wide yeild (recovery)
                   can be passed as a pd.Series instance (a dataframe column)
               
               dr: r value SEMs (experimentally measured by qPCR) 
                               
    Returns:
               pd.Series instance (a dataframe column) holding
               computed yields
    ''' 
    
    def AC_se(f, n):
        return np.sqrt(f * n + 0.5) / n
    
    #Calculate the SEMS for positive and negative samples
    #We use the Agresti-Coull (or the pseudo-count) approximation here
    se_f_pos = AC_se(pos_freq, n_pos)
    se_f_neg = AC_se(neg_freq, n_neg)
    
    denominator = r * pos_freq  + (1 - r) * neg_freq
    term_1 = neg_freq * (1 - r) * r *  se_f_pos
    term_2 = pos_freq * (1 - r) * r *  se_f_neg
    term_3 = pos_freq * neg_freq * dr

    return np.sqrt(np.power(term_1, 2) + 
                   np.power(term_2, 2) + 
                   np.power(term_3, 2)) / np.power(denominator, 2)

@njit
def tQSSA_2param(x, Km, k2):
    '''
    2-parameter tQSSA model. Valid when 
    (sub << Km) and (sub << enz)
           
    Parameters:                   
            x: gridpoint (c, t) values
               a 2D numpy array storing time values in column 1
               and enzyme concentrations in column 2   
               
           k2: model parameter; corresponds to kcat
           Km: model parameter; corresponds to Km
    Returns:
               product yield at (c, t)
    '''      
    t = x[:,0]
    E = x[:,1]
    
    exp = -k2 * t * E / (E + Km)
    return 1 - np.exp(exp)

@njit
def tQSSA_1param(x, specificity):
    '''
    1 parameter tQSSA model. Valid when
    (enz << Km) and (sub << Km).
           
    Parameters:                   
            x: gridpoint (c, t) values           
  specificity: model parameter; corresponds to kcat/Km
                    
    Returns:
               product yield at (c, t)
    '''     
    t = x[:,0]
    E = x[:,1]
    
    exp = -specificity * t * E
    return 1 - np.exp(exp)

def calculate_r2_adj(func, y, x, popt):
    '''
    Calculate "adjusted R-squared" as a metric for
    the goodness of fit. The adjusted version is calculated
    to account for the discrepancy in the number of fitted
    parameters in the tQSSA 1-param and tQSSA 2-param models. 
           
    Parameters:
        
         func: python function corresponding to the model used
               for fitting
            y: observations; calculated yields at (c, t) in our case                
            x: (c, t) values           
         popt: fitted model (func) parameters
                    
    Returns:
               adjusted R squared values
    '''     
    
    n = y.size
    p = popt.size
    
    res = y - func(x.T, *popt)
    ss_res = np.sum(res**2) / (n - p - 1) 
    ss_tot = np.sum((y - np.mean(y))**2) / (n - 1)
    return 1 - (ss_res / ss_tot)

def get_t_critical(n, alpha=0.05):
    '''
    Calculate the t-critical value at alpha to determine 
    the confidence intervals for the fitted parameters.
    
    Parameters:
        
            n: number of data points used for the fitting                
        alpha: the significance level

    Returns:
               t-critical value at alpha     
    '''
    from scipy.stats import t
    dof = n - 1
    return t.ppf(1 - alpha/2, dof)

def kcatKm_to_dG(kcatKm, T=298, in_kcal=True):
    '''
    Convert kcat/Km values to activation energies according
    to the Eyring equation, assuming the transmission
    coefficient of 1. 
    
    Parameters:
        
       kcatKm: kcat/Km; either a singular value or array-like (M-1s-1)
            T: temperature in K
      in_kcal: True/False; if True, return the energy values
               in kcal/mol; otherwise J/mol are returned

    Returns:
               Activation energy values
    
    '''
    kb = 1.380e-23 #Boltzmann constant
    h = 6.626e-34 #Planck constant
    R = 8.314 #Universal gas constant
    
    Ea = R * T * np.log((kb * T) / (kcatKm * h))
    if in_kcal:
        return Ea / 4184
    
    return Ea

def dG_to_kcatKM(dG, T=298):
    '''
    Convert activation energy values to kcat/Km values according
    to the Eyring equation, assuming the transmission coefficient of 1. 
    
    Parameters:
        
            dG: activation ernergy, in kcal/mol
             T: temperature in K
             
    Returns:
              kcat/Km in M-1s-1
    '''
    
    kb = 1.380e-23 #Boltzmann constant
    h = 6.626e-34 #Planck constant
    R = 8.314 #Universal gas constant
    
    kcatKm = (kb * T / h) * np.exp(-dG * 4184 / R / T)   
    return kcatKm










