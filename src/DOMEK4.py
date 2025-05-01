import os, gc
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import warnings, re
warnings.simplefilter(action='ignore')

import numpy as np
np.set_printoptions(suppress=True, precision=3)

import pandas as pd
from numba import njit
from copy import deepcopy

#import prerequisities
from clibas.parsers import FastqParser
from clibas.pipelines import Pipeline
from clibas.dataanalysis import DataAnalysisTools
from clibas.dispatchers import Dispatcher
from clibas.datapreprocessors import DataPreprocessor
from clibas.datatypes import AnalysisSample, Data

handlers = (Pipeline, FastqParser, DataAnalysisTools, DataPreprocessor)
pip, par, dta, pre = Dispatcher.dispatch('POPEK4_config.yaml', handlers)

# pip.enque([
#             par.trim_reads(left='ATGAGT', right='CGGAAA'),
#             par.translate(force_at_frame=0, stop_readthrough=False),
#             dta.length_analysis(where='pep', save_txt=True),
#             par.len_filter(where='pep'),
#             par.cr_filter(where='pep', loc=[0, 2], tol=5),
#             par.vr_filter(where='pep', loc=[1], sets=[1, 2, 3]),
#             dta.q_score_analysis(save_txt=True),
#             par.q_score_filt(minQ=23, loc=[1]),
#             par.fetch_at(where='pep', loc=[1]),
#             par.filt_ambiguous(where='pep'),
#             par.count_summary(where='pep', top_n=5000, fmt='csv'),
#             par.unpad(),
#             par.save(where='pep', fmt='npy')
#           ])

# pip.stream(generator=par.stream_from_gz_dir(), 
#            save_summary=True
# )

def fetch_npy_files():
    import shutil
  
    source_folder = '../parser_outputs'
    destination = '../NGS sample npys'
    everything = list(os.walk(source_folder))
    
    for i in everything:
        for file in i[2]:
            print(file)
            if file.endswith('.npy'):
                
                shutil.copy(os.path.join(i[0], file), 
                            os.path.join(destination, file))

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
    peptide frequencies and the library-wide yield
           
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
    with the estimation of peptide frequencies, and then by propagating
    the errors.
    
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
    2 parameter tQSSA model. Valid when
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
    Calculate the t-critical value at alpha to determine the confidence
    intervals for the fitted parameters.
    
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
        
       kcatKm: kcat/Km; either a singular value or array-like  
            T: temperature in K
      in_kcal: True/False; if True, return the energy values
               in kcal/mol; otherwise J/mol are returned

    Returns:
               Activation energy values
    
    '''
    kb = 1.380e-23
    h = 6.626e-34
    R = 8.314 
    
    Ea = R * T * np.log((kb * T) / (kcatKm * h))
    if in_kcal:
        return Ea / 4184
    
    return Ea

def dG_to_kcatKM(dG, T=298):
    # '''
    # Convert kcat/Km values to activation energies according
    # to the Eyring equation, assuming the transmission
    # coefficient of 1. 
    
    # Parameters:
        
    #    kcatKm: kcat/Km; either a singular value or array-like  
    #         T: temperature in K
    #   in_kcal: True/False; if True, return the energy values
    #            in kcal/mol; otherwise J/mol are returned

    # Returns:
    #            Activation energy values
    
    # '''
    kb = 1.380e-23
    h = 6.626e-34
    R = 8.314 
    
    kcatKm = (kb * T / h) * np.exp(-dG * 4184 / R / T)   
    return kcatKm

'''move parser outputs to npy file folder'''
#fetch_npy_files()
C_DICT = {8: '800', 0.8: '080', 0.08: '008', 0: '000'}
EPSILON = 1e-8 #small non-zero value to avoid infinities

'''process the individual data points'''
# meta = pd.read_csv(os.path.join('../POPEK4_library-wide.csv'))
# for row in meta.iterrows():
#     exp = row[1]
#     t = int(exp['t'])
#     c = exp['c']
#     r = exp['r'] 
#     dr = exp['dr']
    
#     try:
#         pos = np.load(os.path.join('../NGS sample npys', f'p4_c{C_DICT[c]}_t{t}_FT_m_pep.npy'))
#         neg = np.load(os.path.join('../NGS sample npys', f'p4_c{C_DICT[c]}_t{t}_E_m_pep.npy'))
#     except:
#         raise
        
#     print(f'Working on the c={C_DICT[c]} c=t{t} sample. . .')
        
#     #get the individual pos and neg frequency lists
#     df_pos = compute_pep_freq(pos, kind='pos')
#     df_neg = compute_pep_freq(neg, kind='neg')
    
#     #merge them
#     df_m = pd.merge(df_pos, df_neg, on='seq', how='outer')
#     df_m = df_m.sort_values('seq', ascending=True)
    
#     #deal with nans, just fill missing entries with zeros
#     df_m = df_m.fillna(0)

#     df_m['t'] = t
#     df_m['c'] = c
#     df_m['r'] = r
#     df_m['dr'] = dr
    
#     #calculate the yields
#     df_m['y'] = compute_yields(df_m['pos_freq'],
#                                 df_m['neg_freq'],
#                                 df_m['r']
# )       
#     #calculate the yield errors   
#     df_m['y_err'] = compute_yield_SEMS_v3(df_m['pos_freq'], 
#                                           df_m['neg_freq'], 
#                                           df_m['pos_count'].sum(), 
#                                           df_m['neg_count'].sum(), 
#                                           df_m['r'],
#                                           df_m['dr'],
# )
#     #save these intermediate results; this section is time consuming
#     df_m.to_csv(os.path.join('../sample dataframes',
#                               f'p4_c{C_DICT[c]}_t{t}_summary.csv'
#                             ), index=False
# )

'''truncate to a list of sequences which have reads in every sample'''
# parent = '../sample dataframes'
# DFs = []
# fnames = []

# #get a list of datasets (samples) to concatenate
# for f in os.listdir(parent):
#     full_path = os.path.join(parent, f)
#     if os.path.isfile(full_path) and full_path.endswith('_summary.csv'):
#         DFs.append(pd.read_csv(full_path)) 
#         fnames.append(full_path)

# #initialize the so-called master dataframe 
# #that is the merger between between the individual datasets
# time = int(re.search(r't(\d+)', fnames[0])[0][1:])
# conc = re.search(r'c(\d+)', fnames[0])[0][1:]   
# suffix = f'_c{conc}_t{time}'

# master = DFs[0][['seq', 'pos_count', 'neg_count']]
# master['count_sum'] = master['pos_count'] + master['neg_count']
# master.rename(columns={'pos_count': 'pos_count' + suffix,
#                         'neg_count': 'neg_count' + suffix,
#                         'count_sum': 'count_sum' + suffix
#                       }, inplace=True
# )

# #merge the read count information
# for df, fname in zip(DFs[1:], fnames[1:]):
    
#     time = int(re.search(r't(\d+)', fname)[0][1:])
#     conc = re.search(r'c(\d+)', fname)[0][1:]   
#     suffix = f'_c{conc}_t{time}'
    
#     trunc_df = df[['seq', 'pos_count', 'neg_count']]
#     trunc_df['count_sum'] = trunc_df['pos_count'] + trunc_df['neg_count']    
#     trunc_df.rename(columns={'pos_count': 'pos_count' + suffix,
#                               'neg_count': 'neg_count' + suffix,
#                               'count_sum': 'count_sum' + suffix,
#                             }, inplace=True
# )    
#     master = pd.merge(master, 
#                       trunc_df,
#                       on='seq', 
#                       how='outer',
#                       suffixes=(None, None)
# )

# #deal with nans, just fill missing entries with zeros
# master = master.fillna(0)

# #drop a sequence if any of the 'count_sum' values is zero
# #in other words, only keep the peptides which have at least
# #one read in all (c,t) NGS samples
# col = [c for c in master.columns if c.startswith('count_sum')]
# mask = (master[col] >= 1).all(axis=1)

# master = master[mask]
# master.to_csv(os.path.join(parent, 'POPEK4_master_list_pep.csv'), index=False)

# #go through the individual files and make the subsets
# #containing only master sequences
# for df, fname in zip(DFs, fnames):
    
#     filtered_df = df[df['seq'].isin(master['seq'])]
#     name = os.path.split(fname)[-1].split('.')[0] 
#     filtered_df.to_csv(os.path.join(parent, f'{name}_only_master_seq.csv'), index=False)

'''introduce corrections, compute sampling errors'''    
# parent = '../sample dataframes'
# meta = pd.read_csv(os.path.join('../POPEK4_library-wide.csv'))

# #1. compute f_neg correction factors
# d_min = pd.read_csv(os.path.join(parent, 'p4_c000_t0_summary_only_master_seq.csv'))
# d_max = pd.read_csv(os.path.join(parent, 'p4_c800_t540_summary_only_master_seq.csv'))
    
# #the condition specifying the minimal observed neg_freq to consider
# ind = (d_max['y'] > 0.95) & (d_max['y_err'] < 0.10)

# q = (d_max[ind]['neg_freq'] * (1 - d_max[ind]['r']) / (d_min[ind]['neg_freq'] * (1 - d_min[ind]['r'])))
# q = q[q < 0.5].mean()
   
# neg_freq_r_corrections = q * d_min['neg_freq'] * (1 - d_min['r'])

# #fetch dataframe names to correct
# DFs = []
# fnames = []
# for f in os.listdir(parent):
#     full_path = os.path.join(parent, f)
#     if os.path.isfile(full_path) and full_path.endswith('_summary_only_master_seq.csv'):
#         DFs.append(pd.read_csv(full_path)) 
#         fnames.append(full_path)

# for df, fname in zip(DFs, fnames):

#     #1. correct the negative freqs
#     df['neg_freq_corr'] = (df['neg_freq'] * (1 - df['r']) - neg_freq_r_corrections)/(1 - df['r'] - q * (1 - d_min['r']))
#     #make sure we don't have negative or zero frequencies
#     df = df.fillna(EPSILON)
#     df['neg_freq_corr'][df['neg_freq_corr'] <= EPSILON ] = EPSILON
     
#     #2. correct positive freqs (3-point correction)
#     time = int(re.search(r't(\d+)', fname)[0][1:])
#     conc = re.search(r'c(\d+)', fname)[0][1:]
    
#     d_c_0 = pd.read_csv(os.path.join(parent, f'p4_c{conc}_t0_summary_only_master_seq.csv'))
#     d_0_0 = pd.read_csv(os.path.join(parent, 'p4_c000_t0_summary_only_master_seq.csv'))
#     d_0_t = pd.read_csv(os.path.join(parent, f'p4_c000_t{time}_summary_only_master_seq.csv'))

#     df['pos_freq_corr'] = (df['pos_freq'] * df['r'] - d_c_0['pos_freq'] * d_c_0['r'] -
#                            d_0_t['pos_freq'] * d_0_t['r'] + d_0_0['pos_freq'] * d_0_0['r']) / (df['r'] - d_c_0['r'] - d_0_t['r'] + d_0_0['r'])

#     #make sure we don't have negative or zero frequencies
#     df = df.fillna(EPSILON)
#     df['pos_freq_corr'][df['pos_freq_corr'] <= EPSILON ] = EPSILON
      
#     #3. recalculate the yields and errors
#     #calculate corrected yields
#     df['y_corr'] = compute_yields(df['pos_freq_corr'],
#                                   df['neg_freq_corr'],
#                                   df['r']
# )       
    
#     #calculate corrected yield sems   
#     df['y_err_corr'] = compute_yield_SEMS_v3(df['pos_freq_corr'], 
#                                              df['neg_freq_corr'], 
#                                              df['pos_count'].sum(), 
#                                              df['neg_count'].sum(), 
#                                              df['r'],
#                                              df['dr'],
# )
#     df = df.fillna(0)
#     name = os.path.split(fname)[-1].split('.')[0] 
#     df.to_csv(os.path.join(parent, 'corrected', f'{name}_corrected.csv'), index=False)


'''do the fitting, peptide by peptide'''
# MIN_ERR = 0.01

# from scipy.optimize import curve_fit
# parent = os.path.join('../sample dataframes', 'corrected')
# DFs = []

# #since these are seq_filtered, the sequence lists are identical in every file
# pepos = list(pd.read_csv(os.path.join(
#                           parent, 'p4_c000_t0_summary_only_master_seq_corrected.csv')
#                         )['seq'])

# for f in os.listdir(parent):
#     full_path = os.path.join(parent, f)
#     if os.path.isfile(full_path) and full_path.endswith('_summary_only_master_seq_corrected.csv'):
        
#         #(0, t) data are only used for referencing, 
#         #no use for fitting, so these files are not fetched
#         if not'_c000_' in full_path and f.startswith('p4'):
#             DFs.append(pd.read_csv(full_path)) 
            
# fits = []
# counter = 0

# #before we start the fitting, let's find t-critical value to compute
# #confidence intervals later. 
# z = get_t_critical(len(DFs), alpha=0.05)

# for pepo in pepos:
    
#     #assemble a dataframe containing the relevant data for peptide 'pepo'
#     pepo_data = []
#     for df in DFs:
#         pepo_data.append(df[df['seq'] == pepo])
#     df_pepo = pd.concat(pepo_data)

#     #prepare the data for fitting
#     x = np.vstack((np.asarray(df_pepo['t']) * 60, 
#                     np.asarray(df_pepo['c']) * 1e-6)
# )   
#     y = np.asarray(df_pepo['y_corr'])
 
#     sigma = deepcopy(np.asarray(df_pepo['y_err_corr']))
#     sigma[sigma < MIN_ERR] = MIN_ERR
    
#     #first, fit to a 2-param tQSSA
#     popt, pcov = curve_fit(tQSSA_2param, 
#                             x.T,
#                             y,
#                             bounds=(0, np.inf),
#                             maxfev=5000,
#                             sigma=sigma,
#                             absolute_sigma=True
# )
    
#     perr = np.sqrt(np.diag(pcov))
#     r2 = calculate_r2_adj(tQSSA_2param, y, x, popt)
    
#     df_pepo['Km'] = popt[0]
#     df_pepo['Km_err'] = perr[0]
#     df_pepo['Km_CI'] = perr[0] * z
#     df_pepo['kcat'] = popt[1]
#     df_pepo['kcat_err'] = perr[1]
#     df_pepo['kcat_CI'] = perr[1] * z 
#     df_pepo['kcat/Km_2p'] = popt[1]/popt[0]
#     df_pepo['2p R_2_adj'] = r2

#     #now run the 1-param fitting
#     popt, pcov = curve_fit(tQSSA_1param, 
#                             x.T,
#                             y,
#                             bounds=(0, np.inf),
#                             maxfev=5000,
#                             sigma=sigma,
#                             absolute_sigma=True
# )    
    
#     perr = np.sqrt(np.diag(pcov))
#     r2 = calculate_r2_adj(tQSSA_1param, y, x, popt)

#     df_pepo['kcat/Km_1p'] = popt[0] 
#     df_pepo['kcat/Km_1p_err'] = perr[0]
#     df_pepo['kcat/Km_1p_CI'] = perr[0] * z
#     df_pepo['1p R_2_adj'] = r2
    
#     fits.append(df_pepo.copy(deep=True))

#     counter += 1    
#     print(np.random.randint(0, 80, size=1)[0] * ' ' + 'pepo!')    
#     if counter % 1000 == 0:
#         print(f'Processed {counter} pepos. . .')

# #concatenate the dataframes and write to a file
# fits = pd.concat(fits)
# fits.to_csv(os.path.join('../kinetics fitting', 
#                           '2025.01.24__p4_master_kinetic_summary.csv'
#                         ), index=False
# )    

# #save also a compressed version of the fitting summary
# cols = ['Km', 'Km_err', 'Km_CI', 'kcat', 'kcat_err', 'kcat_CI',
#         'kcat/Km_2p', '2p R_2_adj', 'kcat/Km_1p', 'kcat/Km_1p_err',
#         'kcat/Km_1p_CI', '1p R_2_adj']

# S = fits.groupby(by='seq', as_index=False).agg({'pos_count': 'sum', 'neg_count': 'sum'})
# S['tot_count'] = S['pos_count'] + S['neg_count']

# for c in cols:
#     S[c] = fits.groupby(by='seq', as_index=False).agg({c: 'mean'})[c]
    
# S.to_csv(os.path.join('../kinetics fitting', 
#                       '2025.01.24__p4_succint_kinetic_summary.csv'
#                       ), index=False
# )  



'''IMPORTANT APPLICATION SPECIFIC CONSTANTS'''
x_ind = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
alphabet = np.array(('A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                      'N', 'P', 'R', 'S', 'T', 'V', 'W', 'Y', 'a'))

N_POS = len(x_ind)
N_AAS = len(alphabet)

#prepare the data
S = pd.read_csv(os.path.join('../kinetics fitting', 
                              '2025.01.24__p4_succint_kinetic_summary.csv')
)

'''IMPORTANT METAPARAMS'''
MIN_KK = 0.5
MAX_KK = 30000
MIN_READS = 150
MAX_ERR = 0.30
MIN_R2 = 0.85

#get rid of pepos with less than MIN_READS reads
S = S[S['tot_count'] > MIN_READS]

#make sure the error is low or we are outside of the prediction range
ind1 = S['kcat/Km_1p_err']/S['kcat/Km_1p'] < MAX_ERR
ind2 = S['kcat/Km_1p'] < MIN_KK
ind3 = S['kcat/Km_1p'] > MAX_KK
S = S[ind1 | ind2 | ind3]

#make sure that the R_sq is reasonable
ind1 = S['1p R_2_adj'] > MIN_R2
ind2 = S['kcat/Km_1p'] < MIN_KK
ind3 = S['kcat/Km_1p'] > MAX_KK
S = S[ind1 | ind2 | ind3]

#make sure no dxxH motifs are used for training
S = S[~S['seq'].apply(lambda x: x[11] == 'H')]

#flatten kcat/Km values
S['kcat/Km_1p'][S['kcat/Km_1p'] < MIN_KK] = MIN_KK
S['kcat/Km_1p'][S['kcat/Km_1p'] > MAX_KK] = MAX_KK

#this an empirical exclusion: some peptides which appears to be modified
#well, e.g., non-enzymatically, are excluded from further analysis
S = S[~((S['pos_count'] / S['neg_count'] > 2) & (S['kcat/Km_1p'] < 1))]

#center the data
y = kcatKm_to_dG(S['kcat/Km_1p'])
print('y offest: ', y.min())
y = y - y.min()

data = Data([AnalysisSample(X=S['seq'], 
                            y=y,
                            seq=S['seq'])
])

#awkward, but w/e for now
data[0].X = data[0].X[:,x_ind]

pip.enque([         
            pre.int_repr(alphabet=alphabet),
            pre.shuffle(),
            pre.sample(sample_size=200),
            pre.tt_split(test_fraction=0.01),
            pre.featurize_for_RFA(alphabet=alphabet, order='second')
          ])

data = pip.run(data)

#prep the data
X_train = data[0].X
y_train = data[0].y

X_test = data[1].X
y_test = data[1].y

inp_dim = X_train.shape[-1]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tf.metrics import R2Score

model = Sequential([
    
    Dense(1, 
          input_shape=(inp_dim,),
          use_bias=True,
          activation=None,
          kernel_regularizer=l1(0.00008),
          ),

    ReLU(max_value=y.max())
])

LR_schedule = ExponentialDecay(
                                0.001,
                                decay_steps=5000,
                                decay_rate=0.95,
                                staircase=True
)

# opt = tf.keras.optimizers.Adam(
#                                 learning_rate=LR_schedule,
#                                 beta_1=0.9,
#                                 beta_2=0.98, 
#                                 epsilon=1e-9
# )

# model.compile(optimizer=opt,
#               loss='mse',
#               metrics=[R2Score()]
# )

# from tf.callbacks import EarlyStop, Checkpoint
# save_dir = os.path.join('../tf_trained_models', 'checkpoints')
# model_name = '2025.01.26' + '_chkpt_{epoch:02d}.h5'
# filepath = os.path.join(save_dir, model_name)

# callbacks = [
#               EarlyStop(patience=12), 
#               Checkpoint(filepath=filepath, save_best_only=True),
# ]

# training_log = model.fit(
#                           x=X_train,
#                           y=y_train,
#                           batch_size=128,
#                           epochs=1024,            
#                           validation_data=(X_test, y_test),
#                           callbacks=callbacks,
#                           verbose=1
# )

def parse_W_matrix(W, n_pos, n_aas):

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
    
    return e0, e1, e2 #, L, U

def weight_renormalization(W, n_pos, n_aas):

    e0, e1, e2 = parse_W_matrix(W, n_pos, n_aas)
    i, j = np.triu_indices(n_pos, k=1, m=n_pos)
    aas = np.arange(n_aas)
    
    #e0 renormalization
    s0 = e0  + e1.mean(axis=1).sum() + e2[i, j].mean(axis=(1, 2)).sum()
        
    #e1 renormalization: there has to be a vectorized way to do it, but w/e, it works
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

#wrong
def variance_by_order(W, n_pos, n_aas):

    e0, e1, e2 = parse_W_matrix(W, n_pos, n_aas)
    i, j = np.triu_indices(n_pos, k=1, m=n_pos)
    
    e1_var = e1.var()
    e2_var = e2[i, j].var()
    t = e1_var + e2_var
    
    return e1_var / t, e2_var / t

# def top_N_features(pepo, W, N=None, n_pos=None, n_aas=None):
      
#     #pepo must be fully featurized  
#     components = (W[0].ravel() * pepo)
#     total = components.sum()
#     top_N = np.argsort(np.abs(components))[::-1][:N]   
    
#     e0, e1, e2, L, U = parse_W_matrix(W, n_pos, n_aas)
    
#     e1_ind = []
#     e2_ind = []
#     e1_contributions = []
#     e2_contributions = []
#     for ind in top_N:
#         if ind <  n_pos * n_aas:
#             e1_ind.append(ind)
#             e1_contributions.append((components[ind], components[ind]/total))
#         else:
#             e2_ind.append(ind)
#             e2_contributions.append((components[ind], components[ind]/total))

#     e1_ind = np.array(e1_ind)
#     e2_ind = np.array(e2_ind) - n_pos * n_aas
    
#     #e1 component are reported as triplets:
#     #(pos, aa, fitness component)
#     e1_comp = [np.unravel_index(e1_ind[x], e1.shape) + (e1_contributions[x],)
#                         for x in range(len(e1_contributions))
#                        ]

#     #e2 components are reported as pentaplets:
#     #(pos1, pos2, aa1, aa2, fitness component)
#     e2_comp = list()
#     i, j = np.triu_indices(n_pos, k=1, m=n_pos)
#     for x in range(len(e2_contributions)):
        
#         ind = np.unravel_index(e2_ind[x], e2[i, j].shape)
#         coord = (i[ind[0]], j[ind[0]], ind[1], ind[2])
#         e2_comp.append((coord) + (e2_contributions[x],))
        
    
#     return e1_comp, e2_comp

def contribution_by_order(pepo, W, n_pos=None, n_aas=None):
    
    components = (W[0].ravel() * pepo)
    e1_contribution = components[:n_pos * n_aas].sum()
    e2_contribution = components[n_pos * n_aas:].sum()

    return e1_contribution, e2_contribution

def e1_tensor(e1, alphabet, basename):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 10), dpi=300)

    # scale = np.max([np.abs(np.nanmin(e1)), np.abs(np.nanmax(e1))]) + 0.05
    scale = 3
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)
    
    import seaborn as sns
    cmap = sns.diverging_palette(254, 2.9, s=90, l=43, as_cmap=True)

    c = ax.pcolor(e1, cmap=cmap, norm=norm,
                  edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Amino acid ddG contributions", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    
    
    for y in range(e1.shape[0]):
        for x in range(e1.shape[1]):
            
            if np.abs(e1[y, x]) >= 3:
                c = '#f2f2f2'                
            else:
                c = '#323232'
            plt.text(x + 0.5, y + 0.5, '%.2f' % e1[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     c=c,
                     fontdict={'fontname': 'Arial'}
                     
                 )

    #set ticks
    ax.set_xticks(np.arange(e1.shape[1])+0.5)
    ax.set_yticks(np.arange(e1.shape[0])+0.5)   
    ax.set_xticklabels(np.arange(e1.shape[1])+1)
    ax.set_yticklabels(alphabet)

    #set labels
    ax.set_xlabel('Position inside the insert', fontsize=25)
    ax.set_ylabel('Amino acid', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_title('e1 contribution matrix', fontsize=27)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return    

def e2_tensor_slice(e2, pos1, pos2, alphabet, basename):
    '''
    A 2D map of epistatic interactions between amino acids in pos1 and pos2
    Used to make Fig. S7

        Parameters:
                    epi:   4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library
                           
             pos1, pos2:   int
                           
    '''   
    import matplotlib as mpl
    import matplotlib.pyplot as plt     
    
    e2 = e2[pos1, pos2]
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)
    
    import seaborn as sns
    cmap = sns.diverging_palette(254, 2.9, s=90, l=43, as_cmap=True)
    
    # scale = np.max([np.abs(np.nanmin(e2)), np.abs(np.nanmax(e2))]) + 0.02
    scale = 1
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)
    
    c = ax.pcolor(e2, cmap=cmap, norm=norm, edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Epistatic ddG contributions", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    
    
    for y in range(e2.shape[0]):
        for x in range(e2.shape[1]):
            
            if np.abs(e2[y, x]) >= 0.95:
                c = '#f2f2f2'                
            else:
                c = '#323232'    
                
            plt.text(x + 0.5, y + 0.5, '%.2f' % e2[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     c=c,
                     fontdict={'fontname': 'Arial'}
                 )

    #set ticks
    ax.set_xticks(np.arange(e2.shape[1])+0.5)
    ax.set_yticks(np.arange(e2.shape[0])+0.5)
    ax.set_xticklabels(alphabet)
    ax.set_yticklabels(alphabet)

    #set labels
    x_label = 'Amino acid in position ' + str(pos2+1)
    ax.set_xlabel(x_label, fontsize=25)
    
    y_label = 'Amino acid in position ' + str(pos1+1)
    ax.set_ylabel(y_label, fontsize=25)
    
    ax.tick_params(axis='both', which='major', labelsize=21)
    title = 'Epistasis between positions ' + str(pos1+1) + ' and ' + str(pos2+1)
    ax.set_title(title, fontsize=27)
    
    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight')
    plt.close()  

def epistasis_bw_positions(epi, basename):
    '''
    Reduce epi array to (seq_len, seq_len) by averaging along the last 
    two axes and plot the the result.
    Used to make Fig. 3c and 5f

        Parameters:
                    epi:   4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library                   
    '''   
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    epi = np.nanmean(np.abs(epi), axis=(2,3))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300) 
    norm = mpl.colors.Normalize(vmin=0, vmax=0.22)
    import seaborn as sns
    cmap = sns.light_palette("#1571da", as_cmap=True)
    c = ax.pcolor(epi, cmap=cmap, norm=norm, edgecolors='w', linewidths=4)    
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("abs(ddG), kcal/mol", rotation=-90, va="bottom", fontsize=21)
    cbar.ax.tick_params(labelsize=21)    

    for y in range(epi.shape[0]):
        for x in range(epi.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % epi[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     c='#323232',
                     fontdict={'fontname': 'Arial'}
                 )

    #set ticks
    ax.set_xticks(np.arange(epi.shape[1])+0.5)
    ax.set_yticks(np.arange(epi.shape[0])+0.5)
    ax.set_xticklabels(np.arange(epi.shape[0])+1)
    ax.set_yticklabels(np.arange(epi.shape[0])+1)

    #set labels
    ax.set_xlabel('Variable region position', fontsize=21)
    ax.set_ylabel('Variable region position', fontsize=21)
    
    ax.tick_params(axis='both', which='major', labelsize=21)
    title = 'Average positional second order contributions'
    ax.set_title(title, fontsize=23)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return
    

'''renormalize/enforce zero-mean'''
# W = model.get_weights()
# model.set_weights(weight_renormalization(W, N_POS, N_AAS))
# q = np.corrcoef((model.predict(data[1].X).ravel(), data[1].y))**2
# model.save_weights('../tf_trained_models/RFA_20250124_zero_mean.h5')

'''get the e1/e2 plots'''
# model.load_weights('../tf_trained_models/RFA_20250124_zero_mean.h5') 
# W = model.get_weights()
# e0, e1, e2 = parse_W_matrix(W, N_POS, N_AAS)

# idx = [
#         (1, 2),
#         (6, 7),
#         (7, 8),
#         (7, 9),
#         (7, 10),
#         (8, 9),   
#         (8, 10),  
#         (9, 10),    
#         (9, 11),
#         (10, 12),
#         (1, 16),
#         (9, 13),
#       ]

# np.savetxt('../model npys/RFA_20250124_zero_mean_e1.csv', e1.T, delimiter=',')
# e1_tensor(e1.T, alphabet, '../model npys/RFA_20250124_zero_mean_e1')

# for i in idx:
#     np.savetxt(f'../model npys/RFA_20250124_zero_mean_e2_{i[0]}_{i[1]}.csv', 
#                 e2[i[0]-1, i[1]-1], delimiter=',')

#     e2_tensor_slice(e2, i[0]-1, i[1]-1, alphabet,
#                     f'../model npys/RFA_20250124_zero_mean_e2_{i[0]}_{i[1]}')

# epistasis_bw_positions(e2, '../model npys/RFA_20250124_zero_mean_epi_bw_pos')

def pepo_aa_contributions(pepo, W):
      
    #pepo must be a string  
    pepo = list(i for i in pepo if i != 'd')
    e0, e1, e2 = parse_W_matrix(W, N_POS, N_AAS)
    
    e1_contr = []
    e2_contr = []
    for i,aa in enumerate(pepo):
        aa_id = np.where(alphabet == (aa))[0][0]
        e1_contr.append(e1[i,aa_id])
        
        aa_e2 = []
        for j,aa2 in enumerate(pepo):
            aa2_id = np.where(alphabet == (aa2))[0][0]
            aa_e2.append(e2[i,j,aa_id,aa2_id] / 2)
            
        e2_contr.append(sum(aa_e2))
    
    return np.array(e1_contr), np.array(e2_contr)

def aa_contributions(pepo, e1_contr, e2_contr, alphabet, basename):
    
    import matplotlib.pyplot as plt    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6), dpi=300)
    
    pepo = list(f'{aa}{i+1}' for i,aa in enumerate(pepo) if aa != 'd')
    ax.bar(pepo,
           e1_contr + e2_contr,
           0.65, 
           label='Full contribution', 
           alpha=1, 
           color='#323232',
           zorder=1,
           )

    ax.bar(pepo, e1_contr, 0.45, 
           label='First order', alpha=1, 
           color='#1571da',
           zorder=2,
           )
    ax.bar(pepo, e2_contr, 0.45, 
           label='Second order', alpha=1, 
           color='#ef476f',
           zorder=3,
           )

    plt.legend(frameon=False, loc='lower right', prop={'size': 12})
    ax.axhline(y=0, linewidth=1, color='k')

    # w = {"First order contribution": e1_contr,
    #      "Second order contribution": e2_contr,
    # }
    
    # bottom = np.zeros(len(e1_contr))
    # for label, contr in w.items():
    #     ax.bar(pepo, contr, 0.5, label=label, 
    #                 bottom=bottom
    #               )
    #     bottom += contr
     
    # set ticks
    # ax.set_xticks(np.arange(e1_contr.shape[1])+0.5)
    # ax.set_yticks(np.arange(e1_contr.shape[0])+0.5)   
    # ax.set_xticklabels(np.arange(e1_contr.shape[1])+1)
    # ax.set_yticklabels(alphabet)

    #set labels
    ax.set_xlabel('Amino acid', fontsize=15)
    ax.set_ylabel('ddG, kcal/mol', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title('Decomposition', fontsize=20)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return    

def plot_pepo_yields(M, R, pepo=None, corrected=True, fit=None, basename=None):
    '''
    fit argument options: None, no fit is plotted
    1p: plot sQSSA-derived fitted curves
    2p: plot tQSSA-derived fitted curves
    '''

    d = M[M['seq'] == pepo]
    d = d.sort_values('t', ascending=True)
    tot = int(R['total'][R['seq'] == pepo].iloc[0])
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 10), dpi=300)  
        
    if corrected:
        y = 'y_corr'
        err = 'y_err_corr'
    else:
        y = 'y'
        err = 'y_err'

    markers = ['o', 'D', 's']
    colors = ['#f58aa3', '#ef476f', '#a40e32']
    for i,c in enumerate([0.08, 0.8, 8]):
        ind = (d['c'] == c)
        ax.scatter(d[ind]['t'], 
                   d[ind][y], s=350,
                   label=f'c={c}',
                   marker=markers[i],
                   color=colors[i],
                   zorder=1,
)      
        ax.errorbar(d[ind]['t'], d[ind][y],
                    yerr=d[ind][err], 
                    ecolor=colors[i], 
                    capsize=5,
                    lw=3, 
                    elinewidth=3, 
                    ls='none',
                    zorder=1,
    )          
                  
    if fit == '1p':               
        t = np.arange(0, 540, 5)
        s = d['kcat/Km_1p'].iloc[0]
        s_err = d['kcat/Km_1p_err'].iloc[0]
        r2 = d['1p R_2_adj'].iloc[0]
        title = f'{pepo}' + r' $k_{cat}/K_{M}$' f' = {s:.3f}' + r' $\pm$ ' + f'{s_err:.3f}' + r' $M^{-1} s^{-1}$' + '\n' + r'  $R^{2}=$' + f'{r2:.3f}' + f' Total reads: {tot}'          
        ax.set_title(title, fontsize=22)
        for enz in [0.08, 0.8, 8]:

            x = np.vstack((t * 60,
                            np.asarray([enz] * t.size) * 1e-6)
                          ) 
                          
            ax.plot(t, tQSSA_1param(x.T, s), 
                    lw=6, 
                    alpha=0.5,
                    color='#323232',
                    label='fits',
                    zorder=0,
)
        
    if fit == '2p': 
        t = np.arange(0, 540, 5)
        kcat = d['kcat'].iloc[0]
        Km = d['Km'].iloc[0]
        s = d['kcat/Km_2p'].iloc[0]
        kcat_err = d['kcat_err'].iloc[0]
        Km_err = d['Km_err'].iloc[0]
        r2 = d['2p R_2_adj'].iloc[0]
        
        title = f'{pepo}' + r' $k_{cat}/K_{M}$' f' = {s:.2f} ' + '\n'\
                r' $k_{cat} = $' + f'{kcat:.1E}' + r' $\pm$ ' + f'{kcat_err:.1E}' +\
                r' $K_{M} = $' + f'{Km:.1E}' + r' $\pm$ ' + f'{Km_err:.1E}' + r'  $R^{2}=$' + f'{r2:.3f}' + f' {tot} reads'     
        ax.set_title(title, fontsize=22)
        for enz in [0.08, 0.8, 8]:

            x = np.vstack((t * 60,
                           np.asarray([enz] * t.size) * 1e-6)
                          ) 
                          
            ax.plot(t, tQSSA_2param(x.T, Km, kcat), 
                    lw=4, 
                    alpha=0.5,
                    color='#323232',
                    label='fits',
                    zorder=0,
)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-10, 550)
    ax.set_xlabel('Time', fontsize=30)
    ax.tick_params(axis='both', which='major',  labelsize=25)                                                 
    ax.set_ylabel('Yield', fontsize=30)       
    ax.legend()

    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return

'''plot yields for one pepo'''
# M = pd.read_csv(os.path.join('../kinetics fitting', 
#                               '2025.01.24__p4_master_kinetic_summary.csv')
# )

# S = pd.read_csv(os.path.join('../kinetics fitting', 
#                               '2025.01.24__p4_succint_kinetic_summary.csv')
# )

# R = pd.read_csv('../sample dataframes/POPEK4_master_list_pep.csv')
# columns = [x for x in R.columns if x.startswith('count_sum')]
# R['total'] = R[columns].sum(axis=1)

# pepo = 'SaIDTPNRdDGTFATSR'
# plot_pepo_yields(M, 
#                   R,
#                   pepo=pepo,
#                   fit='1p',
#                   corrected=True,
#                   basename=f'../figs/2025-04-09_{pepo}_pepo_fitting'
# )

# pdf = M[M['seq'] == pepo][['t', 'c', 'pos_count', 'neg_count', 'y', 'y_corr', 'y_err_corr']]
# pdf.to_csv(f'../figs/{pepo}_pepo_fitting.csv')

'''predicted vs true a to A mutation'''
model.load_weights('../tf_trained_models/RFA_20250118_ry_renorm.h5')
# raw_diff = []
# dyn_diff = []
# for pos in range(len(x_ind) + 1):
 
#     print(f'pos: {pos}  !! ! ! ! !! ')
#     ind = S['seq'].str[pos] == 'a'  
#     if ind.sum() > 0:
#         data = Data([AnalysisSample(X=S['seq'][ind], y=y[ind])])
        
#         data[0].X[:,pos] = 'A'
#         data[0].X = data[0].X[:,x_ind]
        
#         pip.enque([         
#                     pre.int_repr(alphabet=alphabet),
#                     pre.featurize_for_RFA(alphabet=alphabet, order='second')
#                   ])
        
#         data = pip.run(data)
#         y_pred = model.predict(data[0].X).ravel()
#         raw_diff.append((data[0].y - y_pred).mean())
        
#         ind1 = (data[0].y > 0) & (data[0].y < 6.5149)
#         ind2 = (y_pred > 0) & (y_pred < 6.5149)
#         ind3 = ind1 & ind2
#         dyn_diff.append((data[0].y[ind3] - y_pred[ind3]).mean())

#     else:
#         raw_diff.append(0)
#         dyn_diff.append(0)

'''predicted vs predicted a to A mutation'''
# model.load_weights('../tf_trained_models/RFA_20250124_zero_mean.h5')
# raw_diff = []
# dyn_diff = []
# for pos in range(len(x_ind) + 1):
 
#     print(f'pos: {pos}  !! ! ! ! !! ')
#     ind = S['seq'].str[pos] == 'a'  
#     if ind.sum() > 0:
#         data = Data([AnalysisSample(X=S['seq'][ind])])
        
#         # data[0].X[:,pos] = 'A'
#         data[0].X = data[0].X[:,x_ind]
        
#         pip.enque([         
#                     pre.int_repr(alphabet=alphabet),
#                     pre.featurize_for_RFA(alphabet=alphabet, order='second')
#                   ])
#         data = pip.run(data)
#         y_pred_a = model.predict(data[0].X).ravel()
        
        
#         data = Data([AnalysisSample(X=S['seq'][ind])])
#         data[0].X[:,pos] = 'A'
#         data[0].X = data[0].X[:,x_ind]
        
#         pip.enque([         
#                     pre.int_repr(alphabet=alphabet),
#                     pre.featurize_for_RFA(alphabet=alphabet, order='second')
#                   ])
#         data = pip.run(data)
#         y_pred_A = model.predict(data[0].X).ravel()        
        
#         raw_diff.append((y_pred_A - y_pred_a).mean())
        
#         ind1 = (y_pred_A > 0) & (y_pred_A < 6.5149)
#         ind2 = (y_pred_a > 0) & (y_pred_a < 6.5149)
#         ind3 = ind1 & ind2
#         dyn_diff.append((y_pred_A[ind3] - y_pred_a[ind3]).mean())

#     else:
#         raw_diff.append(0)
#         dyn_diff.append(0)
        
# from clibas.plotters import Y_score_var
# Y_score_var(-np.array(dyn_diff), '../figs/ddG_of_N_Methylation')

'''plot contributions for one pepo'''
# pepo = 'YLYDNYYAdYSDTSDNT'
# model.load_weights('../tf_trained_models/RFA_20250124_zero_mean.h5') 
# W = model.get_weights()
# e0, e1, e2 = parse_W_matrix(W, N_POS, N_AAS)

# a, b = pepo_aa_contributions(pepo, W)
# aa_contributions(pepo, a, b, alphabet, f'../figs/{pepo}_ddG_decomposition')

# off = 11.3383
# real_dG = kcatKm_to_dG(S['kcat/Km_1p'][S['seq'] == pepo].iloc[0], T=298, in_kcal=True)
# pred_dG = off + e0 + (a + b).sum()

# real_kK = S['kcat/Km_1p'][S['seq'] == pepo].iloc[0]
# pred_kK = dG_to_kcatKM(pred_dG)

# print(f'Measured kcat/Km: {real_kK} RFA kcat/kM: {pred_kK}')
# print(f'Measured Eakt: {real_dG} RFA Eakt: {pred_dG}')


'''limbo'''
# fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)  
# ax.scatter(np.log10(S['kcat/Km_1p']), np.log10(S['kcat/Km_2p']), s=5, alpha=0.1, color='#323232')
# ax.set_ylim(-5, 6)
# ax.set_xlim(-5, 6)


# fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)  
# ax.scatter(S['1p R_2_adj'], S['2p R_2_adj'], s=5, alpha=0.1, color='#323232')
# ax.set_ylim(-1, 1)
# ax.set_xlim(-1, 1)

# R = pd.read_csv('../sample dataframes/POPEK4_master_list_pep.csv')
# columns = [x for x in R.columns if x.startswith('count_sum')]
# R['total'] = R[columns].sum(axis=1)

# VAL = [
      
# 'VaTVIRASdESLNNHPG',
# 'aSDFANANdSSNTNTPS',
# 'PDSNGYGRdTaSSISGN',
# 'RTVGGSaPdARTTDYSD',
# 'VSHLPSGSdGTYYRPAY',
# 'DRSGLPLNdGGSLPISL',
# 'SNPSYSPVdTPAPANNH',
# 'NNNRVaYLdKTTAFNSH',
# 'TRLSHDYTdSRADYaPT',
# 'TPATASNRdNFSYPGTS',
# 'TADTRAVSdHTFGAAPA',
# 'RASDARNAdFNSHYTSN',
# 'HaDSNGSNdAVYYRSaV',
# 'NHRVYDNTdASTNGTSa',
# 'TTYAaGHYdALPNNYPT',
# 'DSNSSVFYdTVRHPSSL',
# 'NNIFTTGNdTFSGDHYN',
# 'PAHYSYDRdWAIGGNIN',
# 'ADLPSALLdHANNPFRP',
# 'FDTFPRTTdTLYGTHTP',
# 'NSRANPISdYYSYPLDA',
# 'RSSISNIAdATPGSYTY',
# 'AHRGSRNSdVATTYYGa',
# 'ARPTLSDIdITNPASNY',

# ]

# data = Data([AnalysisSample(X=VAL, 
#                             y=None,
#                             seq=VAL)
# ])

# data[0].X = data[0].X[:,x_ind]

# pip.enque([         
#             pre.int_repr(alphabet=alphabet),
#             # pre.shuffle(),
#             # pre.sample(sample_size=200),
#             # pre.tt_split(test_fraction=0.01),
#             pre.featurize_for_RFA(alphabet=alphabet, order='second')
#           ])

# data = pip.run(data)
# p = model.predict(data[0].X).ravel()

# for pred in p:
#     #print(S['kcat/Km_1p_err'][S['seq'] == pepo].iloc[0])
#     # print(int(R['total'][R['seq'] == pepo].iloc[0]))
#     print(dG_to_kcatKM(pred + off))

# OLD = pd.read_csv(os.path.join('../kinetics fitting', 
#                                '2025.01.13__p4_succint_kinetic_summary.csv')
# )

# NEW= pd.read_csv(os.path.join('../kinetics fitting', 
#                                '2025.01.24__p4_succint_kinetic_summary.csv')
# )

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)  
# ax.scatter(OLD['1p R_2_adj'], NEW['1p R_2_adj'], s=5, alpha=0.1, color='#323232')
# ax.plot((-3, 1),(-3, 1), lw=2, c='red')
# ax.set_ylim(0, 1)
# ax.set_xlim(0, 1)



# R = pd.read_csv('../sample dataframes/POPEK4_master_list_pep.csv')
# columns = [x for x in R.columns if x.startswith('count_sum')]
# R['total'] = R[columns].sum(axis=1)


# idx2 = S['kcat/Km_1p'] > 0.5
# idx3 = S['kcat/Km_1p'] < 30000
# idx7 = S['kcat/Km_1p_err'] / S['kcat/Km_1p'] < 0.3
# idx8 = R['total'] > 200
# idx9 = S['1p R_2_adj'] > 0.5
# S[idx2 & idx7 & idx8 & idx9 & idx3].shape[0]

'''variance analysis'''

# model.load_weights('../tf_trained_models/RFA_20250124_zero_mean.h5') 
# W = model.get_weights()
# e0, e1, e2 = parse_W_matrix(W, N_POS, N_AAS)

# from clibas.plotters import Y_score_var

# def var_at_pos_x(e1, e2, x):

#     e1v = ((e1[x] ** 2) / N_AAS).sum()
#     e2v = (((e2[x] / 2) ** 2) / (N_AAS ** 2)).sum()
#     return e1v + e2v


# v = np.array([var_at_pos_x(e1, e2, x) for x in range(N_POS)])
# Y_score_var(v/v.sum(), '../figs/variance_by_position_e1+e2')

# e1v = ((e1 ** 2) / N_AAS).sum()
# e2v = ((e2 ** 2) / (N_AAS ** 2)).sum()
# t = e1v+ e2v

# v = np.array((e1v/t, e2v/t))
# Y_score_var(v, '../figs/variance_by_order')

'''find top N terms'''
# #prepare data to test against
# data = Data([AnalysisSample(X=S['seq'], 
#                             y=y,
#                             seq=S['seq'])
# ])

# data[0].X = data[0].X[:,x_ind]

# pip.enque([         
#            pre.sample(sample_size=5000), 
#            pre.int_repr(alphabet=alphabet),
#            pre.featurize_for_RFA(alphabet=alphabet, order='second')
#           ])

# data = pip.run(data)

# model.load_weights('../tf_trained_models/RFA_20250124_zero_mean.h5') 
# W = model.get_weights()
# e0, e1, e2 = parse_W_matrix(W, N_POS, N_AAS)

# #find the variance of the first and second order terms
# e1v = (e1 ** 2) / N_AAS
# e2v = (e2 ** 2) / (N_AAS ** 2)

# i, j = np.triu_indices(N_POS, k=1, m=N_POS)
# Wv = np.hstack((e1v.ravel(), e2v[i, j].ravel()))
# w_indx = np.argsort(Wv)[::-1]

# W = np.hstack((e1.ravel(), e2[i, j].ravel()))
# acc = []
# for n_terms in range(1, 501):
#     idx = w_indx[:n_terms]
#     trunc_W = np.zeros(W.shape)
#     trunc_W[idx] = W[idx]  

#     trunc_W = [trunc_W.reshape(-1, 1).astype(np.float32),
#                np.array(e0.astype(np.float32)).reshape(1),
#             ]    

#     model.set_weights(trunc_W)

#     p = model.predict(data[0].X).ravel()
#     goodness = (np.corrcoef((p, data[0].y))**2)[0,1]
#     print(f'n={n_terms}: {goodness}')
#     acc.append(goodness)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)    

# c = np.array(['#1571da'] * 500)
# c[w_indx[:500] > 287] = '#ef476f'

# ax.scatter(np.arange(1, 201), 
#            acc[:200], s=50,
#            alpha=1,
#            c=c[:200], 
#            edgecolor='none',
#            antialiased=True)

# ax.set_xlim(-5, 205)
# ax.set_ylim(0, 1)

# ax.set_ylabel('R2', fontsize=14, color='#323232')
# ax.set_xlabel('N_terms', fontsize=14, color='#323232')
    
# plt.grid(lw=0.5, ls='--', c='slategrey', 
#           dash_capstyle='round', dash_joinstyle='round',
#           antialiased=True, alpha=0.2)  

# basename = '../figs/RFA_Rsq_vs_N_terms' 
# fig.savefig(basename + '.svg', bbox_inches = 'tight')
# fig.savefig(basename + '.png', bbox_inches = 'tight') 

'''box plots'''
# o1 = [
# 0.922,
# 0.9246,
# 0.9265,
# 0.9268,
# 0.9213,
# 0.9202,
# ]

# o2 = [
# 0.9693,
# 0.9664,
# 0.9661,
# 0.9634,
# 0.9676,
# 0.9674,
# ]

# nn = [
# 0.9699,
# 0.9653,
# 0.9666,
# 0.9680,
# 0.9605,
# 0.9663,
# ]

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=300)   

# ax.boxplot((o1, o2, nn),
#             positions=(0, 1, 2), sym=''
#            )

# sns.stripplot(data=(o1, o2, nn), jitter=True,
#               color='black', alpha=0.5, 
#               marker="$\circ$", ec="face", 
#               s=10)

# ax.set_ylim(0.9, 0.98)

# basename = '../figs/model_comparison_R2' 
# fig.savefig(basename + '.svg', bbox_inches = 'tight')
# fig.savefig(basename + '.png', bbox_inches = 'tight') 



# o1 = [
# 0.5048,
# 0.4782,
# 0.4713,
# 0.4709,
# 0.4924,
# 0.5226,
# ]

# o2 = [
# 0.2707,
# 0.2812,
# 0.2845,
# 0.3051,
# 0.2816,
# 0.281,
# ]

# nn = [
# 0.1941,
# 0.2219,
# 0.2121,
# 0.2086,
# 0.249,
# 0.2103,
# ]

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=300)   

# ax.boxplot((o1, o2, nn),
#            positions=(0, 1, 2), sym=''
#           )
# sns.stripplot(data=(o1, o2, nn), jitter=True,
#               color='black', alpha=0.5, 
#               marker="$\circ$", ec="face", 
#               s=10)

# ax.set_ylim(0, 0.60)

# basename = '../figs/model_comparison_MSE' 
# fig.savefig(basename + '.svg', bbox_inches = 'tight')
# fig.savefig(basename + '.png', bbox_inches = 'tight') 



























