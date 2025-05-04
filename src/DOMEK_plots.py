# -*- coding: utf-8 -*-
"""
Created on Fri May  2 01:46:25 2025
@author: Alex Vinogradov
"""

import numpy as np
from DOMEK import tQSSA_2param, tQSSA_1param

def plot_peptide_fits(M, C, pep=None, corrected=True, fit=None, basename=None):
    '''
    Plot DOMEK-derived peptide yields and the associated fits
    Used to generate Fig. 3a plots.

    Parameters:
        
             M: pandas dataframe, master kinetic summary
             
             C: pandas dataframe, master sequence spreadsheet containing
                read count information
                
           pep: str, primary sequence for the peptide of interest
           
     corrected: bool, whether corrected or uncorrected yields and SEs are
                plotted
                
           fit: str or None; 
                "1p": plot 1-parameter tQSSA-derived fits
                "2p": plot 2-parameter tQSSA-derived fits3
                None: no fits are plotted
             
      basename: str, output filename without extension
      
    Returns:
              None; .svg and .png figure files will be written as specified
              by the 'basename' argument
    '''
    
    d = M[M['seq'] == pep]
    d = d.sort_values('t', ascending=True)

    columns = [x for x in C.columns if x.startswith('count_sum')]
    C['total'] = C[columns].sum(axis=1)
    tot = int(C['total'][C['seq'] == pep].iloc[0])
    
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
        r2 = d['1param R_2_adj'].iloc[0]
        title = f'{pep}' + r' $k_{cat}/K_{M}$' f' = {s:.3f}' + r' $\pm$ ' + f'{s_err:.3f}' + r' $M^{-1} s^{-1}$' + '\n' + r'  $R^{2}=$' + f'{r2:.3f}' + f' Total reads: {tot}'          
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
        r2 = d['2param R_2_adj'].iloc[0]
        
        title = f'{pep}' + r' $k_{cat}/K_{M}$' f' = {s:.2f} ' + '\n'\
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




