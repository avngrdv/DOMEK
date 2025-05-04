# -*- coding: utf-8 -*-
"""
Created on Thu May 1 00:03:23 2025
@author: Alex Vinogradov
"""
import numpy as np

def plot_e1_tensor(e1, alphabet, basename, scale=3):
    '''
    Plot e1 matrix (the matrix of first-order RFA terms)

    Parameters:
                 e1: np.array, the matrix of first-order RFA terms
                
           alphabet: the overall amino acid alphabet
           
           basename: str, output filename without extension
           
              scale: float/int, colorbar scale
                           
    Returns:
              None; .svg and .png figure files will be written as specified
              by the 'basename' argument
    '''          
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 10), dpi=300)

    import seaborn as sns
    cmap = sns.diverging_palette(254, 2.9, s=90, l=43, as_cmap=True)
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)

    c = ax.pcolor(e1, cmap=cmap, norm=norm,
                  edgecolors='w', linewidths=4)

    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_ylabel("Amino acid ddG contributions",
                       rotation=-90, 
                       va="bottom", 
                       fontsize=22
)
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

def plot_e2_tensor_slice(e2, pos1, pos2, alphabet, basename, scale=1):
    '''
    Plot a positional slice of an e2 matrix (the matrix of second-order RFA terms); 
    shows second-order interactions between amino acids in pos1 and pos2

    Parameters:
                 e2: 4D np.array, full matrix of RFA second-order terms
                 
                pos1: int, position 1
                
                pos2: int, position 2
                
           alphabet: the overall amino acid alphabet
           
           basename: str, output filename without extension
           
              scale: float/int, colorbar scale
                           
    Returns:
              None; .svg and .png figure files will be written as specified
              by the 'basename' argument
    '''          
        
    import matplotlib as mpl
    import matplotlib.pyplot as plt     
    
    e2 = e2[pos1, pos2]
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)
    
    import seaborn as sns
    cmap = sns.diverging_palette(254, 2.9, s=90, l=43, as_cmap=True)
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)
    
    c = ax.pcolor(e2, cmap=cmap, norm=norm, edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Epistatic ddG contributions", 
                       rotation=-90, 
                       va="bottom", 
                       fontsize=22
)
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
    return  

def epistasis_bw_positions(e2, basename, scale=0.25):
    '''
    Reduce e2 matrix (the matrix of second-order RFA terms) 
    to a 2d map (seq_len, seq_len) by averaging along the last 
    two axes and plot the the result.

    Parameters:
                 e2: 4D np.array, full matrix of RFA second-order terms
            
           basename: str, output filename without extension
           
              scale: float/int, colorbar scale
                           
    Returns:
              None; .svg and .png figure files will be written as specified
              by the 'basename' argument
    '''   
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    e2 = np.nanmean(np.abs(e2), axis=(2,3))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300) 
    norm = mpl.colors.Normalize(vmin=0, vmax=scale)
    import seaborn as sns
    cmap = sns.light_palette("#1571da", as_cmap=True)
    c = ax.pcolor(e2, cmap=cmap, norm=norm, edgecolors='w', linewidths=4)    
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("abs(ddG), kcal/mol", 
                       rotation=-90, 
                       va="bottom", 
                       fontsize=21
)
    cbar.ax.tick_params(labelsize=21)    

    for y in range(e2.shape[0]):
        for x in range(e2.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % e2[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     c='#323232',
                     fontdict={'fontname': 'Arial'}
)

    #set ticks
    ax.set_xticks(np.arange(e2.shape[1])+0.5)
    ax.set_yticks(np.arange(e2.shape[0])+0.5)
    ax.set_xticklabels(np.arange(e2.shape[0])+1)
    ax.set_yticklabels(np.arange(e2.shape[0])+1)

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

def plot_aa_contributions(pep, e1_contr, e2_contr, alphabet, basename):
    '''
    Plot the outputs of 'pep_aa_contributions' function

    Parameters:
                pep: str, primary amino acid sequence for the peptide of interest
            
           e1_contr: position-wise first-order contributions as returned by
                     'pep_aa_contributions' function
                     
           e2_contr: position-wise second-order contributions as returned by
                     'pep_aa_contributions' function      
                    
           alphabet: the overall amino acid alphabet
                       
           basename: str, output filename without extension
            
    Returns:
              None; .svg and .png figure files will be written as specified
              by the 'basename' argument
    '''       
    import matplotlib.pyplot as plt    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6), dpi=300)
    
    #TODO: this will need to be generalized
    pep = list(f'{aa}{i+1}' for i,aa in enumerate(pep) if aa != 'd')
    ax.bar(pep,
           e1_contr + e2_contr,
           0.65, 
           label='Full contribution', 
           alpha=1, 
           color='#323232',
           zorder=1,
)
    ax.bar(pep, e1_contr, 0.45, 
           label='First order', alpha=1, 
           color='#1571da',
           zorder=2,
) 
    ax.bar(pep, e2_contr, 0.45, 
           label='Second order', alpha=1, 
           color='#ef476f',
           zorder=3,
)

    plt.legend(frameon=False, loc='lower right', prop={'size': 12})
    ax.axhline(y=0, linewidth=1, color='k')

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

def plot_term_ablation(acc, w_indx, n_pos, n_aas, max_n_terms, basename):
    '''
    Plot the outputs of 'pep_aa_contributions' function

    Parameters:
                acc: np.array holding out-of-sample R2 values computed at
                     various n_terms
            
             w_indx: an index array for the W weight vector.
                     W[w_indx] should sort individual terms
                     in decreasing order
                     
              n_pos: int, peptide length (number of amino acids)
                 
              n_aas: int, number of amino acids in the genetic alphabet
                                  
        max_n_terms: int, maximum number of terms to plot to
                           
           basename: str, output filename without extension
            
    Returns:
              None; .svg and .png figure files will be written as specified
              by the 'basename' argument
    '''     
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)    
    
    #set colors for first- and second-order terms
    c = np.array(['#1571da'] * w_indx.size)
    
    num_e1_terms = n_pos * n_aas
    c[w_indx > num_e1_terms - 1] = '#ef476f'
    
    ax.scatter(np.arange(1, max_n_terms + 1), 
               acc[:max_n_terms], s=50,
               alpha=1,
               c=c[:max_n_terms], 
               edgecolor='none',
               antialiased=True
)
    
    ax.set_xlim(-5, max_n_terms + 5)
    ax.set_ylim(-0.02, 1.02)
    
    ax.set_ylabel('out-of-sample R2', fontsize=14, color='#323232')
    ax.set_xlabel('number of terms', fontsize=14, color='#323232')
        
    plt.grid(lw=0.5, ls='--', c='slategrey', 
              dash_capstyle='round', dash_joinstyle='round',
              antialiased=True, alpha=0.2)  
    
    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight') 
    return
    













