# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:34:18 2021
@author: Alex Vinogradov
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

class SequencingData:
    #Just a container for FastqParser-related plotters
    
    def L_distribution(X, Y, where=None, basename=None):
        
        if not where: where=''
            
        fig = plt.figure(figsize=(18, 6), dpi=300)
        ax = fig.add_subplot(111)
        plt.bar(X, Y, color='#0091b5')
    
        ax.set_ylim(0, 1.02*np.max(Y))
        ax.set_xlim(np.min(X), np.max(X)+1)
        ax.set_xticks(np.linspace(np.min(X), np.max(X)+1, 10))
        ax.set_xticklabels(np.linspace(np.min(X), np.max(X)+1, 10, dtype=int))
        
        ax.set_xlabel('Sequence length', fontsize=30)
        ax.tick_params(axis='both', which='major',  labelsize=25)                                                 
        ax.set_ylabel('Count', fontsize=30)                     
    

        title = f'Distribution of sequence lengths in {where} dataset'
        ax.set_title(title, fontsize=34, y=1.04)
                    
        if basename is not None:                          
            #save png and svg, and close the file
            svg = basename + '.svg'
            png = basename + '.png'
            fig.savefig(svg, bbox_inches = 'tight')
            fig.savefig(png, bbox_inches = 'tight')
            plt.close()    

    def dataset_convergence(C, shannon, where, basename):
    
        y = np.sort(C)
        x = 100 * np.divide(np.arange(y.size), y.size)
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)    
        plt.plot(x, y, lw=2.5, c='#3b61b1', antialiased=True)

        ax.set_xlim(0, 101)
        ax.set_xticks(np.arange(0, 125, 25))
        
        ax.set_yscale('log')
        
        ax.set_ylabel(f'{where} sequence count', fontsize=14)
        ax.set_xlabel('Sequence percentile', fontsize=14)
        ax.set_title(f'Sequence-level convergence of {where} dataset', fontsize=16)
        plt.text(x=2, 
                 y=y.max(),
                 s=f'normalized Shannon entropy: {shannon:1.4f}', 
                 size=12,
                 horizontalalignment='left',
                 verticalalignment='center',)
            
        plt.grid(lw=0.5, ls='--', c='slategrey', 
                 dash_capstyle='round', dash_joinstyle='round',
                 antialiased=True, alpha=0.2)  
        
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()      
            
    def conservation(conservation, where, basename):
        
        fig = plt.figure(figsize=(18, 6), dpi=300)
        ax = fig.add_subplot(111)
        plt.plot(conservation, lw=3.5, c='#3b61b1')
    
        y_lim = np.ceil(np.max(conservation))
        ax.set_ylim(0, y_lim)
        ax.set_xlim(0, len(conservation))
        ax.set_xticks(np.linspace(0, len(conservation), 10))
        ax.set_xticklabels(np.linspace(0, len(conservation), 10, dtype=int))
        
        ax.set_xlabel('Sequence index', fontsize=30)
        ax.tick_params(axis='both', which='major',  labelsize=25)                                                 
        ax.set_ylabel('Conservation, norm', fontsize=30)                     
    
        title = f'Token-wise sequence conservation plot for {where} dataset'
        ax.set_title(title, fontsize=34, y=1.04)
                                              
        #save png and svg, and close the file
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()


    def tokenwise_frequency(freq, yticknames, where=None, loc=None, basename=None):
    
        if not where: where=''
        
        if where == 'dna': ylabel = 'Base'
        elif where == 'pep': ylabel = 'Amino acid'
        else: ylabel = 'Token'
            
        figsize = (1 + freq.shape[1] / 2, freq.shape[0] / 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(freq))
        c = ax.pcolormesh(freq, cmap=plt.cm.Blues, norm=norm, edgecolors='w', linewidths=4)
        cbar = fig.colorbar(c, ax=ax)
    
        cbar.ax.set_ylabel("frequency", rotation=-90, va="bottom", fontsize=22)
        cbar.ax.tick_params(labelsize=20)    
    
        #set ticks
        ax.set_xticks(np.arange(freq.shape[1])+0.5)
        ax.set_yticks(np.arange(freq.shape[0])+0.5)
        ax.set_xticklabels(np.arange(freq.shape[1])+1)
        ax.set_yticklabels(yticknames)
    
        #set labels
        if loc is not None:
            ax.set_xlabel(f'Position inside library region(s) {loc}', fontsize=21)
            
        ax.set_ylabel(ylabel, fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f'Position-wise frequency map for {where} dataset', fontsize=25)
        
        if basename is not None:
            #save png and svg, and close the file
            svg = basename + '.svg'
            png = basename + '.png'
            fig.savefig(svg, bbox_inches = 'tight')
            fig.savefig(png, bbox_inches = 'tight')
            plt.close()
            
        return

    def Q_score_summary(avg, std, loc, basename):
         
        fig = plt.figure(figsize=(18, 6), dpi=300)
        ax = fig.add_subplot(111)
        plt.plot(avg, lw=4, c='#3b61b1')
        plt.plot(avg+std, lw=1, c='#0091b5')
        plt.plot(avg-std, lw=1, c='#0091b5')
        ax.fill_between(np.arange(len(avg)), avg-std, avg+std, color='#0091b5', alpha=0.15)
    
        ax.set_ylim(0, np.nanmax(avg + std) + 5)
        ax.set_xlim(0, avg.size)
        ax.set_xticks(np.linspace(0, avg.size, 10))
        ax.set_xticklabels(np.linspace(1, avg.size+1, 10, dtype=int))
  
        ax.set_xlabel(f'{loc} region(s) index', fontsize=30)
        ax.tick_params(axis='both', which='major',  labelsize=25)                                               
        ax.set_ylabel('Q, average log score', fontsize=30)                     
    
        title = 'Q-score plot'
        ax.set_title(title, fontsize=34, y=1.04)
                                              
        #save png and svg, and close the file
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()   

class Analysis:
    
    def UMAP_HDBSCAN(Y,
                     labels,
                     C=None,
                     colors=None,
                     sample_name=None, 
                     basename=None, 
                     show_annotations=False):

        #matplotlib func; here mostly for posterity

        #cmap = sns.color_palette("husl", labels.max(), as_cmap=True)
        cmap = sns.color_palette("cividis", n_colors=10)
        colors = [x-3 if x > 3 else 0 for x in colors]
        colors = [cmap[x] for x in colors]
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_subplot(111)
        
        if C is not None:
            sizes = 5500 * np.power(np.divide(C, C.sum()), 0.55)
        else:
            sizes = 5500 * np.power(np.divide(1, Y.shape[0]), 0.55)
        
        if not sample_name: sample_name = 'unnamed sample'

        Q = plt.scatter(Y[:,0][::-1], 
                    Y[:,1][::-1], 
                    alpha=0.7, 
                    c=colors[::-1],
                    # cmap = cmap,
                    marker='o', edgecolors='none', 
                    s=sizes[::-1]
                    )      

        # fig.colorbar(Q, ax=ax)
        # noise = labels == -1
        # plt.scatter(Y[:,0][noise][::-1], 
        #             Y[:,1][noise][::-1], 
        #             alpha=0.6, 
        #             c='#070D0D',
        #             marker='o', edgecolors='none', 
        #             s=sizes[noise][::-1]
        #             )     
        
        # aint_noise = labels != -1
        # plt.scatter(Y[:,0][aint_noise][::-1], 
        #             Y[:,1][aint_noise][::-1], 
        #             alpha=0.6, 
        #             c=labels[aint_noise][::-1],
        #             cmap = cmap,
        #             marker='o', edgecolors='none', 
        #             s=sizes[aint_noise][::-1]
        #             )     
      
        if show_annotations:
            for cluster in labels:
                if not cluster == -1:
                    x_coord = np.average(Y[:,0][labels == cluster])
                    y_coord = np.average(Y[:,1][labels == cluster])
                    plt.text(x=x_coord, 
                              y=y_coord,
                              s=f'{cluster}', 
                              size=15,
                              weight='bold',
                              alpha=0.3,
                              color='#323232',
                              horizontalalignment='center',
                              verticalalignment='center')      
         
        plt.axis('off')

        title = f'umap/hdbscan: {sample_name}'
        ax.set_title(title, fontsize=20, y=1.04)
    
        if basename is not None:
            #save png and svg, and close the file
            svg = basename + '.svg'
            png = basename + '.png'
            fig.savefig(svg, bbox_inches = 'tight')
            fig.savefig(png, bbox_inches = 'tight')
            plt.close()
            
        plt.ion
        return

    def ClusteringHyperParams(min_clusters, min_samples, scores, 
                              sample_name=None, 
                              basename=None):
        
        fig = plt.figure(figsize=(7, 6), dpi=300)
        ax = fig.add_subplot(111)
        c = plt.pcolor(scores, cmap=sns.color_palette('mako', as_cmap=True))
        fig.colorbar(c, ax=ax)
        
        ax.set_xlabel('min_sample', fontsize=16)
        ax.set_ylabel('min_cluster', fontsize=16)      
    
        ax.set_xticks(np.arange(len(min_samples)) + 0.5)
        ax.set_yticks(np.arange(len(min_clusters)) + 0.5)
        
        ax.set_xticklabels(min_samples)
        ax.set_yticklabels(min_clusters)    
        
        ax.tick_params(axis='both', which='major',  labelsize=14)                                               
                   
        title = f'hdbscan clustering scores: {sample_name}'
        ax.set_title(title, fontsize=18, y=1.02)    
        
        if basename is not None:
            #save png and svg, and close the file
            svg = basename + '.svg'
            png = basename + '.png'
            fig.savefig(svg, bbox_inches = 'tight')
            fig.savefig(png, bbox_inches = 'tight')
            plt.close()

        return


class Miscellaneous:
    
    def single_linkage_dendrogram(link, labels=None, basename=None):
        
        from scipy.cluster.hierarchy import dendrogram
        
        dims = ((link.shape[0] + 1) * 0.3, 3)
        fig = plt.figure(figsize=dims, dpi=300)
        ax = fig.add_subplot(111)
        dendrogram(link, labels=labels, p=50, ax=ax)
        
        ax.set_xlabel('Tokens', fontsize=16)
        ax.set_ylabel('Height', fontsize=16)              
        ax.tick_params(axis='both', which='major',  labelsize=14)                                               
                   
        title = 'Ward linkage dendrogram for the feature matrix'
        ax.set_title(title, fontsize=18, y=1.02)    
        
        if basename is not None:
            #save png and svg, and close the file
            svg = basename + '.svg'
            png = basename + '.png'
            fig.savefig(svg, bbox_inches = 'tight')
            fig.savefig(png, bbox_inches = 'tight')
            plt.close()

        return        




#plotter for "../interrogators/virtual_mutagenesis.py"
def virtual_mutagenesis(proba, parent, alphabet, basename):
    
    #shape = 4 + (np.array(proba.T.shape) / 2)
    shape = (10.5, 10)
    
    fig, ax = plt.subplots(1, 1, figsize=shape, dpi=300)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    
    cmap = sns.color_palette("mako", as_cmap=True)
    
    c = ax.pcolor(proba, cmap=cmap, norm=norm,  
                  edgecolors='w', linewidths=4)
    
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("CNN calls", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(proba.shape[1])+0.5)
    ax.set_yticks(np.arange(proba.shape[0])+0.5)   
    ax.set_xticklabels(parent)
    ax.set_yticklabels(alphabet)

    #set labels
    ax.set_xlabel('Wild type amino acid', fontsize=21)
    ax.set_ylabel('Mutated to', fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_title(f'Virtual mutagenesis for {"".join(parent)}', fontsize=24)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return


def s_distribution_hist(s_scores, basename):
    '''
    Fig. 2e, S13b
        
        Parameters:
               s_scores:   list of tuples where each tuple corresponds to a sample:
                           (np ndarray of s scores, sample name)
                           
    '''  
  
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    s_min = 999  
    s_max = -999
     
    ct = ['#0091b5' ,'#db5000', '#3b61b1']
    
    for i,sample in enumerate(s_scores):
        data = sample[0]
        
        if data.max() > s_max:
            s_max = data.max()
            
        if data.min() < s_min:
            s_min = data.min()        
    
        plt.hist(data, bins=100, color=ct[i], label=sample[1], alpha=0.5, density=True)

    s_min = -17.4

    ax.set_xlim(1.05*s_min, 1.05*s_max)
    ax.set_xticks(np.linspace(s_min, s_max, 6))
    ax.set_xticklabels(np.round(np.linspace(s_min, s_max, 6), decimals=1))
        
    ax.set_xlabel('S score', fontsize=24)                                          
    ax.set_ylabel('Distribution density', fontsize=24)                     
    ax.tick_params(axis='both', which='major',  labelsize=20)     
    plt.legend(frameon=False, loc='upper left', prop={'size': 10})

    title = 'S score distribution density'
    ax.set_title(title, fontsize=24, y=1.04)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()    


def cum_e_distribution(s_scores, basename):
    '''
    Fig. S2
        
        Parameters:
               s_scores:   list of tuples where each tuple corresponds to a sample:
                           (np ndarray of s scores, sample name)
                           
    '''    
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    ymin = 999
    ymax = -999
    c = np.linspace(0, 1, len(s_scores) + 2)
    cmap = mpl.cm.get_cmap('cividis')
    
    for i,sample in enumerate(s_scores):
        y = sample[0]
        x = np.linspace(0, 100, y.size)
        y.sort()
        
        if y.max() > ymax:
            ymax = y.max()
        
        if y.min() < ymin:
            ymin = y.min()
            
        plt.plot(x, y, lw=4, c=cmap(c[i+1]), label=sample[1], alpha=0.8)

    ax.set_xlim(-5, 105)
    ax.set_xticks(np.linspace(0, 100, 6))
    ax.set_xticklabels(np.linspace(0, 100, 6))
    
    ax.set_ylim(1.05*ymin, 1.05*ymax)
    ax.set_yticks(np.linspace(ymin, ymax, 6))
    ax.set_yticklabels(np.round(np.linspace(ymin, ymax, 6), decimals=1))
    
    ax.set_xlabel('Peptides, percentile', fontsize=30)
    ax.tick_params(axis='both', which='major',  labelsize=25)                                               
    ax.set_ylabel('S score', fontsize=30)                     
    plt.legend(frameon=False, loc='lower right', prop={'size': 10})


    title = 'S cumulative statistic'
    ax.set_title(title, fontsize=34, y=1.04)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()            


def positional_epistasis(epi, pos1, pos2, alphabet, basename):
    '''
    A 2D map of epistatic interactions between amino acids in pos1 and pos2
    Used to make Fig. S7

        Parameters:
                    epi:   4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library
                           
             pos1, pos2:   int
                           
    '''   
     
    epi = epi[pos1, pos2]
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)
    
    scale = np.max([np.abs(np.nanmin(epi)), np.abs(np.nanmax(epi))])
    # scale = 4
    
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)
    c = ax.pcolor(epi, cmap=plt.cm.RdBu, norm=norm, edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Epistasis log score", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(epi.shape[1])+0.5)
    ax.set_yticks(np.arange(epi.shape[0])+0.5)
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
    
    epi = np.nanmean(np.abs(epi), axis=(2,3))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300) 
    norm = mpl.colors.Normalize(vmin=0, vmax=0.4)
    c = ax.pcolor(epi, cmap=mpl.cm.cividis, norm=norm, edgecolors='w', linewidths=4)    
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("abs(epi) score", rotation=-90, va="bottom", fontsize=21)
    cbar.ax.tick_params(labelsize=21)    

    for y in range(epi.shape[0]):
        for x in range(epi.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % epi[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
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
    title = 'Average positional epistasis'
    ax.set_title(title, fontsize=23)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return
    
def pep_epistatic_interactions(epi, pep, alphabet, basename):
    '''
    Plot all pairwise  epistatic interactions for peptide pep.
    Used to make Fig. 3e and S15b

        Parameters:
                    epi:   4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library
                           
                    pep:   peptide to the computation (1D np array, dtype=int)
    '''   
    
    fig, ax = plt.subplots(1, 1, figsize=(13, 7), dpi=300)
    
    ax.set_xlim(-0.5, pep.size - 0.5)
    ax.set_ylim(-0.5, np.divide(pep.size - 1, 2) + 0.5)
    
    rel_t = np.zeros((pep.size, pep.size))
    for pos1 in range(pep.size):
        for pos2 in range(pos1 + 1, pep.size):
            rel_t[pos1, pos2] = epi[pos1, pos2, pep[pos1], pep[pos2]]  
    
    #cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True)
    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    scale = np.max([np.abs(np.nanmin(rel_t)), np.abs(np.nanmax(rel_t))])
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)
    
    def arc(x1, x2, n=1000):
        #n - number of points to approximate the arc with
        x0 = np.divide(x1 + x2, 2)
        r = x2 - x0
        x = np.linspace(x1, x2, num=n)
        y = np.sqrt(r**2 - (x - x0)**2)
        return x, y

    for pos1 in range(pep.size):
        for pos2 in range(pos1 + 1, pep.size):
            x, y = arc(pos1, pos2)
            epi = rel_t[pos1, pos2]
            lw = 20 * np.power(np.divide(np.abs(epi), scale), 0.67)
            alpha = 1 * np.power(np.divide(np.abs(epi), scale), 0.5)
            
            ax.plot(x, y, color=cmap(norm(epi)), linewidth=lw, alpha=alpha)
    
    ax.scatter(np.arange(pep.size), np.zeros(pep.size,), s=700, alpha=1, color=cmap(norm(0)), zorder=10)
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, pad=0.01)
    cbar.ax.set_ylabel("Epistasis log score", rotation=-90, va="bottom", fontsize=18)
    cbar.ax.tick_params(labelsize=16)        
      
    seq = [alphabet[x] for x in pep]
    for x in np.arange(pep.size):
        
        ax.text(x, 
                -0.1,
                seq[x],
                zorder=20, 
                fontsize=20,
                fontweight='bold', 
                color=(0.33, 0.33, 0.33, 1),
                ha='center')
        
    seq = ''.join(seq)
    ax.set_title(f'Position-wise epistasis in peptide {seq}', fontsize=18)

    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight')
    return


def classifier_auroc_comparison(df, basename):
    '''
    Fig. 2h, S14
        
        Parameters:
                    df:   pandas dataframe holding accuracy and auroc values
                          for the classifiers; should contains the following
                          columns: classifier, accuracy, auroc
    ''' 
    
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    y = -np.log10(1 - df['auroc'].to_numpy())
    plt.bar(df['classifier'], y)


    y_min = 0.8
    y_max = 0.9999
    
    def t(x):
        return -np.log10(1 - x)
        
    ax.set_ylim(t(y_min), t(y_max))
    ytick = np.array([0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999])    
    ax.set_yticks(t(ytick))
    ax.set_yticklabels(ytick)


    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return  


def classifier_acc_comparison(df, basename):
    '''
    Fig. 2h, S14
        
        Parameters:
                    df:   pandas dataframe holding accuracy and auroc values
                          for the classifiers; should contains the following
                          columns: classifier, accuracy, auroc
    ''' 
    
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    y = -np.log10(1 - df['accuracy'].to_numpy())
    plt.bar(df['classifier'], y)

    y_min = 0.8
    y_max = 0.9985
    
    def t(x):
        return -np.log10(1 - x)
        
    ax.set_ylim(t(y_min), t(y_max))
    ytick = np.array([0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997, 0.9985])    
    ax.set_yticks(t(ytick))
    ax.set_yticklabels(ytick)


    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return  


def pep_yield_vs_prediction(y, proba, basename):
    '''
    Fig. 2i, 5e
        
        Parameters:
                    y:   1D np.ndarray, real peptide modification efficiencies
                proba:   1D np.ndarray, model calls for the same peptides
    ''' 

    fig = plt.figure(figsize=(14, 4), dpi=300)
    ax = fig.add_subplot(111)

    # cmap = sns.diverging_palette(20, 220, as_cmap=True)
    # norm = mpl.colors.Normalize(vmin = 1.05 * np.min(proba - y), 
    #                             vmax = 1.05 * np.max(proba - y))

    for i in range(y.size):
        
        y1 = y[i]
        y2 = proba[i]
        if y2 != y1:
            # plt.plot((i, i), (y1, y2), color=cmap(norm(y2-y1)), lw=4, alpha=0.7)
            plt.plot((i, i), (y1, y2), color='#323232', lw=4, alpha=0.7)

    plt.scatter(np.arange(y.size), y, s=100, marker='o', edgecolors='none', color='#1571da', zorder=100)
    plt.scatter(np.arange(proba.size), proba, s=100, marker='o', edgecolors='none', color='#ef476f', zorder=100)
        
    ax.set_xlim(-1, y.size)
    ax.set_ylim(-0.05, 1.05)
    
    ax.set_xticks(np.linspace(1, y.size, num=17))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(np.linspace(1, y.size, num=17))
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # plt.colorbar(sm, pad=0.01)

    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()      


def s1_vs_s2(s1, s2, basename):
    '''
    fill
    '''
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)   
    
    plt.hexbin(s1, 
               s2,
               gridsize=100,
               # mincnt=2,
               xscale='linear',
               yscale='linear',
               # bins='log',
               # color='#323232',
               cmap=sns.light_palette("#101010", as_cmap=True)
)

    ax.set_xlim(-12, 9)
    ax.set_ylim(-12, 9)    
    
    #set labels
    ax.set_xlabel('S1 score', fontsize=21)
    ax.set_ylabel('S2 score', fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)
    
    # title = 'Average positional epistasis'
    # ax.set_title(title, fontsize=23)    
    

    #save png and svg, and close the file    
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return  

def s_hist(s, basename):
    '''
    fill
    '''
    fig = plt.figure(figsize=(6, 2), dpi=300)
    ax = fig.add_subplot(111)   
    
    plt.hist(s, 
             bins=80,          
             color='#323232',
             
)

    ax.set_xlim(-12, 9)
    # ax.set_ylim(-15, 13)    
    
    #set labels
    # ax.set_xlabel('S1 score', fontsize=21)
    # ax.set_ylabel('S2 score', fontsize=21)
    # ax.tick_params(axis='both', which='major', labelsize=21)
    
    # title = 'Average positional epistasis'
    # ax.set_title(title, fontsize=23)    
    

    #save png and svg, and close the file    
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return  



def pep_yield_vs_s(y, S, basename):
    '''
    Fig. 3a, 5g
        
        Parameters:
                    y:   1D np.ndarray, real peptide modification efficiencies
                    S:   1D np.ndarray, S scores for the same peptides
    ''' 
    
    ind = np.argsort(S)
    S = S[ind]
    y = y[ind]

    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    plt.scatter(S, y, s=100, marker='^', edgecolors='none', color='#3b61b1', zorder=100)

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.set_xlim(-13, 8)

    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()   


def S_vs_proba_v3(S, p, basename):
    '''
    Fig. 3b, 5h
        
        Parameters:
                    S:   1D np.ndarray, S scores for a peptide dataset
                    p:   1D np.ndarray, model probability calls for the same peptides
    ''' 
    
    from scipy.stats import binned_statistic
    
    fig = plt.figure(figsize=(6, 2), dpi=300)
    ax1 = fig.add_subplot(111)

    ind = np.argsort(S)
    S = S[ind]
    p = p[ind]

    std, bin_edges, _ = binned_statistic(S, p, statistic='std', bins=100)
    mean, bin_edges, _ = binned_statistic(S, p, statistic='mean', bins=100)
    x = bin_edges[:-1]  + np.diff(bin_edges) / 2
    
    ax1.hist(S, bins=bin_edges, color='#323232', alpha=0.3, density=True)
    ax2 = ax1.twinx()

    # ax2.scatter(x, mean, color='#3b61b1', s=80, edgecolors='none', alpha=0.8)
    # ax2.errorbar(x, mean, yerr=std, color='#ef476f', lw=3, elinewidth=1.5, capsize=5)
    ax2.errorbar(x, mean, color='#ef476f', lw=3)
    # ax2.scatter(x, mean+std,color='#db5000', s=30, edgecolors='none', alpha=0.6, marker='v')
    # ax2.scatter(x, mean-std, color='#db5000', s=30, edgecolors='none', alpha=0.6, marker='^')
    ax2.fill_between(x, mean-std, mean+std, color='#ef476f', alpha=0.1)
    
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks(np.linspace(0, 1, 6))                                        
    ax2.set_ylabel('Model calls', fontsize=24)                     

    ax1.set_xlim(-12, 9)
    ax1.set_xticks(np.linspace(-12, 9, num=9))

    ax1.tick_params(axis='both', which='major',  labelsize=20)     
    ax2.tick_params(axis='both', which='major',  labelsize=20)

    title = 'S vs proba'
    ax1.set_title(title, fontsize=24, y=1.04)    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return   







def proba_vs_position(p, pos, axis, alphabet, basename):
    '''
    A 2D map of epistatic interactions between amino acids in pos1 and pos2
    Used to make Fig. S7

        Parameters:
                     pos: (7, 9, 10)
                     axis: (0, 1)
            
                           
    '''   

    to_average_over = np.setdiff1d(np.arange(len(pos)), axis)      
    p = np.nanmean(p, axis=tuple(to_average_over))
    
    # p[p > 0.5] = 0.95
    p[p < 0.5] = 0
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)
        
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # c = ax.pcolor(p, cmap=plt.cm.RdBu, norm=norm, edgecolors='w', linewidths=4)
    # c = ax.pcolor(p, cmap=sns.light_palette("#101010", as_cmap=True), norm=norm, edgecolors='w', linewidths=4)
    
    x = np.sort(np.tile(np.arange(p.shape[0]), p.shape[1]))
    y = np.tile(np.arange(p.shape[1]), p.shape[0])
    
    plt.scatter(y, x, s=800*p.flatten()**2, c='#323232', edgecolor=None)
    
    # cbar = fig.colorbar(c, ax=ax)
    # cbar.ax.set_ylabel("cnn proba", rotation=-90, va="bottom", fontsize=22)
    # cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(p.shape[1]))
    ax.set_yticks(np.arange(p.shape[0]))
    ax.set_xticklabels(alphabet)
    ax.set_yticklabels(alphabet)

    #set labels
    x_label = 'Amino acid in position ' + str(pos[axis[1]]+1)
    ax.set_xlabel(x_label, fontsize=25)
    
    y_label = 'Amino acid in position ' + str(pos[axis[0]]+1)
    ax.set_ylabel(y_label, fontsize=25)
    
    ax.tick_params(axis='both', which='major', labelsize=21)
    title = 'Sublibrary substrate fitness [cnn call] ' + str(pos[axis[0]]+1) + ' and ' + str(pos[axis[1]]+1)
    ax.set_title(title, fontsize=27)
    
    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight')
    plt.close()  





def Y_score(Y, alphabet, basename):
    '''
    Plot the Y score matrix
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 10), dpi=300)

    scale_min = -2
    scale_max = 3
    norm = mpl.colors.Normalize(vmin=scale_min,vmax=scale_max)
    
    import seaborn as sns
    cmap = sns.diverging_palette(254, 2.9, s=90, as_cmap=True)
    
    c = ax.pcolor(Y, cmap=cmap, norm=norm,
                  edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Y score", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(Y.shape[1])+0.5)
    ax.set_yticks(np.arange(Y.shape[0])+0.5)   
    ax.set_xticklabels(np.arange(Y.shape[1])+1)
    ax.set_yticklabels(alphabet)

    #set labels
    ax.set_xlabel('Position inside the insert', fontsize=25)
    ax.set_ylabel('Amino acid', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_title('Y-score map', fontsize=27)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return    
    
def Y_score_var(var, basename):
    '''
    Fig. 2d, S13
        
        Parameters:
                    var:   Positional variance of Y scores, 1D np.ndarray
                            i.e., var = np.var(Y, axis=0)
    ''' 

    import matplotlib.pyplot as plt    
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=300)
    ax.bar(np.arange(var.size), var)

    ax.set_ylim((0, 1))
    ax.set_yticks(np.arange(0, 0.8, 0.2))
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()    


def attribution_colorbar(norm, cmap, name):

    
    fig, ax = plt.subplots(1, 1, figsize=(0.3, 5), dpi=300)
    
    mpl.colorbar.ColorbarBase(ax, orientation='vertical', 
                              cmap=cmap,
                              norm=norm,  # vmax and vmin
                              label='Attribution magnitude')
    
    fig.savefig(name, bbox_inches='tight')
    plt.close()
    return


def avg_fitness_vs_aa_type(proba, pepos, basename):
    
    import scipy
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)    
    
    def data_for_aa_type(proba, pepos, aas):
    
        mask = np.in1d(pepos.ravel(), aas)
        mask = mask.reshape(pepos.shape)
        aa_per_pep = mask.sum(axis=1)
        x = np.unique(aa_per_pep)
        aa_proba = [proba[aa_per_pep == i].mean() for i in x]
        
        aa_sem = [scipy.stats.sem(proba[aa_per_pep == i]) for i in x]
        return x, aa_proba, aa_sem

    neut = ['S', 'T', 'A', 'Q', 'N']
    x, aa_proba, aa_sem = data_for_aa_type(proba, pepos, neut)
   
    ax.errorbar(x, aa_proba,
                yerr=aa_sem, 
                lw=3, 
                c='#a1a1aa', 
                capsize=5, 
                elinewidth=1.5,
                )
    
    ax.scatter(x, aa_proba, s=80, c='#a1a1aa', marker='o')  
    
    char = ['R', 'K', 'E', 'D']
    x, aa_proba, aa_sem = data_for_aa_type(proba, pepos, char)
   
    ax.errorbar(x, aa_proba,
                yerr=aa_sem, 
                lw=3, 
                c='#323232', 
                capsize=5, 
                elinewidth=1.5,
                )
    
    ax.scatter(x, aa_proba, s=80, c='#323232', marker='D')    

    phob = ['L', 'V', 'I', 'F', 'W', 'Y']
    x, aa_proba, aa_sem = data_for_aa_type(proba, pepos, phob)
   
    ax.errorbar(x, aa_proba,
                yerr=aa_sem, 
                lw=3, 
                c='#ef476f', 
                capsize=5, 
                elinewidth=1.5,
                )
    
    ax.scatter(x, aa_proba, s=80, c='#ef476f', marker='s')

    ax.set_ylim((0, 1))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    
    ax.plot((0, 13), (proba.mean(), proba.mean()), lw=1, ls='-', alpha=0.8, c='#323232')
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()    
    return



def lnk(Y, alphabet, basename):
    '''
    Plot the Y score matrix
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 10), dpi=300)

    scale_min = -1
    scale_max = 2
    norm = mpl.colors.Normalize(vmin=scale_min,vmax=scale_max)
    
    import seaborn as sns
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    c = ax.pcolor(Y, cmap=cmap, norm=norm,
                  edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Y score", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(Y.shape[1])+0.5)
    ax.set_yticks(np.arange(Y.shape[0])+0.5)   
    ax.set_xticklabels(np.arange(Y.shape[1])+1)
    ax.set_yticklabels(alphabet)

    #set labels
    ax.set_xlabel('Position inside the insert', fontsize=25)
    ax.set_ylabel('Amino acid', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_title('Y-score map', fontsize=27)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return    





