# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:27:57 2021
@author: Alex Vinogradov
"""

import os, importlib
import datetime
import yaml
from clibas.baseclasses import (
                                Logger,
                                DirectoryTracker
)

_reserved_aa_names = (
                      '_',
                      '+', 
                      '*', 
                      '1',
                      '2', 
                      '3',
                      '4', 
                      '5', 
                      '6', 
                      '7', 
                      '8',
                      '9', 
                      '0'
)

class Dispatcher:
    '''
    An object used to coordinate and configure data handlers. Its primary 
    purpose is to parse config files and use the specified information to  
    dispatch various data handlers in a coordinated way. 
    '''
    
    def __init__(self, config_fname):
                       
        self._preparse_yaml_config(config_fname)
        self._parse_config()
        return

    def __repr__(self):
        return f'<Dispatcher object at {hex(id(self))}>'
    
    def _preparse_yaml_config(self, config_fname):

        try:
            with open(config_fname, 'r') as file:
                raw = yaml.safe_load(file)
        except:
            msg = '<Dispatcher>: could not load the config file. . .'
            self.L.error(msg)
            raise IOError(msg) 
    
        config_dict = {}
        for config_type, attribs in raw.items():
            if not isinstance(attribs, dict):
                attribs = {'value': attribs}
    
            config_dict[config_type] = type(config_type, (object,), attribs)
            
        self.config = config_dict 
        return
    
    def _parse_dir_config(self):
        if not 'TrackerConfig' in self.config.keys():    
            self.dirs = DirectoryTracker()
        
        else:
            #don't have to check individual attribs becase DirectoryTracker
            #can handle it by itself
            self.dirs = DirectoryTracker(self.config['TrackerConfig'])
        return
      
    def _parse_logger_config(self):

        #logger: just need to set up the log file location, if needed
        if 'LoggerConfig' in self.config.keys():
            if hasattr(self.config['LoggerConfig'], 'log_to_file'):
                if self.config['LoggerConfig'].log_to_file:
                    if not hasattr(self.config['LoggerConfig'], 'log_fname'):
                        
                        t = datetime.datetime.now()
                        timestamp = f'_{t.year}_{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}'
                        self.config['LoggerConfig'].log_fname = os.path.join(
                        
                            self.dirs.logs, 
                            self.config['experiment'].value + timestamp + '_logs.txt'
)
        
            self.L = Logger(config=self.config['LoggerConfig']).logger

        else:
            self.L = Logger()

        return

    def _parse_constants_config(self):    

        #setup constants: some missing attribs might be inferred, but
        #at least something needs to be present
        if not 'constants' in self.config:    
            msg = '<Dispatcher>: config is missing the minimal definitions of constants. Aborting. . . '
            self.L.error(msg)
            raise ValueError(msg)
                    
        self.constants = self.config['constants']
        #deal with amino acids
        #amino acids can be inferred from the translation table
        if not hasattr(self.constants, 'aas'):
            if hasattr(self.constants, 'translation_table'):
                self.constants.aas = tuple(sorted(set(
                    
                                 x for x in self.constants.translation_table.values() 
                                 if x not in _reserved_aa_names))
                     )
        
        #deal with aa_SMILES: make sure that the aaSMILES alphabet and the
        #translation table one match. Make the aa_SMILES tuple with its values
        #corresponding to the sorted aa alphabet
        if hasattr(self.constants, 'aa_SMILES'):
            if not isinstance(self.constants.aa_SMILES, dict):
                msg = f'<Dispatcher>: config must supply aa_SMILES as a dictionary, found {type(self.constants.aa_SMILES)} instead. . . '
                self.L.error(msg)
                raise ValueError(msg)    
        
            sorted_keys = tuple(sorted(list(self.constants.aa_SMILES.keys())))
            if sorted_keys != self.constants.aas:
                msg = '<Dispatcher>: Amino acid alphabet derived from the translation table and aa_SMILES dictionary do not match. . . '
                self.L.error(msg)
                raise ValueError(msg)                   
        
            self.constants.aa_SMILES = tuple(self.constants.aa_SMILES[k] 
                                             for k in sorted_keys)

        #deal with nucleic acids, if any  
        #bases can be inferred from the translation table
        if not hasattr(self.constants, 'bases'):
            if hasattr(self.constants, 'translation_table'):
            
                self.constants.bases = tuple(sorted(set(
                    
                                ''.join(self.constants.translation_table.keys())))
                      )
        return
 
    def _parse_lib_design_config(self):
        
        from clibas.lib_design import LibraryDesign
        if not 'LibraryDesigns' in self.config:
            msg = '<Dispatcher>: config has not specified any library designs. . .'
            self.L.warning(msg)
            return
        
        #deal with peptide libraries first
        if hasattr(self.config['LibraryDesigns'], 'pep_templates') and \
           hasattr(self.config['LibraryDesigns'], 'pep_monomers'):
            
            if hasattr(self.constants, 'aas'):             
                self.P_design = LibraryDesign(
                                              templates=self.config['LibraryDesigns'].pep_templates,
                                              monomers=self.config['LibraryDesigns'].pep_monomers,
                                              lib_type='pep',
                                              val_monomer=self.constants.aas
                                             ) 
            else:
                msg = 'Dispatcher: cannot setup a peptide library design without the amino acid alphabet specification. . .'
                self.L.error(msg)
                raise ValueError(msg)
                
        #if only of the two necessary atrributes is present, raise
        #note that if both are absent, it's OK
        elif hasattr(self.config['LibraryDesigns'], 'pep_templates') !=   \
             hasattr(self.config['LibraryDesigns'], 'pep_monomers'):
                 
            msg = 'Dispatcher: config is missing necessary parameters to setup peptide library designs. . .'
            self.L.error(msg)
            raise ValueError(msg)

        #same, but now for DNA libs
        if hasattr(self.config['LibraryDesigns'], 'dna_templates') and  \
           hasattr(self.config['LibraryDesigns'], 'dna_monomers'):
            
            if hasattr(self.constants, 'bases'):
                self.D_design = LibraryDesign(
                                              templates=self.config['LibraryDesigns'].dna_templates,
                                              monomers=self.config['LibraryDesigns'].dna_monomers,
                                              lib_type='dna',
                                              val_monomer=self.constants.bases
                                             )
            else:
                msg = 'Dispatcher: cannot setup a DNA library design without the nucleotide alphabet specification. . .'
                self.L.error(msg)
                raise ValueError(msg)
        
        #if only of the two necesasry atrributes is present, raise
        elif hasattr(self.config['LibraryDesigns'], 'dna_templates') != \
             hasattr(self.config['LibraryDesigns'], 'dna_monomers'):
                 
            msg = 'Dispatcher: config is missing necessary parameters to setup DNA library designs. . .'
            self.L.error(msg)
            raise ValueError(msg)
            
        return
        
    def _parse_config(self):
        
        #check that all necessary config params are in place, one by one
        #if not, fall back to some innocuous defaults where possible
        if 'experiment' in self.config:
            self.experiment = self.config['experiment'].value
        else: 
            self.experiment = 'untitled_experiment'
        
        self._parse_dir_config()
        self._parse_logger_config()
        self._parse_constants_config()
        self._parse_lib_design_config()
        
        self.common = {'logger': self.L,
                       'dirs': self.dirs, 
                       'exp_name': self.experiment,
                       'constants': self.constants,
                       'P_design': self.P_design,
                       'D_design': self.D_design,
                      }
        return
    
    def _config_to_dict(self, conf):        
        return dict((name, getattr(conf, name)) for name 
                    in dir(conf) if not name.startswith('__'))        

    def _dispatch_handler(self, handler):
       
        lookup_modules = ['parsers',
                          'dataanalysis',
                          'datapreprocessors',
                          'pipelines',
                          'classifier'
                         ]
        
        config_mapping = {
                          'Pipeline': 'PipelineConfig',
                          'FastqParser': 'FastqParserConfig',
                          'DataAnalysisTools': 'DataAnalysisConfig',
                          'DataPreprocessor': 'PreproConfig',
                          'Classifier': 'ClassifierConfig'
                         }
        
        for module in lookup_modules:
            try:
                m =  importlib.import_module(f'clibas.{module}')
                obj = getattr(m, handler.__name__)
                break
            except:
                obj = None
            
        if not obj: 
            msg = f'<Dispatcher> could not set up {handler}'
            self.L.error(msg)
            raise ImportError(msg)

        params = dict()
        if config_mapping[handler.__name__] in self.config:
            params = self._config_to_dict(self.config[config_mapping[handler.__name__]])
         
        params.update(self.common)
        return obj(params)
   
    @classmethod
    def dispatch(cls, config_fname, handlers):
        '''
        Load a config file, parse it, and dispatch the handlers.
        
        The main (and the only) publically accesible method of the class.
        
        Take a tuple of handlers and initialize each by giving 
        them relevant configs params. Also makes sure that the
        information about file directories and logger is shared
        between all initialized objects
        
        Parameters
        ----------
        config   : location to a config file; only .yaml files 
                   are currently accepted as configs
        
        handlers : a tuple of handlers to be instantiated or an individual
                   handler
                   
                   ex: 
                      FastqParser or (Pipeline, FastqParser, DataPreprocessor)
                     
        Returns
        -------
        A tuple of initialized handler instances OR an indivual handler instance
        '''
    
        self = cls(config_fname)
    
        #make sure that either individual handlers or tuples thereof work
        handlers = [handlers]
        try:
            handlers = [item for sublist in handlers for item in sublist]
        except:
            pass
     
        h = list()
        for handler in handlers:
            h.append(self._dispatch_handler(handler))
            msg = f'{handler} was succesfully initialized'
            self.L.info(msg)
        
        if len(h) == 1:
            return h[0]
        else:
            return tuple(h)





