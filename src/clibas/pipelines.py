# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 16:48:39 2022
@author: Alex Vinogradov
"""

import os, time, copy, datetime, gc
import numpy as np
import pandas as pd
from clibas.baseclasses import Handler

class Pipeline(Handler):
    
    def __init__(self, *args):
        super(Pipeline, self).__init__(*args)
        self._on_startup()
        return

    def __repr__(self):
        return f'<Pipeline object; current queue size: {len(self.que)} op(s)>'

    def _on_startup(self):
        self.que = []
        if not hasattr(self, 'exp_name'):
            self.exp_name = 'untitled_exp'
        return

    def _describe_data(self, data=None):
        '''
        Go over every dataset for every sample and
        log all array shapes. Used during dequeing
        to keep track of data flows.
        '''
        data_descr = []
        
        if data is None:
            return data_descr
        
        for sample in data:
            
            data_descr.append((sample.name, sample.size))
            for tup in sample.iterarrays():
                
                if tup[0].shape:
                    shape = tup[0].shape
                else:
                    shape = None
                
                msg = f'{sample.name} {tup[1]} dataset shape: {shape}'
                self.logger.info(msg)
            
            if shape:
                if shape[0] == 0:
                    msg = 65 * '-'
                    msg = f'Sample {sample.name} has zero entries remaining!'
                    self.logger.warning(msg)
                
        msg = 65 * '-'
        self.logger.info(msg)
    
        return data_descr

    def _reassemble_summary(self, summary):
        
        ops = []
        times = []
        samples = []
        
        #the code below is a mess, but the task is trivial, 
        #so whatever; fix if nothing better to do
        for x in summary:
            ops.append(x['op'])
            times.append(x['op_time'])
            for j in x['data_description']:
                samples.append(j[0])
        
        samples = list(set(samples))
        sizes = np.zeros((len(summary), len(samples)))
        for i,entry in enumerate(summary):
            for tup in entry['data_description']:
                for j, name in enumerate(samples):
                    if tup[0] == name:
                        sizes[i,j] = tup[1]
        
        df = pd.DataFrame(columns=['time'] + samples, index=ops)
        df['time'] = times
        for i,name in enumerate(samples):
            df[name] = sizes[:,i]
        
        return df
    
    def enque(self, ops):
        '''
        Takes a list of functions and adds them to the pipeline queue.
        self.deque will take some data as an argument and apply dump
        the queue on it, i.e. sequentially transform the data by applying
        the queued up ops. 

        Parameters
        ----------
        ops :      a list of functions capable of acting on data.
                   every op should take data as the only argument
                   and return transformed data in the same format (Data object)

        Returns
        -------
        None.

        '''
        
        for func in ops:
            self.que.append(func)
            
        msg = f'{len(ops)} ops appended to pipeline; current queue size: {len(self.que)}'
        self.logger.info(msg)        
        return        

    def run(self, data=None, save_summary=True):
        '''
        Chainlinks the ops from the que list one by one to sequentially 
        transform the data. The method will execute the enqueued pipeline.
        
        Parameters
        ----------
        data :        Data object or None
                      if None, the first func in the que
                      has to load the data

        save_summary: save a .csv summary file containing
                      the progress of the experiment and
                      the basic description of data at
                      every stage. location: logs

        Returns
        -------
        transformed data as a Data object

        '''
        summary = list()
        data_descr = self._describe_data(data)
        summary.append({
                        'op': None, 'op_time': None, 
                        'data_description': data_descr
}
)
        
        for _ in range(len(self.que)):
        
            func = self.que.pop(0)
            msg = f'Queuing <{func.__name__}> op. . .'
            self.logger.info(msg)
            
            t = time.time()
            data = func(data)
            op_time = np.round(time.time() - t, decimals=3)
            
            msg = f'The operation took {op_time} s'
            self.logger.info(msg)
            data_descr = self._describe_data(data)
            
            summary.append({
                            'op': func.__name__,
                            'op_time': op_time,
                            'data_description': data_descr
}
)
        
        if save_summary:
            summary = self._reassemble_summary(summary)
            n = datetime.datetime.now()
            timestamp = f'_{n.year}_{n.month}_{n.day}_{n.hour}_{n.minute}_{n.second}'
            
            fname = f'{self.exp_name}_pipeline_summary_{timestamp}.csv'
            path = os.path.join(self.dirs.logs, fname)
            summary.to_csv(path)
        
        return data
    
    def stream(self, generator, save_summary=True):
        '''
        Execute the enqueued run by feeding the data
        sample by sample to the pipeline.

        Use this when the total amount of data exceeds
        the available machine memory.
        
        Parameters
        ----------
        generator :   a python generator function which
                      returns Sample type objects as data

        save_summary: save a .csv summary file containing
                      the progress of the experiment and
                      the basic description of data at
                      every stage. location: logs

        Returns
        -------
                      None
        '''
        
        from clibas.datatypes import Data
        
        que = copy.deepcopy(self.que)
        for sample in generator:
            #setting the exp name will help writing 
            #pipeline summaries for each sample
            self.exp_name = sample.name
        
            #turn sample into a Data instance and pass it though the pipeline
            data = Data([sample])
            self.que = copy.deepcopy(que)
            self.run(data=data, save_summary=save_summary)
            
            #unless the data is deleted, two datasets are stored in memory 
            #before the next call to data when a new sample is made.
            del data
            gc.collect()
            
            #reset library designs back to the original
            if hasattr(self, 'P_design'):
                self.P_design.rebuild()
                
            if hasattr(self, 'D_design'):
                self.D_design.rebuild()                
            
        #unset the experiment name after all done
        self.exp_name = None
        return
        
    
    