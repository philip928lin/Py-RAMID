# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:31:32 2020

@author: CYLin
"""
import pandas as pd
import numpy as np
from datetime import datetime
from func_timeout import func_timeout, FunctionTimedOut
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import pickle
import os
from inspect import signature
from PyRAMID.Setting import ConsoleLogParm, MsglevelDict, AddLocalLogFile, RemoveLocalLogFile

class GeneticAlgorithm(object):
    def __init__(self, function = lambda x: 0, dimension = None, variable_type = 'bool', \
                 variable_boundaries = None,\
                 variable_type_mixed = None, \
                 WD = None,\
                 saveGADataPerIter = False,\
                 function_timeout = 1000,\
                 parallel = 0,\
                 algorithm_parameters = {'max_num_iteration': None,\
                                         'population_size':100,\
                                         'mutation_probability':0.1,\
                                         'elit_ratio': 0.01,\
                                         'crossover_probability': 0.5,\
                                         'parents_portion': 0.3,\
                                         'crossover_type':'uniform',\
                                         'max_iteration_without_improv': None},\
                 continue_file = None,\
                 Msglevel = None):
        
        self.__name__ = "GA"
        # Setup the log msg (console) (log file is added below.)
        self.logger = logging.getLogger(__name__)
        if Msglevel is None: Msglevel = ConsoleLogParm['Msglevel']
        else:
            assert Msglevel in ['debug', 'info', 'warning', 'error'], print("ValueError Msglevel must be one of these [None, 'debug', 'info', 'warning', 'error'].")
            Msglevel = MsglevelDict[Msglevel]
        self.logger.setLevel(Msglevel)   
        self.CreateFileHandler = False
        
        # Setup input parameter
        if continue_file is not None:
            assert os.path.exists(continue_file), self.logger.error("PathError given continue_file is not exist {}.".format(continue_file))
            self.continue_file = continue_file
            self.load_continue_file()
            self.continue_file = continue_file # load file will overwrite the new value, so we need to assign it again.
            self.logger, self.fh = AddLocalLogFile(self.__name__, self.logger, self.WD) # since fh cannot be saved
        else:
            # Set working folder and check saveGADataPerIter
            self.saveGADataPerIter = saveGADataPerIter
            if WD is None:
                self.WD = None
                if saveGADataPerIter:
                    self.logger.error("ValueError To enable saveGADataPerIter and log file, valid WD must be given.")
                    self.saveGADataPerIter = False
            else:
                assert os.path.isdir(WD), self.logger.error("PathError given WD is not exist {}.".format(WD))
                self.WD = WD
                # Add local log file
                self.logger, self.fh = AddLocalLogFile('GA.log', self.logger, self.WD)
                self.CreateFileHandler = True
            
            # Output
            self.pop = None      # This option provide an opportunity to continue last run if program break down. 
            self.best_var = None
            self.best_minobj = None
            self.report = []
            self.iter = 0
        
            # Check inputs
            assert isinstance(dimension, (float, int)), self.logger.error("TypeError dimension must be integer.")
            assert(variable_type=='bool' or variable_type=='int' or  variable_type=='real' or variable_type=='cate'), \
                   self.logger.error("TypeError variable_type must be 'bool', 'int', 'real', or 'cate'.")
            assert parallel in [0,1,2], self.logger.error("TypeError parallel must be 0: no parallel, 1: parallel, 2: parallel with new sub-working folders.")
            assert (callable(function)), self.logger.error("TypeError function must be callable.")
            if parallel == 2:
                assert [i for i in signature(function).parameters] == ['var', 'GA_WD'],  self.logger.error("ValueError To run GA for parallel = 2 (coupling), given sim function has to contain two input arguments: 'var' (1d array) and GA_WD, which user should use RiverwareWrap.createFiles(GA_WD) to create subfolder in their sim function and conduct the simulation under this new directory.")
                self.SubfolderPath = os.path.join(self.WD, "AutoCalibration")
                if os.path.isdir(self.SubfolderPath) is not True:
                    os.mkdir(self.SubfolderPath)
                    self.logger.info("Create subfolder AutoCalibration at {}".format(self.SubfolderPath))
                    
            
            # Assign input
            self.dim = int(dimension)
            self.func = function
            self.parallel = parallel
            self.var_index = {}
            if function_timeout is None:
                function_timeout = 86400 # If given None, we set timeout = 1 day
            self.funtimeout = int(function_timeout)
            self.continue_file = continue_file
            
            # Assign var_type and var_bound and var_index
            if variable_type_mixed is None: 
                # We assign identical type according to variable_type to each variable.
                if variable_type == 'real': 
                    self.var_type = np.array([['real']]*self.dim)
                    self.var_index["cate"] = np.array([])
                    self.var_index["int"] = np.array([])
                    self.var_index["real"] = np.where(self.var_type == 'real')[0]
                else:
                    self.var_type = np.array([['int']]*self.dim)    # 'int', 'bool', 'cate'
                    if variable_type == 'cate':
                        self.var_index["cate"] = np.where(self.var_type == 'int')[0]
                        self.var_index["int"] = np.array([])
                        self.var_index["real"] = np.array([])
                    else:
                        self.var_index["cate"] = np.array([])
                        self.var_index["int"] = np.where(self.var_type == 'int')[0]
                        self.var_index["real"] = np.array([])
                # Assign var_bound if it is not given
                if variable_boundaries is None:
                    self.var_bound = np.array([[0,1]]*self.dim)
                else:
                    assert isinstance(variable_boundaries, (list, np.ndarray)), self.logger.error("TypeError variable_boundaries must be numpy array or list.") 
                    variable_boundaries = np.array(variable_boundaries) 
                    assert (variable_boundaries.shape == (self.dim,2)),  self.logger.error("ValueError variable_type_mixed must have a shape (dimension, 2).") 
                    self.var_bound = variable_boundaries
            else: 
                # var type should be defined in variable_type_mixed
                assert isinstance(variable_type_mixed, (list, np.ndarray)), self.logger.error("TypeError variable_type_mixed must be numpy array or list.") 
                assert isinstance(variable_boundaries, (list, np.ndarray)), self.logger.error("TypeError variable_boundaries must be numpy array or list.")  
                variable_type_mixed = np.array(variable_type_mixed)
                variable_boundaries = np.array(variable_boundaries)            
                assert (len(variable_type_mixed) == self.dim),  self.logger.error("ValueError variable_type_mixed must have a length equal dimension.") 
                assert (variable_boundaries.shape == (self.dim,2)),  self.logger.error("ValueError variable_type_mixed must have a shape (dimension, 2).") 
                self.var_type = variable_type_mixed
                self.var_bound = variable_boundaries
                self.var_index["cate"] = np.where(self.var_type == 'cate')[0]
                self.var_index["int"] = np.where(self.var_type == 'int')[0]
                self.var_index["real"] = np.where(self.var_type == 'real')[0]
                self.var_type = np.where(self.var_type=='cate', 'int', self.var_type) # Replace cate as int for rest of the calculation
                
    
            # Check algorithm_parameters
            assert set(['max_num_iteration', 'population_size', 'mutation_probability', 'elit_ratio', 'crossover_probability', 'parents_portion', 'crossover_type', 'max_iteration_without_improv']).issubset(set(algorithm_parameters.keys())), self.logger.error("KeyError Missing keys in the algorithm_parameters.")
            self.par = algorithm_parameters 
            self.par['population_size'] = int(self.par['population_size'])
            assert (self.par['parents_portion'] <= 1 and self.par['parents_portion'] >= 0), self.logger.error("ValueError parents_portion must be in range [0,1].")
            assert (self.par['mutation_probability'] <= 1 and self.par['mutation_probability'] >= 0), self.logger.error("ValueError mutation_probability must be in range [0,1].")
            assert (self.par['crossover_probability'] <= 1 and self.par['crossover_probability'] >= 0), self.logger.error("ValueError crossover_probability must be in range [0,1].")
            assert (self.par['elit_ratio'] <= 1 and self.par['elit_ratio'] >= 0), self.logger.error("ValueError elit_ratio must be in range [0,1].")
            assert (self.par['mutation_probability'] <= 1 and self.par['mutation_probability'] >= 0), self.logger.error("ValueError mutation_probability must be in range [0,1].")
            assert (self.par['crossover_type'] == 'uniform' or self.par['crossover_type'] == 'one_point' or self.par['crossover_type'] == 'two_point'), self.logger.error("ValueError crossover_type must be 'uniform', 'one_point', or 'two_point' Enter string.")
            
            # Make sure that population_size is properly assigned
            self.par['parent_size'] = int(self.par['parents_portion']*self.par['population_size'] )
            trl = self.par['population_size'] - self.par['parent_size']
            if trl % 2 != 0: 
                self.par['parent_size'] += 1  # To guarentee even number 
            
            # Make sure that num_elit is properly assigned
            trl = self.par['population_size']*self.par['elit_ratio']
            if trl < 1 and self.par['elit_ratio'] > 0:  # At least 1 elite
                self.par['num_elit'] = 1
            else:
                self.par['num_elit'] = int(trl)     # Round down
            
            # Make sure that max_num_iteration is properly assigned
            if self.par['max_num_iteration'] is None:
                self.par['max_num_iteration'] = 0
                for i in range (0, self.dim):
                    if self.var_type[i] == 'int':
                        self.par['max_num_iteration'] += (self.var_bound[i][1] - self.var_bound[i][0])*self.dim*(100/self.par['population_size'])
                    else:
                        self.par['max_num_iteration'] += (self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.par['population_size'])
                self.par['max_num_iteration'] = int(self.par['max_num_iteration'])
                if (self.par['max_num_iteration']*self.par['population_size']) > 10000000:
                    self.par['max_num_iteration'] = 10000000/self.par['population_size']
            else:
                self.par['max_num_iteration'] = int(self.par['max_num_iteration'])
                
            # Make sure that max_num_iteration is properly assigned    
            if self.par['max_iteration_without_improv'] == None:
                self.par['max_iteration_without_improv']= self.par['max_num_iteration'] + 1
            else:
                self.par['max_iteration_without_improv'] = int(self.par['max_iteration_without_improv'])      
            self.logger.info("The GA object have been initiated: \n"+"\n".join(['{:^23} :  {}'.format(keys, values) for keys,values in self.par.items()]))
        return None
    
    def load_continue_file(self):
        filepath = self.continue_file
        with open(filepath, "rb") as f:
            dictionary = pickle.load(f)
        #print(dictionary)
        # Load back all the previous class attributions.
        for key in dictionary:
            setattr(self, key, dictionary[key])
        self.logger.info("The previous GA object have been loaded back and ready to run.")
        
    def save_attribution(self, path):
        dictionary = self.__dict__.copy()
        dictionary.pop('fh', None)  # handler cannot be pickled.
        dictionary.pop('logger', None)  # handler cannot be pickled.
        with open(os.path.join(path, "GAobject.pickle"), 'wb') as outfile:
            pickle.dump(dictionary, outfile)
        
    def Print(self):
        print(self.__dict__)
        return self.__dict__
    
    
    def initializePop(self):
        '''
        Randomly generate the initial population.
        '''
        index_real = self.var_index["real"].astype(int)
        index_int = np.concatenate((self.var_index["int"], self.var_index["cate"])).astype(int)
        pop_size = self.par['population_size']
        dim = self.dim
        var_bound = self.var_bound
        
        ## Create empty arrays
        self.pop = np.array([np.zeros(dim + 1)]*pop_size) # +1 for storing obj
        self.var = np.zeros(dim)       
        
        ## Randomly generate the initial variables set for members in the population
        for p in range(0, pop_size):
            for i in index_int:
                self.var[i] = np.random.randint(var_bound[i][0], var_bound[i][1]+1)  
            for i in index_real:
                self.var[i] = var_bound[i][0] + np.random.random()*(var_bound[i][1] - var_bound[i][0])    
            self.pop[p,:dim] = self.var
            self.pop[p, dim] = np.nan       # no obj yet
      
        return None  
    
    def simPop(self, initialRun = False):  
        '''
        Simulate the whole population.

        '''
        pop = self.pop.copy()
        if initialRun:
            parent_size = 0
        else:
            parent_size = self.par['parent_size']
        pop_size = self.par['population_size']
        dim = self.dim
        maxIter = self.par['max_num_iteration']
        currentIter = self.iter
        saveGADataPerIter = self.saveGADataPerIter
        funtimeout = self.funtimeout
        function = self.func
        
        # Define sim function for non-parallel but with timeout
        def sim0(X):
            def evaluation():   # In order to use func_timeout
                return function(X)
            obj = None
            try:
                obj = func_timeout(funtimeout, evaluation)
            except FunctionTimedOut:
                print("given function is not applicable")
            assert (obj!=None), self.logger.error("FunctionTimedOut After {} seconds delay, the given function does not provide any output.".format(str(funtimeout)) )
            return obj
        
        def sim1(X):    # Parallel (need to add logger inside this function and cannot use imported function)
            obj = None
            try:
                obj = function(X)
            except:
                print("FunctionError given function is not applicable.") # Will not be printed out.
            return obj
        
        def sim2(X, WD, Iteration, member):  # Only for riverware coupling purpose.
            # Assigned copied subfolder name
            subfolderName = os.path.join(WD,"Iter{}_{}".format(Iteration, member)) 
            obj = None
            try:
                obj = function(X, subfolderName)
            except FunctionTimedOut:
                 print("FunctionError given function is not applicable.") # Will not be printed out.
            return obj
        
        # Parallel 0: Simple for loop. No parallelization 
        if self.parallel == 0:
            for k in tqdm(range(parent_size, pop_size, 1), desc = "Iter {}/{}".format(currentIter, maxIter)):
                obj = sim0(pop[k, :dim])
                pop[k, dim] = obj
        
        # Parallel 1: User defined function is run in parallel. Only use this when no working folder is needed.
        elif self.parallel == 1:
            self.NumCore = multiprocessing.cpu_count()  # or type -1       # Max cores number
            self.logger.info("Iter {}/{} Start parallel simulation with {} cores.".format(currentIter, maxIter, self.NumCore))
            ParallelResults = Parallel(n_jobs = self.NumCore, prefer="threads", timeout=funtimeout)(delayed(sim1)(X=pop[k, :dim]) for k in range(parent_size, pop_size, 1)) 
            # Collect results
            for k in range(parent_size, pop_size, 1):
                pop[k, dim] = ParallelResults[k - parent_size]  
        
        # Parallel 2: User defined function is run in parallel with assigned sub-working folder name. User can copy the necessary files into this folder and run the simulation in the isolated environment.
        elif self.parallel == 2:
            SubfolderPath = self.SubfolderPath
            self.NumCore = multiprocessing.cpu_count()  # or type -1       # Max cores number
            self.logger.info("Iter {}/{} Start parallel simulation (subfolder) with {} cores.".format(currentIter, maxIter, self.NumCore))
            ParallelResults = Parallel(n_jobs = self.NumCore, prefer="threads", timeout=funtimeout)(delayed(sim2)(X=pop[k, :dim], WD=SubfolderPath, Iteration=currentIter, member=k) for k in range(parent_size, pop_size, 1)) 
            # Collect results
            for k in range(parent_size, pop_size, 1):
                pop[k, dim] = ParallelResults[k - parent_size] 


        # Sorted by obj (last index) to an order of low obj (good) to high obj (bad).       
        pop = pop[pop[:, dim].argsort()]                    
        self.pop = pop
        
        # Save current iteration in case program crush.
        # If crush down reload the saved pickle file and continue the run.
        if saveGADataPerIter:
            self.save_attribution(self.WD)
        self.logger.info("Iter {}/{} done.".format(currentIter, maxIter))
        return None
        

    def runGA(self, plot = True):
        # Start timing
        self.start_time = datetime.now()

        # Initial Population (if it is to continue from last run with given pickle file, this step will be skipped.)
        if self.continue_file is None:
            self.initializePop()            # Randomly generate self.pop 
            self.simPop(initialRun = True)  # Calculate obj for members in self.pop
        
        # Store the best var and obj
        dim = self.dim
        self.best_minobj = self.pop[0, dim].copy()
        self.best_var = self.pop[0, :dim].copy()
        self.report.append(self.best_minobj)        # record the history obj
        
        # Start the while loop for evolution
        pop_size = self.par['population_size']
        parent_size = self.par['parent_size']
        num_elit = self.par['num_elit']
        maxIter = self.par['max_num_iteration']
        mniwi = self.par['max_iteration_without_improv']
        prob_cross = self.par['crossover_probability']
        cross_type = self.par['crossover_type']
     
        self.iter += 1          # Iteration (generation of the population)
        mniwi_counter = 0       # max_iteration_without_improv
        
        while self.iter <= maxIter and mniwi_counter <= mniwi:
            pop = self.pop.copy()
            # Normalizing objective function for calculating prob
            normobj = np.zeros(pop_size)
            minobj = pop[0, dim]
            if minobj < 0:      # to nonnegative values
                normobj = pop[:, dim] + abs(minobj)
            else:
                normobj = pop[:, dim]
            maxnorm = np.amax(normobj)
            normobj = maxnorm-normobj + 1     # The lowest obj has highest fitness. +1 to avoid 0.
        
            # Calculate probability
            sum_normobj = np.sum(normobj)
            prob = np.zeros(pop_size)
            prob = normobj/sum_normobj
            cumprob = np.cumsum(prob)
        
             # Select parents
            parents = np.array([np.zeros(dim + 1)]*parent_size) # Create empty parents
            ## First fill with elites
            for k in range(0, num_elit):
                parents[k] = pop[k].copy()
            ## Then fill the rest by wheel withdrawing.
            for k in range(num_elit, parent_size):
                index = np.searchsorted(cumprob,np.random.random())
                parents[k] = pop[index].copy()
            ## From the selected parents, we further randomly choose those who actually reproduce offsprings
            ef_par_list = np.array([False]*parent_size)
            par_count = 0
            while par_count == 0:   # has to at least 1 parents generate be selected
                for k in range(0, parent_size):
                    if np.random.random() <= prob_cross:
                        ef_par_list[k] = True
                        par_count += 1
            ## Effective parents
            ef_parents = parents[ef_par_list].copy()    
            
            # New generation
            pop = np.array([np.zeros(dim + 1)]*pop_size) # Create empty new gen pop
            ## First, fill with those selected parents without any modification
            for k in range(0, parent_size):
                pop[k] = parents[k].copy()
            ## Then, fill the rest with crossover and mutation process
            for k in range(parent_size, pop_size, 2):
                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                parent_var1 = ef_parents[r1, :dim].copy()
                parent_var2 = ef_parents[r2, :dim].copy()
                # Crossover
                children = self.cross(parent_var1, parent_var2, cross_type)
                child1 = children[0].copy()
                child2 = children[1].copy()
                # Mutation
                child1 = self.mut(child1)   # re-generate vars
                child2 = self.mutmiddle(child2, parent_var1, parent_var2)  # re-generate within parents range except cate type var             
                ## Only copy the variables. We haven't calculate obj
                pop[k, :dim] = child1.copy()    # Assign var
                pop[k, dim] = np.nan            # No obj yet
                pop[k+1, :dim] = child2.copy() 
                pop[k+1, dim] = np.nan  
            self.pop = pop      # Assign new population ready for simulation.
            
            # Calculate objs for pop
            self.simPop()     # Here is the safe point if WD is assigned and saveGADataPerIter = True   
            if pop[0, dim] >= self.best_minobj:
                mniwi_counter += 1
                self.report.append(self.best_minobj)  
                if mniwi_counter > mniwi:
                    self.logger.warning("Reach the max_iteration_without_improv. GA stop.")
            else:
                self.best_minobj = self.pop[0, dim].copy()
                self.best_var = self.pop[0, :dim].copy()
                self.report.append(self.best_minobj)        # record the history obj
            
            # Log current result
            current_result = {'Variable': self.best_var, 
                              'Objective': self.best_minobj,
                              'Improve rate': (self.report[-1] - self.report[-2])/self.report[-2],
                              'Duration': datetime.now() - self.start_time}
            self.logger.info("\n=============> Results (Iter {}) <=============\n".format(self.iter) + "\n".join(['{:^15} :  {}'.format(keys, values) for keys,values in current_result.items()]) )
            # Next iteration
            self.iter += 1    # Iteration (generation of the population)    
            # End while
        
        # Final report
        self.end_time = datetime.now()
        self.duration = self.end_time - self.start_time
        self.output_dict = {'Variable': self.best_var, 
                            'Objective': self.best_minobj,
                            'Duration': self.duration,
                            'Iteration': self.iter}
        self.logger.info("\n=============> Results <=============\n" + "\n".join(['{:^15} :  {}'.format(keys, values) for keys,values in self.output_dict.items()]) )
        
        
        # Remove the created file handler.
        if self.CreateFileHandler:    
            self.logger = RemoveLocalLogFile(self.logger, self.fh)
        
        if plot:
            re = np.array(self.report)
            fig, ax = plt.subplots()
            ax.plot(re)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Objective function (min)')
            ax.set_title('Genetic Algorithm')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
            string = "Min objective: {}\nDuration: {}\nIteration: {}".format(round(self.best_minobj, 3), self.duration, self.iter)
            ax.annotate(string, xy= (0.6, 0.95), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, fontsize=9, bbox = props)
        return None
    
    
    def cross(self, x, y, cross_type):
        ofs1 = x.copy()
        ofs2 = y.copy()
        dim = self.dim
        
        if cross_type=='one_point':
            rnd = np.random.randint(0, dim)
            for i in range(0,rnd):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()
  
        if cross_type=='two_point':
            ran1 = np.random.randint(0, dim)
            ran2 = np.random.randint(ran1, dim)
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if cross_type=='uniform':
            for i in range(0, dim):
                rnd = np.random.random()
                if rnd <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                   
        return np.array([ofs1,ofs2])
 
    
    def mut(self, x):
        prob_mut = self.par['mutation_probability']
        
        index_real = self.var_index["real"].astype(int)
        index_int = np.concatenate((self.var_index["int"], self.var_index["cate"])).astype(int)
        
        for i in index_int:
            rnd = np.random.random()
            if rnd < prob_mut:
                
                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        for i in index_real:                
            rnd = np.random.random()
            if rnd < prob_mut:   

               x[i]=self.var_bound[i][0]+np.random.random()*(self.var_bound[i][1]-self.var_bound[i][0])              
        return x


    def mutmiddle(self, x, p1, p2):
        prob_mut = self.par['mutation_probability']
        index_real = self.var_index["real"].astype(int)
        index_int = self.var_index["int"].astype(int)
        index_cate = self.var_index["cate"].astype(int)
        
        for i in index_int:
            rnd = np.random.random()
            if rnd < prob_mut:
                if p1[i] < p2[i]:
                    x[i] = np.random.randint(p1[i],p2[i])
                elif p1[i] > p2[i]:
                    x[i] = np.random.randint(p2[i],p1[i])
                else:
                    x[i] = np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in index_cate:  # mutmiddle() is not appliable.
            rnd = np.random.random()
            if rnd < prob_mut:
                x[i] = np.random.randint(self.var_bound[i][0],\
                self.var_bound[i][1]+1)
                    
        for i in index_real:                           
            rnd = np.random.random()
            if rnd < prob_mut:   
                if p1[i] < p2[i]:
                    x[i] = p1[i]+np.random.random()*(p2[i] - p1[i])  
                elif p1[i] > p2[i]:
                    x[i] = p2[i] + np.random.random()*(p1[i] - p2[i])
                else:
                    x[i] = self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1] - self.var_bound[i][0]) 
        return x
        
##############################################################################
##############################################################################    

class GADataConverter(object):
    def __init__(self, Msglevel = None):
        # Set loggar
        self.logger = logging.getLogger(__name__)
        if Msglevel is None: Msglevel = logging.INFO
        self.logger.setLevel(Msglevel)
        
        self.orgpar_convert = False # To check the original data has been converted to var before convert var back.
        return None
    
    
    def Covert2GAArray(self, DataList, order = "C"):
    # For now this function is able to convert a list of 1d or 2d array or df to 1d array. In the future this can be modified to support higher dimension.
    # order{'C', ‘F’, ‘A’} 
    #‘C’ means to flatten in row-major (C-style) order. 
    #‘F’ means to flatten in column-major (Fortran- style) order. 
    #‘A’ means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.
    
        assert isinstance(DataList, list), self.logger.error("DataList needs to be a list.")
        for item in DataList:
            assert isinstance(item, (np.ndarray, pd.DataFrame)), self.logger.error("Elements in the DataList have to be either array or dataframe.")

        self.orgpar_shape = []
        self.orgpar_type = {}
        self.orgpar_order = order
        self.orgpar_convert = True
        self.orgpar_index = [0]
        var = []
        
        for i, data in enumerate(DataList):
            if len(data.shape) == 2:
                if isinstance(data, pd.DataFrame):
                    self.orgpar_shape.append(data.shape)
                    self.orgpar_type[i] = {}
                    self.orgpar_type[i]["col"] = list(data.columns)
                    self.orgpar_type[i]["ind"] = list(data.index)
                    var = var + list( data.to_numpy().flatten(order) )
                    self.orgpar_index.append(self.orgpar_index[-1] + self.orgpar_shape[-1][0]*self.orgpar_shape[-1][1])
                elif isinstance(data, np.ndarray):
                    self.orgpar_shape.append(data.shape)
                    self.orgpar_type[i] = np.ndarray
                    var = var + list( data.flatten(order) )
                    self.orgpar_index.append(self.orgpar_index[-1] + self.orgpar_shape[-1][0]*self.orgpar_shape[-1][1])
                else:
                    print("error")
            elif len(data.shape) == 1:
                self.orgpar_shape.append(data.shape)
                self.orgpar_type[i] = "1d"
                var = var + list( data.flatten(order) )
                self.orgpar_index.append(self.orgpar_index[-1] + len(data))
        return var

    def GAArray2OrgPar(self, var, setting = None):
        if setting is None:
            assert self.orgpar_convert, self.logger.error("ValueError The function Covert2GAArray() has to be exercuted first or provide setting dictionary.")
            orgpar_type = self.orgpar_type
            order = self.orgpar_order
            orgpar_index = self.orgpar_index
            orgpar_shape = self.orgpar_shape
        else:
            assert set(["orgpar_type","orgpar_order","orgpar_index","orgpar_shape"]).issubset(setting.keys()), self.logger.error("KeyError Setting dictionary has to contain keys: {}".format(["orgpar_type","orgpar_order","orgpar_index","orgpar_shape"]))
            orgpar_type = setting["orgpar_type"]
            order = setting["orgpar_order"]
            orgpar_index = setting["orgpar_index"]
            orgpar_shape = setting["orgpar_shape"]
            
        self.orgParList = []
        for i, v in orgpar_type.items():
            if isinstance(v, dict):
                df = np.reshape(var[orgpar_index[i]:orgpar_index[i+1]], orgpar_shape[i], order)
                df = pd.DataFrame(df)
                df.columns = v['col']
                df.index = v['ind']
                self.orgParList.append(df)
            elif v == np.ndarray:
                self.orgParList.append(np.reshape(var[orgpar_index[i]:orgpar_index[i+1]], orgpar_shape[i], order))
            elif v == "1d":
                self.orgParList.append(list(var[orgpar_index[i]:orgpar_index[i+1]]))
        return self.orgParList
    
    def outputSetting(self):
        Setting = {"orgpar_type": self.orgpar_type,
                   "orgpar_order": self.orgpar_order,
                   "orgpar_index": self.orgpar_index,
                   "orgpar_shape": self.orgpar_shape}
        return Setting
    
#%%
# Demo code
# import numpy as np
# from PyRAMID import RwABM


# def f(var, GA_WD):
#     print(GA_WD)
#     return np.sum(var)


# var_bound = np.array([[0,10]]*3)
# model = RwABM.GeneticAlgorithm(function=f,dimension=3, WD = r"C:\Users\ResearchPC\Desktop",parallel = 2,
#             saveGADataPerIter = True, variable_type='real',variable_boundaries=var_bound, Msglevel = 'info')
# model.runGA()


# #%%
# import numpy as np
# from PyRAMID import RWABM
# model2 = RWABM.GeneticAlgorithm(continue_file =  r"C:\Users\ResearchPC\Desktop\GAobject.pickle")
# model2.runGA()
