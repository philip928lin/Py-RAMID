# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:31:32 2020

@author: CYLin
"""
import pandas as pd
import numpy as np
from datetime import datetime
from func_timeout import func_timeout, FunctionTimedOut
from joblib import Parallel, delayed, parallel_backend
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import pickle
import os
from inspect import signature
from pyramid.setting import (ConsoleLogParm, MsglevelDict, 
                             addLocalLogFile, removeLocalLogFile)

class GeneticAlgorithm(object):
    """ GeneticAlgorithm with parallel in computing."""
    def __init__(self, function=lambda x: 0,
                 dimension=None,
                 variable_type='bool',
                 variable_boundaries=None,
                 variable_type_mixed=None,
                 wd=None,
                 saveGADataPerIter=False,
                 function_timeout=1000,
                 parallel=0,
                 threads=None,
                 algorithm_parameters={'max_num_iteration': None,
                                       'population_size':100,
                                       'mutation_probability':0.1,
                                       'elit_ratio': 0.01,
                                       'crossover_probability': 0.5,
                                       'parents_portion': 0.3,
                                       'crossover_type':'uniform',
                                       'max_iter_without_improv': None},
                 continue_file=None,
                 seed=None,
                 msg_level=None):
        """
        Parameters
        ----------
        function : Callable function with input argument <var> or 
            <var, GA_WD> if parallel = 2. var is a 1-D array. 
            GA_WD is subfolder path.
        dimension : int, The dimension of calibrated parameters.
        variable_type : 'bool', 'int', 'real', 'cate'. 
            The default is 'bool'.
        variable_boundaries : A list of boundary for each parameter in  
            the format of [upper bound, lower bound].
        variable_type_mixed : None, True. If True, corresponding
            variable_type and variable_boundaries needs to be given.
        wd : Needs to be given if saveGADataPerIter is Ture or 
            parallel = 2.
        saveGADataPerIter : True, False. If True, auto-save per 
            iteration will be opened. The saved GAobject.pickle could be
            used later to continue the previous interupted run. We
            highly recommend to provide wd and turn on this option. The 
            default is False.
        function_timeout : Maximum seconds for the simulation for each 
            member. The default is 1000.
        parallel : 0, 1, 2. 0: no parallel. 1: parallel without creating 
                   sub-working folders. 2: parallel with creating 
                   sub-working folders. The default is 0.
        threads : Number of threads to be used in parallel. -1: Max, 
                  -2: Max-1. The default is None.
        algorithm_parameters : dict. The default is 
            {'max_num_iteration': None,
            'population_size':100,
            'mutation_probability':0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type':'uniform',
            'max_iter_without_improv': None}.
        continue_file : Assign the path of GAobject.pickle to continue 
            the simulation. The default is None.
        seed : Random seed for random number generator.
        msg_level : 'debug', 'info', 'warning', 'error'. 
            Level of print out message. The default is info 
            (ConsoleLogParm['Msglevel']).
        """
        
        self.__name__ = "GA"
        ################################################################
        # Setup the log msg (console) (log file is added below.)
        self.logger = logging.getLogger(__name__)
        if msg_level is None: msg_level = ConsoleLogParm['Msglevel']
        else:
            assert msg_level in ['debug', 'info', 'warning', 'error'],\
                print("ValueError msg_level must be one of these "+\
                      "[None, 'debug', 'info', 'warning', 'error'].")
            msg_level = MsglevelDict[msg_level]
        self.logger.setLevel(msg_level)   
        self.CreateFileHandler = False
        
        ################################################################
        # Setup input parameter
        if continue_file is not None:
            # Load the GAobject.pickle to continue previous run.
            assert os.path.exists(continue_file),\
                self.logger.error("PathError given continue_file is not "+\
                                  "exist {}.".format(continue_file))
            self.continue_file = continue_file
            self.load_continue_file()
            # Load file will overwrite self.continue_file, so we need to
            # assign it again.
            self.continue_file = continue_file 
            # Re-assign fh with mode = "a", appending to the previous 
            # GA.log.
            self.logger, self.fh = addLocalLogFile('GA.log', self.logger,\
                                                   self.wd, mode = "a") 
            self.logger.info("\n========== Continue ==========\n")
            # Set random seed
            if self.seed is not None:
                np.random.seed(self.seed)
        else:
            # Check all input settings are valid.
            ############################################################
            # Check wd and add GA.log if wd is given.
            self.saveGADataPerIter = saveGADataPerIter
            if wd is None:
                self.wd = None
                if saveGADataPerIter:
                    self.logger.error("ValueError To enable "+\
                                      "saveGADataPerIter and log file, "+\
                                      "valid wd must be given.")
                    self.saveGADataPerIter = False
            else:
                assert os.path.isdir(wd),\
                    self.logger.error("PathError given wd is not exist {}."\
                                      .format(wd))
                self.wd = wd
                # Add local log file
                self.logger, self.fh = addLocalLogFile('GA.log', self.logger,\
                                                       self.wd)
                self.CreateFileHandler = True
            ############################################################
            # Create output related attributions
            self.pop = None      
            self.best_var = None
            self.best_minobj = None
            self.report = []
            self.iter = 0
            self.pop_record = {}
        
            ############################################################
            # Check inputs
            # Dimension
            assert isinstance(dimension, (float, int)),\
                self.logger.error("TypeError dimension must be integer.")
            # Variable types
            assert(variable_type=='bool' or variable_type=='int' or \
                   variable_type=='real' or variable_type=='cate'), \
                   self.logger.error("TypeError variable_type must be "+\
                                     "'bool', 'int', 'real', or 'cate'.")
            # parallel options
            assert parallel in [0,1,2],\
                self.logger.error("TypeError parallel must be "+\
                                  "0: no parallel, 1: parallel, "+\
                                  "2: parallel with new sub-working folders.")
            # function
            assert (callable(function)),\
                self.logger.error("TypeError function must be callable.")
            # function arguments
            if parallel == 2:
                assert [i for i in signature(function).parameters] == \
                    ['var', 'GA_WD'],  self.logger.error("ValueError To "+\
                    "run GA for parallel = 2 (coupling), given sim function "+\
                    "has to contain two input arguments: 'var' (1d array) "+\
                    "and GA_WD, which user should use "+\
                    "RiverwareWrap.createFiles(GA_WD) to create subfolder "+\
                    "in their sim function and conduct the simulation under "+\
                    "this new directory.")
                self.SubfolderPath = os.path.join(self.wd, "AutoCalibration")
                if os.path.isdir(self.SubfolderPath) is not True:
                    os.mkdir(self.SubfolderPath)
                    self.logger.info("Create subfolder AutoCalibration at {}"\
                                     .format(self.SubfolderPath))
            # Check and assign threads input
            if parallel != 0:
                # Max threads number/2 
                MaxThreads = int(os.cpu_count()/2)
                if threads is None or threads > MaxThreads:
                    self.NumThreads = MaxThreads
                elif threads < 0: # -1: = MaxThreads
                    self.NumThreads = MaxThreads + 1 + threads
                else:
                    self.NumThreads = threads
                self.NumThreads = int(self.NumThreads)
            else:
                self.NumThreads = 1
            # Check random seed
            assert isinstance(seed, (type(None), int)),\
                self.logger.error("TypeError seed must be integer or None.")
            ############################################################
            # Assign input
            self.dim = int(dimension)
            self.func = function
            self.parallel = parallel
            self.var_index = {}
            if function_timeout is None:
                function_timeout = 86400 # If None, we set timeout=1day
            self.funtimeout = int(function_timeout)
            self.continue_file = continue_file
            self.seed = seed
            ############################################################
            # Set random seed
            if self.seed is not None:
                np.random.seed(self.seed)
                
            # Assign var_type and var_bound and var_index
            if variable_type_mixed is None: 
                # We assign identical type according to variable_type to  
                # each variable.
                if variable_type == 'real': 
                    self.var_type = np.array([['real']]*self.dim)
                    self.var_index["cate"] = np.array([])
                    self.var_index["int"] = np.array([])
                    self.var_index["real"] = \
                        np.where(self.var_type == 'real')[0]
                else:   # 'int', 'bool', 'cate'
                    self.var_type = np.array([['int']]*self.dim)    
                    if variable_type == 'cate':
                        self.var_index["cate"] = \
                            np.where(self.var_type == 'int')[0]
                        self.var_index["int"] = np.array([])
                        self.var_index["real"] = np.array([])
                    else:
                        self.var_index["cate"] = np.array([])
                        self.var_index["int"] = \
                            np.where(self.var_type == 'int')[0]
                        self.var_index["real"] = np.array([])
                        
                # Assign var_bound if it is not given
                if variable_boundaries is None:
                    self.var_bound = np.array([[0,1]]*self.dim)
                else:
                    assert isinstance(variable_boundaries, (list,np.ndarray)),\
                        self.logger.error("TypeError variable_boundaries "+\
                                          "must be numpy array or list.") 
                    variable_boundaries = np.array(variable_boundaries) 
                    assert (variable_boundaries.shape == (self.dim,2)),\
                        self.logger.error("ValueError variable_type_mixed "+\
                                          "must have a shape (dimension, 2).") 
                    self.var_bound = variable_boundaries
            else: 
                # var types should be defined in variable_type_mixed
                assert isinstance(variable_type_mixed, (list, np.ndarray)),\
                    self.logger.error("TypeError variable_type_mixed must "+\
                                      "be numpy array or list.") 
                assert isinstance(variable_boundaries, (list, np.ndarray)),\
                    self.logger.error("TypeError variable_boundaries must "+\
                                      "be numpy array or list.")  
                variable_type_mixed = np.array(variable_type_mixed)
                variable_boundaries = np.array(variable_boundaries)            
                assert (len(variable_type_mixed) == self.dim),\
                    self.logger.error("ValueError variable_type_mixed must "+\
                                      "have a length equal dimension.") 
                assert (variable_boundaries.shape == (self.dim,2)),\
                    self.logger.error("ValueError variable_type_mixed must "+\
                                      "have a shape (dimension, 2).") 
                self.var_type = variable_type_mixed
                self.var_bound = variable_boundaries
                self.var_index["cate"] = np.where(self.var_type == 'cate')[0]
                self.var_index["int"] = np.where(self.var_type == 'int')[0]
                self.var_index["real"] = np.where(self.var_type == 'real')[0]
                # Replace cate as int for rest of the calculation
                self.var_type = \
                    np.where(self.var_type=='cate', 'int', self.var_type) 
                
            ############################################################
            # Check algorithm_parameters
            assert set(['max_num_iteration', 'population_size',\
                        'mutation_probability', 'elit_ratio', \
                        'crossover_probability', 'parents_portion', \
                        'crossover_type', 'max_iter_without_improv'])\
                        .issubset(set(algorithm_parameters.keys())),\
                self.logger.error("KeyError Missing keys in the "+\
                                  "algorithm_parameters.")
            self.par = algorithm_parameters 
            self.par['population_size'] = int(self.par['population_size'])
            assert (self.par['parents_portion'] <= 1 and \
                    self.par['parents_portion'] >= 0), \
                self.logger.error("ValueError parents_portion must be in "+\
                                  "range [0,1].")
            assert (self.par['mutation_probability'] <= 1 and \
                    self.par['mutation_probability'] >= 0), \
                self.logger.error("ValueError mutation_probability must be "+\
                                  "in range [0,1].")
            assert (self.par['crossover_probability'] <= 1 and \
                    self.par['crossover_probability'] >= 0), \
                self.logger.error("ValueError crossover_probability must be "+\
                                  "in range [0,1].")
            assert (self.par['elit_ratio'] <= 1 and \
                    self.par['elit_ratio'] >= 0), \
                self.logger.error("ValueError elit_ratio must be in "+\
                                  "range [0,1].")
            assert (self.par['mutation_probability'] <= 1 and \
                    self.par['mutation_probability'] >= 0), \
                self.logger.error("ValueError mutation_probability must be "+\
                                  "in range [0,1].")
            assert (self.par['crossover_type'] == 'uniform' or \
                    self.par['crossover_type'] == 'one_point' or \
                    self.par['crossover_type'] == 'two_point'), \
                self.logger.error("ValueError crossover_type must be "+\
                                  "'uniform', 'one_point', or 'two_point'")
            
            # Make sure that population_size is properly assigned
            self.par['parent_size'] = int(self.par['parents_portion']\
                                          *self.par['population_size'] )
            trl = self.par['population_size'] - self.par['parent_size']
            if trl % 2 != 0: 
                self.par['parent_size'] += 1  # To guarentee even number 
            
            # Make sure that num_elit is properly assigned
            trl = self.par['population_size']*self.par['elit_ratio']
            # At least 1 elite
            if trl < 1 and self.par['elit_ratio'] > 0:  
                self.par['num_elit'] = 1
            else:
                self.par['num_elit'] = int(trl)     # Round down
            
            # Make sure that max_num_iteration is properly assigned
            if self.par['max_num_iteration'] is None:
                self.par['max_num_iteration'] = 0
                for i in range (0, self.dim):
                    if self.var_type[i] == 'int':
                        self.par['max_num_iteration'] += \
                            (self.var_bound[i][1] - self.var_bound[i][0]) \
                                *self.dim*(100/self.par['population_size'])
                    else:
                        self.par['max_num_iteration'] += \
                            (self.var_bound[i][1]-self.var_bound[i][0]) \
                                *50*(100/self.par['population_size'])
                self.par['max_num_iteration'] = \
                    int(self.par['max_num_iteration'])
                if (self.par['max_num_iteration'] \
                    *self.par['population_size']) > 10000000:
                    self.par['max_num_iteration'] = \
                        10000000/self.par['population_size']
            else:
                self.par['max_num_iteration'] = \
                    int(self.par['max_num_iteration'])
                
            # Make sure that max_num_iteration is properly assigned    
            if self.par['max_iter_without_improv'] == None:
                self.par['max_iter_without_improv'] = \
                    self.par['max_num_iteration'] + 1
            else:
                self.par['max_iter_without_improv'] = \
                    int(self.par['max_iter_without_improv'])      
            
            # Print out the summary of GA object settings.
            self.logger.info("The GA object have been initiated: \n"+"\n" \
                             .join(['{:^23} :  {}'.format(keys, values) for \
                                    keys,values in self.par.items()]))
        return None
    
    def load_continue_file(self):
        """Load GAobject.pickle """
        filepath = self.continue_file
        with open(filepath, "rb") as f:
            dictionary = pickle.load(f)
        #print(dictionary)
        # Load back all the previous class attributions.
        for key in dictionary:
            setattr(self, key, dictionary[key])
        self.logger.info("The previous GA object have been loaded back "+\
                         "and ready to run.")
        
    def save_attribution(self, path):
        """Save GAobject.pickle """
        dictionary = self.__dict__.copy()
        dictionary.pop('fh', None)  # handler cannot be pickled.
        dictionary.pop('logger', None)  # handler cannot be pickled.
        with open(os.path.join(path, "GAobject.pickle"), 'wb') as outfile:
            pickle.dump(dictionary, outfile)
        
    def Print(self):
        """Turn the attributions of GA object into dictionary."""
        print(self.__dict__)
        return self.__dict__
    
    
    def initializePop(self, InitialPop=None):
        """Randomly generate the initial population."""
        # If user provide their own InitialPop, then we don't generate 
        # ini pop.
        if InitialPop is not None:
            self.pop = InitialPop
            return None
        
        index_real = self.var_index["real"].astype(int)
        index_int = np.concatenate((self.var_index["int"], \
                                    self.var_index["cate"])).astype(int)
        pop_size = self.par['population_size']
        dim = self.dim
        var_bound = self.var_bound
        
        ## Create empty arrays
        self.pop = np.array([np.zeros(dim + 1)]*pop_size) # +1 for storing obj
        self.var = np.zeros(dim)       
        
        ## Randomly generate the initial variables set for members in the pop.
        for p in range(0, pop_size):
            for i in index_int:
                self.var[i] = np.random.randint(var_bound[i][0], \
                                                var_bound[i][1]+1)  
            for i in index_real:
                self.var[i] = var_bound[i][0] + np.random.random()* \
                              (var_bound[i][1] - var_bound[i][0])    
            self.pop[p,:dim] = self.var
            self.pop[p, dim] = np.nan       # no obj yet
      
        return None  
    
    def simPop(self, initialRun=False):  
        """Simulate the whole population."""
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
        
        def sim0(X):
            """for loop """
            def evaluation():   # In order to use func_timeout
                return function(X)
            obj = None
            try:
                obj = func_timeout(funtimeout, evaluation)
            except FunctionTimedOut:
                print("given function is not applicable")
            assert (obj!=None), \
                self.logger.error("FunctionTimedOut After {} seconds delay, "\
                                      .format(str(funtimeout)) + \
                                  "the given function does not provide any "+\
                                  "output.")
            return obj
        
        def sim1(X):    
            """Parallel without creating subfolder"""
            obj = None
            try:
                obj = function(X)
            except:
                # Will not be printed out. (Run in backend)
                print("FunctionError given function is not applicable.") 
            return obj
        
        # For riverware coupling model.
        def sim2(X, WD, Iteration, member):  
            """Parallel with assigned copied subfolder path"""
            SubFolderName = os.path.join(WD,"Iter{}_{}"\
                                         .format(Iteration, member)) 
            obj = None
            try:
                obj = function(X, SubFolderName)
            except FunctionTimedOut:
                # Will not be printed out. (Run in backend)
                print("FunctionError given function is not applicable.") 
            return obj
        
        ################################################################
        # Parallel 0: Simple for loop. No parallelization 
        if self.parallel == 0:
            for k in tqdm(range(parent_size, pop_size, 1),\
                          desc = "Iter {}/{}".format(currentIter, maxIter)):
                obj = sim0(pop[k, :dim])
                pop[k, dim] = obj
        
        # Parallel 1: User defined function is run in parallel.
        # Only use this when no working folder is needed.
        elif self.parallel == 1:
            self.logger.info("Iter {}/{} Start parallel simulation with {} "\
                             .format(currentIter, maxIter, self.NumThreads)+\
                             "threads.")
            ParallelResults = Parallel(n_jobs = self.NumThreads, \
                                       prefer="threads", \
                                       timeout=funtimeout)\
                                      (delayed(sim1)(X=pop[k, :dim]) \
                                      for k in range(parent_size, pop_size, 1)) 
            # Collect results
            for k in range(parent_size, pop_size, 1):
                pop[k, dim] = ParallelResults[k - parent_size]  
        
        # Parallel 2: User defined function is run in parallel with
        # assigned sub-working folder name. User can copy the necessary
        # files into this folder and run the simulation in the isolated
        # environment.
        elif self.parallel == 2:
            SubfolderPath = self.SubfolderPath
            self.logger.info("Iter {}/{} Start parallel simulation with {} "\
                             .format(currentIter, maxIter, self.NumThreads)+\
                             "threads.")
            
            ParallelResults = Parallel(
                n_jobs = self.NumThreads, prefer="threads",
                timeout=funtimeout)\
                (delayed(sim2)(X=pop[k, :dim], wd=SubfolderPath, 
                               Iteration=currentIter, member=k) \
                for k in range(parent_size, pop_size, 1))
            
            # Collect results
            for k in range(parent_size, pop_size, 1):
                pop[k, dim] = ParallelResults[k - parent_size] 


        # Sorted by obj (last index) to an order of low obj (good) to
        # high obj (bad).       
        pop = pop[pop[:, dim].argsort()]                    
        self.pop = pop
        
        # Save current iteration in case program crush.
        # If crush down reload the saved pickle file and continue the 
        # run.
        if saveGADataPerIter:
            self.save_attribution(self.wd)
        self.logger.info("Iter {}/{} done.".format(currentIter, maxIter))
        return None
        

    def runGA(self, plot = True, InitialPop = None, start_from_iter = None):
        # Start timing
        self.start_time = datetime.now()

        # Initial Population (if it is to continue from last run with 
        # given pickle file, this step will be skipped.)
        if self.continue_file is None:
            self.mniwi_counter = 0      # max_iter_without_improv
            # Randomly generate self.pop 
            self.initializePop(InitialPop = InitialPop) 
            self.pop_record["Iter0"] = self.pop
            # Calculate obj for members in self.pop
            self.simPop(initialRun=True)
            self.pop_record["Iter0"] = self.pop
            
        ################################################################
        # Start from recorded specific iteration
        # So the GA will use this iteration as intial "result" to form
        # the next generation. Simulation happens at iteration + 1.
        if start_from_iter is not None and start_from_iter <= self.iter and \
            start_from_iter != 0:
            # Clean report and assign pop and iter
            dim = self.dim
            self.report = self.report[:start_from_iter] 
            self.pop = self.pop_record["Iter{}".format(start_from_iter)]
            self.iter = start_from_iter
            
        # Store the best var and obj
        dim = self.dim
        self.best_minobj = self.pop[0, dim].copy()
        self.best_var = self.pop[0, :dim].copy()
        self.report.append(self.best_minobj)    # record the history obj
        
        # Start the while loop for evolution
        pop_size = self.par['population_size']
        parent_size = self.par['parent_size']
        num_elit = self.par['num_elit']
        maxIter = self.par['max_num_iteration']
        mniwi = self.par['max_iter_without_improv']
        prob_cross = self.par['crossover_probability']
        cross_type = self.par['crossover_type']
     
        self.iter += 1        # Iteration (generation of the population)
        
        
        ################################################################
        while self.iter <= maxIter and self.mniwi_counter <= mniwi:
            pop = self.pop.copy()
            # Normalizing objective function for calculating prob
            normobj = np.zeros(pop_size)
            minobj = pop[0, dim]
            if minobj < 0:      # to nonnegative values
                normobj = pop[:, dim] + abs(minobj)
            else:
                normobj = pop[:, dim]
            maxnorm = np.amax(normobj)
            # The lowest obj has highest fitness. +1 to avoid 0.
            normobj = maxnorm-normobj + 1    
        
            # Calculate probability
            sum_normobj = np.sum(normobj)
            prob = np.zeros(pop_size)
            prob = normobj/sum_normobj
            cumprob = np.cumsum(prob)
        
            # Select parents
            ## Create empty parents
            parents = np.array([np.zeros(dim + 1)]*parent_size) 
            ## First fill with elites
            for k in range(0, num_elit):
                parents[k] = pop[k].copy()
            ## Then fill the rest by wheel withdrawing.
            for k in range(num_elit, parent_size):
                index = np.searchsorted(cumprob,np.random.random())
                parents[k] = pop[index].copy()
            ## From the selected parents, we further randomly choose 
            ## those who actually reproduce offsprings
            ef_par_list = np.array([False]*parent_size)
            par_count = 0
            # At least 1 parents generate be selected
            while par_count == 0:   
                for k in range(0, parent_size):
                    if np.random.random() <= prob_cross:
                        ef_par_list[k] = True
                        par_count += 1
            ## Effective parents
            ef_parents = parents[ef_par_list].copy()    
            
            # New generation
            ## Create empty new gen pop
            pop = np.array([np.zeros(dim + 1)]*pop_size) 
            ## First, fill with those selected parents without any 
            ## modification
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
                ## re-generate within parents range except cate type var             
                child2 = self.mutmiddle(child2, parent_var1, parent_var2)  
                ## Only copy the variables. We haven't calculate obj
                pop[k, :dim] = child1.copy()    # Assign var
                pop[k, dim] = np.nan            # No obj yet
                pop[k+1, :dim] = child2.copy() 
                pop[k+1, dim] = np.nan  
            self.pop = pop # Assign new population ready for simulation.
            self.pop_record["Iter{}".format(self.iter)] = self.pop
            
            # Calculate objs for pop
            # Here is the safe point if wd is assigned and
            # saveGADataPerIter = True   
            self.simPop()    # Will update self.pop
            self.pop_record["Iter{}".format(self.iter)] = self.pop
            if self.pop[0, dim] >= self.best_minobj:
                self.mniwi_counter += 1
                self.report.append(self.best_minobj)  
                if self.mniwi_counter > mniwi:
                    self.logger.warning("Reach the max_iter_without_improv. "+\
                                        "GA stop.")
            else:
                self.best_minobj = self.pop[0, dim].copy()
                self.best_var = self.pop[0, :dim].copy()
                # record the history obj
                self.report.append(self.best_minobj)
            
            self.end_time = datetime.now()
            self.duration = self.end_time - self.start_time
            # Log current result
            current_result = {'Variable': self.best_var, 
                              'Objective': self.best_minobj,
                              'Improve rate': (self.report[-1] - \
                                              self.report[-2])/self.report[-2],
                              'Duration': self.duration}
            self.logger.info("\n===========> Results (Iter {}) <===========\n"\
                             .format(self.iter) + \
                             "\n".join(['{:^15} :  {}'.format(keys, values) \
                             for keys,values in current_result.items()]) )
            self.logger.info("Obj records: {}\n".format(self.report))
            if plot and self.parallel == 2:
                self.plotReport()
            # Next iteration
            self.iter += 1    # Iteration (generation of the population)    
            # End while
        
        ################################################################
        # Final report
        self.end_time = datetime.now()
        self.duration = self.end_time - self.start_time
        self.output_dict = {'Variable': self.best_var, 
                            'Objective': self.best_minobj,
                            'Duration': self.duration,
                            'Iteration': self.iter}
        self.logger.info("\n=============> Results <=============\n" + \
                         "\n".join(['{:^15} :  {}'.format(keys, values)\
                          for keys,values in self.output_dict.items()]))
        self.output_dict["ObjRecords"] = self.report
        
        # Remove the created file handler.
        if self.CreateFileHandler:    
            self.logger = removeLocalLogFile(self.logger, self.fh)
        
        if plot:
            self.plotReport()
        return None
    
    def plotReport(self):
        re = np.array(self.report)
        fig, ax = plt.subplots()
        ax.plot(re)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective function (minization)')
        ax.set_title('Genetic Algorithm')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        string = "Min objective: {}\nDuration: {}\nIteration: {}" \
            .format(round(self.best_minobj, 3), self.duration, 
                    self.iter)
        ax.annotate(string, xy= (0.6, 0.95), xycoords='axes fraction',
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, fontsize=9, bbox = props)
        plt.savefig(os.path.join(self.wd, "GA_report.png"), dpi = 500)
            
    def cross(self, x, y, cross_type):
        ofs1 = x.copy()
        ofs2 = y.copy()
        dim = self.dim
        
        if cross_type == 'one_point':
            rnd = np.random.randint(0, dim)
            for i in range(0,rnd):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()
  
        if cross_type == 'two_point':
            ran1 = np.random.randint(0, dim)
            ran2 = np.random.randint(ran1, dim)
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if cross_type == 'uniform':
            for i in range(0, dim):
                rnd = np.random.random()
                if rnd <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                   
        return np.array([ofs1,ofs2])
 
    
    def mut(self, x):
        prob_mut = self.par['mutation_probability']
        
        index_real = self.var_index["real"].astype(int)
        index_int = np.concatenate((self.var_index["int"], \
                                    self.var_index["cate"])).astype(int)
        
        for i in index_int:
            rnd = np.random.random()
            if rnd < prob_mut:
                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        for i in index_real:                
            rnd = np.random.random()
            if rnd < prob_mut:   
               x[i]=self.var_bound[i][0]+np.random.random()* \
                   (self.var_bound[i][1]-self.var_bound[i][0])              
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
        
########################################################################
########################################################################   

class GADataConverter(object):
    def __init__(self, msg_level=None):
        # Set loggar
        self.logger = logging.getLogger(__name__)
        if msg_level is None: msg_level = logging.INFO
        self.logger.setLevel(msg_level)
        
         # To check the original data has been converted to var before convert 
         # var back.
        self.orgpar_convert = False
        return None
    
    
    def Covert2GAArray(self, dataList, order="C"):
        """Convert a list of 1d or 2d array or df to 1d array. 
        
        order: "C", "F", "A". The Default is "C".
        #"C" means to flatten in row-major (C-style) order. 
        #"F" means to flatten in column-major (Fortran- style) order. 
        #"A" means to flatten in column-major order if a is Fortran  
             contiguous in memory, row-major order otherwise.
        """
        assert isinstance(dataList, list),\
            self.logger.error("dataList needs to be a list.")
        for item in dataList:
            assert isinstance(item, (np.ndarray, pd.DataFrame)), \
                self.logger.error("Elements in the dataList have to be "+\
                                  "either array or dataframe.")

        self.orgpar_shape = []
        self.orgpar_type = {}
        self.orgpar_order = order
        self.orgpar_convert = True
        self.orgpar_index = [0]
        var = []
        
        for i, data in enumerate(dataList):
            if len(data.shape) == 2:
                if isinstance(data, pd.DataFrame):
                    self.orgpar_shape.append(data.shape)
                    self.orgpar_type[i] = {}
                    self.orgpar_type[i]["col"] = list(data.columns)
                    self.orgpar_type[i]["ind"] = list(data.index)
                    var = var + list( data.to_numpy().flatten(order) )
                    self.orgpar_index.append(self.orgpar_index[-1] + \
                                             self.orgpar_shape[-1][0]* \
                                             self.orgpar_shape[-1][1])
                elif isinstance(data, np.ndarray):
                    self.orgpar_shape.append(data.shape)
                    self.orgpar_type[i] = np.ndarray
                    var = var + list( data.flatten(order) )
                    self.orgpar_index.append(self.orgpar_index[-1] + \
                                             self.orgpar_shape[-1][0]* \
                                             self.orgpar_shape[-1][1])
                else:
                    print("error")
            elif len(data.shape) == 1:
                self.orgpar_shape.append(data.shape)
                self.orgpar_type[i] = "1d"
                var = var + list( data.flatten(order) )
                self.orgpar_index.append(self.orgpar_index[-1] + len(data))
        return var

    def GAArray2OrgPar(self, var, setting=None):
        """Convert 1_D array back to original dfs and arrays."""
        if setting is None:
            assert self.orgpar_convert, \
                self.logger.error("ValueError The function Covert2GAArray() "+\
                                  "has to be exercuted first or provide "+\
                                      "setting dictionary.")
            orgpar_type = self.orgpar_type
            order = self.orgpar_order
            orgpar_index = self.orgpar_index
            orgpar_shape = self.orgpar_shape
        else:
            assert set(["orgpar_type","orgpar_order","orgpar_index", \
                        "orgpar_shape"]).issubset(setting.keys()), \
                self.logger.error("KeyError Setting dictionary has to "+\
                                  "contain keys: {}".format(["orgpar_type",\
                                                             "orgpar_order",\
                                                             "orgpar_index",\
                                                             "orgpar_shape"]))
            orgpar_type = setting["orgpar_type"]
            order = setting["orgpar_order"]
            orgpar_index = setting["orgpar_index"]
            orgpar_shape = setting["orgpar_shape"]
            
        self.orgParList = []
        for i, v in orgpar_type.items():
            if isinstance(v, dict):
                df = np.reshape(var[orgpar_index[i]:orgpar_index[i+1]], \
                                orgpar_shape[i], order)
                df = pd.DataFrame(df)
                df.columns = v['col']
                df.index = v['ind']
                self.orgParList.append(df)
            elif v == np.ndarray:
                self.orgParList.append(np.reshape(\
                                      var[orgpar_index[i]:orgpar_index[i+1]],\
                                      orgpar_shape[i], order))
            elif v == "1d":
                self.orgParList.append(\
                                 list(var[orgpar_index[i]:orgpar_index[i+1]]))
        return self.orgParList
    
    def outputSetting(self):
        Setting = {"orgpar_type": self.orgpar_type,
                   "orgpar_order": self.orgpar_order,
                   "orgpar_index": self.orgpar_index,
                   "orgpar_shape": self.orgpar_shape}
        return Setting
    