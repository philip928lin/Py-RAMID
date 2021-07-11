Calibration
============
The calibration module of Py-RAMID allows users to calibrate the coupled model with genetic algorithm in parallel. Due to the file-based coupling structure between RiverWare and human models (e.g., ABM), seperated working directory for each simuteniously simulated evaluation are required. To do that, users have to define an objective function with **var** and **SubWD** arguments as shown below.

.. code-block:: python 

	def ObjFunc(var, SubWD):
	    # Create RiverwareWrap object at SubWD, which files will be copy and 
	    # modified automatically from the source directory.
	    RwWrap = PyRAMID.RiverwareWrap(SubWD , "Source WD")
	    
	    # Update parameters using var from GA.
	    # Covert var (1sD array) from GA to original formats, DataFrame or Array.
	    Converter = PyRAMID.GADataConverter() 
	    # ParDF1 and ParDF2 are given uncalibrated parameter dataframes.
	    ConvertedVar = Converter.Covert2GAArray([ParDF1, ParDF2])
	    # Update ParDF1 and ParDF2 with var.
	    ParDF1, ParDF2 = Converter.GAArray2OrgPar(var)
	    # Save Par1DF and Par2Arr to ABM folder at RwWrap.PATH["ABM_Path"]
	    ParDF1.to_csv(os.path.join(RwWrap.PATH["ABM_Path"], "ParDF1.csv"))
	    ParDF2.to_csv(os.path.join(RwWrap.PATH["ABM_Path"], "ParDF2.csv"))
	    
	    # Create files and start the simulation
	    RwWrap.createFiles()
	    RwWrap.runPyRAMID()
	    
	    # Calculate objective value for minimization optimization
	    objective = ObjectiveFunction( Simulation outputs )
	    return objective

.. note::
   `PyRAMID.GADataConverter()`_ is a function to convert a list of dataframes and 
   1d/2d array to 1D array. Also, it can convert 1d array back to original 
   format. For details about PyRAMID.GADataConverter(), please see here_. 

Example
------------

.. code-block:: python 

	import os
	import pyramid as PyRAMID
	
	# Define an objective function with var and SubWD arguments
	def ObjFunc(var, SubWD):
	    # Create RiverwareWrap object at SubWD, which files will be copy and 
	    # modified automatically from the source directory.
	    RwWrap = PyRAMID.RiverwareWrap( SubWD , "Source WD")
	    
	    # Update parameters using var from GA.
	    # Covert var (1D array) from GA to original formats, DataFrame or Array.
	    Converter = PyRAMID.GADataConverter() 
	    # ParDF1 and ParDF2 are given uncalibrated parameter dataframes.
	    ConvertedVar = Converter.Covert2GAArray([ParDF1, ParDF2])
	    # Update ParDF1 and ParDF2 with var.
	    ParDF1, ParDF2 = Converter.GAArray2OrgPar(var)
	    # Save Par1DF and Par2Arr to ABM folder at RwWrap.PATH["ABM_Path"]
	    ParDF1.to_csv(os.path.join(RwWrap.PATH["ABM_Path"], "ParDF1.csv"))
	    ParDF2.to_csv(os.path.join(RwWrap.PATH["ABM_Path"], "ParDF2.csv"))
	    
	    # Create files and start the simulation
	    RwWrap.createFiles()
	    RwWrap.runPyRAMID()
	    
	    # Calculate objective value for minimization optimization
	    objective = ObjectiveFunction( Simulation outputs )
	    return objective

	# Create GA object with given working directory ga_WD
	algorithm_parameters = {'max_num_iteration': 100,
                        	'population_size':100,
                        	'mutation_probability':0.3,
                        	'elit_ratio': 0.03,
                        	'crossover_probability': 0.5,
                        	'parents_portion': 0.3,
                        	'crossover_type':'uniform',
                        	'max_iter_without_improv': None}
    
	NumAgent = 6          	
	varbound = [[0,1]]*NumAgent + [[0,2]]*NumAgent + [[0,2]]*NumAgent       
	vartype =  [['real'], ['real'], ['real']]*NumAgent

	AutoGA = PyRAMID.GeneticAlgorithm(function = ObjFunc, 
                                      wd = ga_WD,
                                      dimension = len(vartype), 
                                      variable_boundaries = varbound, 
                                      variable_type_mixed = vartype, 
                                      threads = 8, 
                                      seed = 2,
                                      saveGADataPerIter = True, 
                                      function_timeout = 300000, 
                                      parallel = 2, 
                                      algorithm_parameters = algorithm_parameters,
                                      continue_file = None, 
                                      msg_level = None)

	# Start calibration
	AutoGA.runGA()

	# Or to continue previous run by loading GAobject.pickle.
	AutoGA = PyRAMID.GeneticAlgorithm(continue_file = "GAobject.pickle") 
	AutoGA.runGA()

.. _here:

PyRAMID.GADataConverter()
-------------------------
GADataConverter() is a class that can convert between a list of Dataframes or arrays (1D or 2D) and a 1D array. We design this for assisting calibration. Below is an example.

.. code-block:: python 

	import pandas as pd
	import pyramid as PyRAMID


	ParDF1 = pd.DataFrame({"Agent1": [1,2,3], "Agent2": [4,5,6]}, 
	                      index = ["Par1", "Par2", "Par3"])
	ParDF2 = pd.DataFrame({"Agent3": [9,8,7], "Agent4": [6,5,4]}, 
	                      index = ["Par1", "Par2", "Par3"])

	# Create a object called Converter.
	Converter = PyRAMID.GADataConverter()


	# ParDF1 and ParDF2 are given uncalibrated parameter dataframes.
	ConvertedVar = Converter.Covert2GAArray([ParDF1, ParDF2])
	# ConvertedVar
	# Out[7]: [1, 4, 2, 5, 3, 6, 9, 6, 8, 5, 7, 4]


	# Covert 1D ConvertedVar back to a DataFrame list.
	DFList = Converter.GAArray2OrgPar(ConvertedVar)

	# DFList
	# Out: 
	# [      Agent1  Agent2
	#  Par1       1       4
	#  Par2       2       5
	#  Par3       3       6,
	#        Agent3  Agent4
	#  Par1       9       6
	#  Par2       8       5
	#  Par3       7       4]


	# Update ParDF1 and ParDF2 with var.
	var = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
	ParDF1, ParDF2 = Converter.GAArray2OrgPar(var)

	# ParDF1
	# Out: 
	#       Agent1  Agent2
	# Par1       1       2
	# Par2       3       4
	# Par3       5       6

	# ParDF2
	# Out: 
	#       Agent3  Agent4
	# Par1       7       8
	# Par2       9      10
	# Par3      11      12