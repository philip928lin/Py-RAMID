# Py-RAMID
A **Py**thon package of a **R**iverware and **A**gent-based **M**odeling **I**nterface for **D**evelopers.

Py-RAMID is designed to work under Windows environment. The following instructions are mainly done by conda environment.


# Install
Py-RAMID can be installed by download the package from the [Py-RAMID Github repository](https://github.com/philip928lin/Py-RAMID). After that, unzip the file and move to the directory containing setting.py in the terminal or the conda prompt. Then, you should be able to install the package by
```python
pip install .
```
Py-RAMID is designed to work under Python 3.7 and Windows environment. The following instructions are mainly done by conda environment.


Note: 
> Before using Py-RAMID, please make sure the .py file can be executed through CMD with correct environment. In other words, evironment path has to be set correctly. For more details, please see Q&A "**Setting evironment path**".

# Pseudo code for coupling model development
## Coupling model simulation
```python=
# =============================================================================
# PyRAMID: Coupling Model Simulation
# =============================================================================
from PyRAMID import PyRAMID

# Load model setting json file
ModelSetting = PyRAMID.readModelSettingFromJson("ModelSetting.json")


# Create RiverwareWrap object with given working directory
RwWrap = PyRAMID.RiverwareWrap( "WD" )
# or copy from existed working folder, which path in RW model will be auto-updated.
RwWrap = PyRAMID.RiverwareWrap( "New WD" , "Source WD")

# Create simulation files 
RwWrap.createFiles(FileContentDict = ModelSetting["FileContentDict"],
                   ABMpyFilename = "ABM.py")

# Run simulation
RwWrap.runPyRAMID(RiverwarePath = "Riverware executable file path", 
                  BatchFileName = "BatchFile.rcl", 
                  ExecuteNow = True, Log = True)
```

## Auto calibration with parallel genetic algorithm
```python=
from PyRAMID import PyRAMID
import os

# Define simulation function with var and GA_WD arguments
def CaliFunc(var, GA_WD):
    # Create RiverwareWrap object at GA_WD, which files will be copy and modified automaticall from the source directory.
    RwWrap = PyRAMID.RiverwareWrap( GA_WD , "Source WD")
    
    # Update parameters using var from GA.
    # Covert var (1-D array) from GA to original formats, DataFrame or Array.
    Converter = PyRAMID.GADataConverter() 
    ConvertedVar = Converter.Covert2GAArray([ParDF1, ParDF2])
    ParDF1, ParDF2 = Converter.GAArray2OrgPar(var)
    # Save Par1DF and Par2Arr to ABM folder at GA_WD ( RwWrap.PATH["ABM_Path"] )
    ParDF1.to_csv(os.path.join(RwWrap.PATH["ABM_Path"], "ParDF1.csv"))
    ParDF2.to_csv(os.path.join(RwWrap.PATH["ABM_Path"], "ParDF2.csv"))
    
    # Create files and start the simulation
    RwWrap.createFiles()
    RwWrap.runPyRAMID()
    
    # Calculate objective value for minimization optimization
    objective = ObjectiveFunction( Simulation outputs )
    return objective

# Create GA object with given working directory ga_WD
AutoGA = PyRAMID.GeneticAlgorithm(function = CaliFunc, Other settings)

# Start calibration
AutoGA.runGA()

# Or to continue previous run by loading GAobject.pickle.
AutoGA = PyRAMID.GeneticAlgorithm(continue_file = "GAobject.pickle") 
AutoGA.runGA()
```


# Q&A
## Setting evironment path 
To ensure the .py file can be exercuted with proper python environment, environment path must be correstly assigned. To setup the environment path please folloe the steps below.
1.	Open anaconda prompt.
2.	Inside anaconda prompt write command, **where python**, then a list of corresponding python will appear.
![](https://i.imgur.com/OO1YDmy.png)
3.  Copy the one that is in your working environment. In our example, our working environment is at **C:\Users\ResearchPC\anaconda3\envs\PyRAMID**.
4. 	Open windows search and search Edit System Environment Variables. 

![](https://i.imgur.com/mZqyW3I.png)

![](https://i.imgur.com/9n14GRQ.png)

![](https://i.imgur.com/uT0YRp4.png)

