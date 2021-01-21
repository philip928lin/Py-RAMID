# Py-RAMID
A **Py**thon package of a **R**iverware and **A**gent-based **M**odeling **I**nterface for **D**evelopers.

We developed a Python package of Riverware (RW, a river-reservoir routing model) and Agent-based Modeling (ABM, a human decision model) Interface for Developers, Py-RAMID, to address co-evolution challenges in a coupled natural-human system.

Py-RAMID is designed to work under Windows environment. The following instructions are mainly done by conda environment.

Contact: C.Y. Lin at philip928lin@gmail.com.

# Install
Py-RAMID can be installed by download the package from the [Py-RAMID Github repository](https://github.com/philip928lin/Py-RAMID). After that, unzip the file and move to the directory containing setting.py in the terminal or the conda prompt. Then, you should be able to install the package by
```python
pip install .
```
Py-RAMID is designed to work under Python 3.7 and Windows environment. The following instructions are mainly done by conda environment.


# Py-RAMID coupling concept
![](https://i.imgur.com/WQhMuvi.png)

**Fig. 1.** Py-RAMID framework and calibration structure. Two grey arrows indicate primary tasks performed by the Py-RAMID framework. Three background colors distinguish three modules in the Py-RAMID. Three user-prepared items are highlighted in red boxes. Feedback loop is shown with thick solid arrows.

More details please see (Lin et al., 2021, submitted).

# How to use it?
## Prerequisites
1. Install Py-RAMID.
2. Make sure the .py file can be executed through CMD with correct environment. In other words, evironment path has to be set correctly. For more details, please see Q&A “Setting evironment path”.
3. A valid [RiverWare](https://www.riverware.org/) license.

## Inputs preparation
With the assistance of Py-RAMID, modelers only need to prepare three items, with some modifications, in the original RW model. The three user-prepared items (highlighted by red rectangles in Fig. 1) are 
1. ModelSetting (.json) file [Sample file is provided in the sample folder.]
2. The modified RW model 
4. The ABM model (.py)

In the **ModelSetting (.json)** file, modelers define the information flow for data exchange between RW and ABM (import/export slots of the RW), the RW simulation periods, and other RW actions using RW command language (e.g., LoadRules). Using the information in ModelSetting (.json), Py-RAMID will create control (.control) and batch (.rcl) files. Data management interface (DMI) from the RW uses control files to determine the imported/exported slots. A batch file is used to execute the RW model with predefined action order (e.g., OpenWorkspace, LoadRules, and SetRunInfo). Therefore, Py-RAMID serves as a wrapper to help modelers form all required coupling files. However, modelers **must add two additional policies that are associated with the RW-to-ABM and ABM-to-RW DMIs into the original RW policy rules (.rls**) for the very first time. Inside those two additional policies, modelers can define data exchange frequency; for example, to export the RW data on 31 December of a year and re-import the data on 1 January of a year. For ABM.py, modelers have complete freedom to define agents and their interactions. Finally, Py-RAMID enables modelers to integrate their ABM.py into a unified logging system using **`setLoggerForCustomizedFile()`**. 
## Sample code for a coupling model development
### Coupling model simulation
```python=
# =============================================================================
# PyRAMID: Coupling Model Simulation
# =============================================================================
from PyRAMID import PyRAMID

# Step 1: Load model setting json file
ModelSetting = PyRAMID.readModelSettingFromJson("ModelSetting.json")


# Step 2: Create RiverwareWrap object with given working directory
RwWrap = PyRAMID.RiverwareWrap( "WD" )
# or copy from existed working folder, which path in RW model will be auto-updated.
RwWrap = PyRAMID.RiverwareWrap( "New WD" , "Source WD")

# Step 3: Create simulation related files 
RwWrap.createFiles(FileContentDict = ModelSetting["FileContentDict"],
                   ABMpyFilename = "ABM.py")

# Step 4: Run simulation
RwWrap.runPyRAMID(RiverwarePath = "Riverware executable file path", 
                  BatchFileName = "BatchFile.rcl", 
                  ExecuteNow = True, Log = True)
```

### Auto calibration with parallel genetic algorithm
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

