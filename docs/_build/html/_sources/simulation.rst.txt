Simulation
==========

.. _inputs:

Inputs preparation
------------------

To run a coupled model simulation, users need to prepare three inputs.

1.  ModelSetting (.json) file. Template file is provided in the example folder.
2.  The modified RiverWare model (.mdl) and corresponding policy file (.rls).
3.  User-defined human model (.py) such as ABM.py.

In the ModelSetting (.json) file, modelers define the information flow for data exchange between RiverWare (RW) and Human/Agent-based Model (ABM) (import/export slots of the RW), the RW simulation periods, and other RW actions using RW command language (e.g., LoadRules). Using the information in ModelSetting (.json), Py-RAMID will create control (.control) and batch (.rcl) files. Data management interface (DMI) from the RW uses control files to determine the imported/exported slots. A batch file is used to execute the RW model with predefined action order (e.g., OpenWorkspace, LoadRules, and SetRunInfo). Therefore, Py-RAMID serves as a wrapper to help modelers form all required coupling files. However, modelers must add two additional policies that are associated with the RW-to-ABM and ABM-to-RW DMIs into the original RW policy rules (.rls) for the very first time. Inside those two additional policies, modelers can define data exchange frequency; for example, to export the RW data on 31 December of a year and re-import the data on 1 January of a year. For ABM.py, modelers have complete freedom to define agents and their interactions. 

.. note::
   Please create the working directory using Py-RAMID and put the corresponding files in the correct folder first; then, `Modify RiverWare model`_.  

Working folder structure
------------------------

The working folder structure created by Py-RAMID is shown below

::

    WD
    ├── BatchFiles
    ├── batch file (.rcl)
    ├── ABM          
    │   ├── ABM.py
    │   └── ABM.bat/ABM.exe
    ├── RW_Ini_Input          
    ├── RW_Output_toABM
    ├── ABM_Output_toRW          
    └── RWModel
        ├── Control.control
        ├── RWModel.mdl
        └── RWPolicy.rls

This folder structure can be automatically created by running the following code.

.. code-block:: python

   import pyramid as PyRAMID
   RwWrap = PyRAMID.RiverwareWrap( "WD" )

After initializing the working folder, users need to place the prepared inputs_ into the corresponding folder.

*  Put ModelSetting.json under WD
*  Put RW model (.mdl), RW policy file (.rls), and other RW-related files under WD/RWModel

.. note::
   Only one .mdl file is allowed to be in RWModel folder.

* Put human model (e.g., ABM.py) under ABM. 


Modify RiverWare model
----------------------

Py-RAMID builds the coupled model by utilizing the Data Management Interface (DMI) provided by RiverWare. Therefore, it is important for users to manually add DMI setting and two additional policies into policy file (.rls) at least one time. These two additional policies can trigger DMIs (one for import data and the other to export RW output) at desire timestep.

Here we provide an example. However, we refer users to `RiverWare.org <https://www.riverware.org/>`_ for detail instructions.

1.  After setup the ModelSetting.json, run the following code to create files for coupling.

.. code-block:: python

   # Load ModelSetting.json
   ModelSetting = PyRAMID.readModelSettingFromJson("ModelSetting.json")

   # Create files
   RwWrap.createFiles(FileContentDict=ModelSetting["FileContentDict"],
	                  ABMpyFilename="ABM.py")

For DMIs setup, we need **DoNothing.bat** and **ABM.bat** or **ABM.exe**. If given ABM.py, Py-RAMID will automatically create **DoNothing.bat** and **ABM.bat** at the ABM folder.

2.  Open RiverWare model (.mdl) and load Policy file (.rls). Then, add two DMIs shown below.

.. image:: images/DMI.png

3.  Open policy and add following two additional policies (usually with highest policy piority). At this step, users can assign desire information exchange frequency. In this example, we import the ABM output to RW on 1 January and export RW output on 31 December.

.. image:: images/Policy.png

Copying existed working folder to a new directory
-------------------------------------------------

Once the first working directory is setup and successfully run the simulation. Users can use following code to copy existed working folder to a new directory for a new numerical experiment. Py-RAMID will automatically correct the path setting in RW. There is no need for manually modifying RiverWare model again.

.. code-block:: python

   RwWrap = PyRAMID.RiverwareWrap( "New WD" , "Source WD")


Example
----------
Here we provide a sample code for running a coupled model simulation using Py-RAMID.

.. code-block:: python

	# =============================================================================
	# PyRAMID: Coupled Model Simulation
	# =============================================================================
	import pyramid as PyRAMID

	# Step 1: Load model setting json file
	ModelSetting = PyRAMID.readModelSettingFromJson("ModelSetting.json")


	# Step 2: Create RiverwareWrap object with given working directory
	RwWrap = PyRAMID.RiverwareWrap( "WD" )
	# or copy from existed working folder, which path in RW model will be auto-updated.
	RwWrap = PyRAMID.RiverwareWrap( "New WD" , "Source WD", OverWrite=True)

	# Step 3: Create simulation related files 
	RwWrap.createFiles(FileContentDict=ModelSetting["FileContentDict"],
	                   ABMpyFilename="ABM.py")

	# Step 4: Run simulation
	RwWrap.runPyRAMID(RiverwarePath="Riverware executable file path", 
	                  BatchFileName="BatchFile.rcl", 
	                  ExecuteNow=True, Log=True)