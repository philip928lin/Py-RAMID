"""
This file is the wraper for RiverWare including creating control file, 
batch file, write input data files, read RW output files, and run 
PyRAMID. 
@author: CYLin
"""
import os
import sys
import shutil
import errno
import pickle
import subprocess
from datetime import datetime
import logging
import pandas as pd
from .setting import (ConsoleLogParm, MsglevelDict, 
                      addLocalLogFile, removeLocalLogFile, 
                      addGlobalLogFile, setLoggerForCustomizedFile)

class RiverwareWrap(object):
    def __init__(self, WD, RootPath=None, Overwrite=False, MsgLevel=None):
        """RiverWare wrapper for coupling.

        Args:
            WD (str): Working directory.
            RootPath (str, optional): Source folder directory for copying.
            Defaults to None.
            Overwrite (bool, optional): Overwrite the folder if it exists.
            Defaults to False.
            MsgLevel (str, optional): Message level. Defaults to None.

        Returns:
            None
        """
                 
        # Setup the log msg
        self.logger = logging.getLogger(__name__)
        if MsgLevel is None: 
            MsgLevel = ConsoleLogParm['MsgLevel']
        else:
            assert MsgLevel in ['debug', 'info', 'warning', 'error'], \
                print("ValueError MsgLevel must be one of these [None, "\
                      + "'debug', 'info', 'warning', 'error'].")
            MsgLevel = MsglevelDict[MsgLevel]
        self.logger.setLevel(MsgLevel)     
          
        
        # Folders' directories
        if RootPath is not None:
            # For GA to work in the sub working folder.
            assert os.path.isdir(RootPath), \
                self.logger.error("PathError: The given RootPath is not exist"\
                                  +" {}.".format(RootPath))
            self.RootPath = RootPath
            if not os.path.exists(WD):
                os.makedirs(WD)
                self.logger.info("Sub-working folder has been created for "\
                                 + "at {}.".format(WD))
        else:                
            assert os.path.isdir(WD), \
                self.logger.error("PathError: The working directory is not "\
                                  +"exist {}.".format(WD))
        self.WD = WD
        self.PATH = {
            "ABM_Path": os.path.join(WD, "ABM"),
            "ABM_Output_toRW_Path": os.path.join(WD, "ABM_Output_toRW"),
            "BatchFiles_Path": os.path.join(WD, "BatchFiles"),
            "RW_Final_Output_Path": os.path.join(WD, "RW_Final_Output"),
            "RW_Ini_Input_Path": os.path.join(WD, "RW_Ini_Input"),
            "RW_Output_toABM_Path": os.path.join(WD, "RW_Output_toABM"),
            "RWModel_Path": os.path.join(WD, "RWModel")
            }
        self.logger.info("Set working directory to {}.".format(WD))
        
        if RootPath is not None:
            self.copyFolder(Src = os.path.join(RootPath, "ABM"), 
                            Dest = self.PATH["ABM_Path"], Overwrite=Overwrite)
            self.copyFolder(Src = os.path.join(RootPath, "RWModel"), 
                            Dest = self.PATH["RWModel_Path"], 
                            Overwrite = Overwrite)
            self.copyFolder(Src = os.path.join(RootPath, "RW_Ini_Input"), 
                            Dest = self.PATH["RW_Ini_Input_Path"], 
                            Overwrite = Overwrite)
            self.logger.info("ABM and RWModel folders have been copied to {}."\
                             .format(self.WD))
            self.resetDMIPath2mdlFile() # modify .mdl DMI path
            
        # Check the path exist. If not, make the path
        for path in list(self.PATH.keys()):
            if not os.path.exists(self.PATH[path]):
                os.makedirs(self.PATH[path])
                self.logger.info("{} folder has been created."\
                                 .format(path[:-5]))
        self.logger.info("Subfolders' path:\n"+"\n".join(['{:^23} :  {}'\
                         .format(keys, values) for keys,values in \
                             self.PATH.items()]))

        # Folder deletion lock => reduceStorageOfGivenFolder() can only 
        # be executed when it is a copied folder.
        self.FolderCanBeDeleted = False
        if RootPath is not None:
            self.FolderCanBeDeleted = True
        
        # ArchiveFile: Store historical values in txt EX par that is 
        # dynamically updated.
        try: 
            ArchiveFilePath = os.path.join(self.WD, 
                                           "ArchiveFilePath.pickle")
            with open(ArchiveFilePath, "rb") as File:
                self.ArchiveFiles = pickle.load(File)  
            self.logger.info("Load ArchiveFilePath.pickle")
        except:
            self.ArchiveFiles = {}
        return None

    def copyFolder(self, Src, Dest, Overwrite=False):
        """Copy the from Src folder to Dest.

        Note this function will not copy .control and year.txt files.
        .control and year.txt files should be created by PyRAMID.
        
        Args:
            Src (str): Source folder path.
            Dest (str): Destination folder path.
            Overwrite (bool, optional): Overwrite the folder if exist. 
            Defaults to False.

        Returns:
            None
        """
        # Remove the existed folder.
        if os.path.exists(Dest) and Overwrite:
            shutil.rmtree(Dest)
            self.logger.warning('The following folder will be overwrited. {}'\
                                .format(Dest))
        try:
            shutil.copytree(Src, Dest, \
                       ignore=shutil.ignore_patterns('*.control', 'year.txt'))
        except OSError as e:
            # If the error was caused because the source wasn't a directory                                                    
            if e.errno == errno.ENOTDIR:
                shutil.copy(Src, Dest)
            else:
                self.logger.error('PathError Directory not copied. '+\
                                  'Error: %s' % e)
        return None
    
    def copyFile(self, Src, Dest):
        """shutil.copyfile(Src, Dest)

        Args:
            Src (str): Source file path.
            Dest (str): Destination file path.

        Returns:
            None
        """
        shutil.copyfile(Src, Dest) # Will overwrite existed file 
        return None

    def reduceStorageOfGivenFolder(self, ReduceLevel="keep_result_only"):
        """Delete folders to reduce storage.
        
        Be careful when using this function. This function will delete 
        the entire folder. Only applicable for the copied folders.

        Args:
            ReduceLevel (str, optional): 'all', 'keep_result_only'. 
            Defaults to "keep_result_only".

        Returns:
            None
        """
        
        if self.FolderCanBeDeleted:
            if ReduceLevel == 'all':     
                # Whole working folder will be deleted. 
                shutil.rmtree(self.WD)
            elif ReduceLevel == 'keep_result_only':
                # Only keep BatchFiles, ABM and RWModel folders
                shutil.rmtree(self.PATH["ABM_Output_toRW_Path"])
                shutil.rmtree(self.PATH["RW_Ini_Input_Path"])
                shutil.rmtree(self.PATH["RW_Output_toABM_Path"])
                shutil.rmtree(self.PATH["RWModel_Path"])               
        else:
            self.logger.warning("Folders can only be deleted when they are "+\
                                "copied from the root WD.")
        return None

    def resetDMIPath2mdlFile(self):
        """Modify DMI path in the RiverWare .mdl file.

        Returns:
            None
        """
        
        Findmdl = os.listdir(self.PATH["RWModel_Path"])
        Findmdl = [i for i in Findmdl if i[-4:] == ".mdl"]
        assert len(Findmdl) == 1, self.logger.error('ValueError We expect '+\
              'to see one .mdl file at {}.'.format(self.PATH["RWModel_Path"]))
        
        # r+: read and write
        File = open(os.path.join(self.PATH["RWModel_Path"], Findmdl[0]), "r+") 
        # Read
        RWFile = File.read()
        oldpath = self.RootPath.replace("\\","/") + "/"
        newpath = self.WD.replace("\\","/") + "/"
        RWFile = RWFile.replace(oldpath, newpath)
        # Write
        File.seek(0) # assign the pointer bact to the beginning.
        File.write(RWFile)
        # eliminate previous data that has not been overwrited
        File.truncate() 
        File.close()
        del RWFile # Release the memory.
        self.logger.info("DMI path setting in {} has been modified to {}."\
                         .format(Findmdl[0], self.WD))
        return None
        
    def createControlFile(self, FileName, DataFolder, ObjectList=[],\
                          Units=None, Scales="1.0"):
        """Create control files for RiverWare.

        Args:
            FileName (str): The FileName should consist to the name 
            using in DMI setting in the Riverware.
            DataFolder (str): Data folder in the working directory 
            or the absolute folder directory.
            ObjectList (list, optional): Data files that you want to 
            read/write by DMI. EX ["Yakima River at Parker PARW.Gage 
            Inflow"]. Defaults to [].
            Units (string/dict, optional): Units for each data file. 
            If enter a string, all data will be considered as this 
            assiged unit. Or a dictionary should be provided. 
            ex: {"cfs": [obj1, obj2, obj4], "m": [obj3]} 
            Scales (string/dict, optional): Scale. Same setting logic
            as Units. The default is "1.0".. Defaults to "1.0".

        Returns:
            None
        """
        if os.path.isdir(DataFolder) and os.path.isabs(DataFolder):
            DataFolder_Path = DataFolder
        else:
            assert {DataFolder+"_Path"}.issubset(set(self.PATH.keys())), \
                self.logger.error("PathError Given DataFolder is not valid {}"\
                                  .format(DataFolder))
            DataFolder_Path = self.PATH[DataFolder+"_Path"]
        
        # Form content dictionary
        Content = {key: [] for key in ObjectList} 
        if isinstance(Units, str):
            for o in ObjectList:
                Content[o].append(Units)
        else:
            for u in list(Units.keys()):
                objlist = Units[u]
                for o in objlist:
                    Content[o].append(u)
        if Scales == "1.0":
            for o in ObjectList:
                Content[o].append("1.0")
        else:
            for s in list(Scales.keys()):
                objlist = Scales[s]
                for o in objlist:
                    Content[o].append(s)
                    
        File = open(os.path.join(self.PATH["RWModel_Path"], FileName), "w")
        for obj in ObjectList:
            File.write("{}: file={}/%o.%s units={} scale={} import=resize\n"\
                    .format(obj, DataFolder_Path, Content[obj][0], \
                            Content[obj][1]))
        File.close()
        self.logger.info("{} is created at {}"\
                         .format(FileName, self.PATH["RWModel_Path"]))
        return None
    
    def createBatchFile(self, FileName, RWModelName, RWModelRuleName, \
                        OtherActionList=["StartController"]):
        """Create batch file for RiverWare.

        Args:
            FileName (str): Batch file name.
            RWModelName (str): RiverWare .mdl file name.
            RWModelRuleName (str): RiverWare policy .rls file name.

        Returns:
            None
        """
        # Form content dictionary
        File = open(os.path.join(self.PATH["BatchFiles_Path"], FileName), "w")
        File.write("OpenWorkspace {}\n"\
                .format(os.path.join(self.PATH["RWModel_Path"], RWModelName)\
                        .replace('\\','\\\\')))
        File.write("LoadRules {}\n"\
                .format(os.path.join(self.PATH["RWModel_Path"], \
                                     RWModelRuleName).replace('\\','\\\\')))
        for a in OtherActionList:
            File.write(a+"\n")   
        File.write("CloseWorkspace")
        File.close()
        self.logger.info("{} is created at {}"\
                         .format(FileName, self.PATH["BatchFiles_Path"]))
        return None

    def createYeartxt(self, StartYear, EndYear, ABMoffsetYear=2):
        """Create year.txt for ABM model to read as reference.
        
        Note that according to rule set in RW ABM might start at different 
        year. Use ABMoffsetYear to specify the offset year for the ABM.

        Args:
            StartYear (int): Simulation start year.
            EndYear (int): Simulation end year.
            ABMoffsetYear (int, optional): Offset years for starting ABM. 
            Defaults to 2.
        """
    
        self.StartYear = StartYear
        self.EndYear = EndYear
        ### Create txt for dynamically record the current year
        File = open(os.path.join(self.PATH["ABM_Path"], "year.txt"), "w")
        # This is the year that ABM start to simulate the diversion. 
        # According to our algorithm, we need at least a year record to  
        # run the algorithm.
        y = StartYear + ABMoffsetYear
        File.write(str(y) + '\n')
        File.write("Start_year: " + str(StartYear) + '\n')
        File.write("End_year: " + str(EndYear) + '\n')
        File.close()
        self.logger.info("year.txt is created at {}"\
                     .format(os.path.join(self.PATH["ABM_Path"], "year.txt")))
        
    def getYearfromYeartxt(self):
        """Read year from year.txt.

        Returns:
            int: Current simulated year.
        """
        File = open(os.path.join(self.PATH["ABM_Path"], "year.txt"), "r") 
        y = int(File.readline()) # current year
        y_start = int(File.readline().split()[1])   
        y_end = int(File.readline().split()[1])  
        File.close()
        
        self.StartYear = y_start
        self.EndYear = y_end
        
        return y 
    
    def addYear2Yeartxt(self):
        """Add 1 year to year.txt.

        Returns:
            None
        """
        # r+: read and write
        File = open(os.path.join(self.PATH["ABM_Path"], "year.txt"), "r+") 
        # Read
        y = int(File.readline())  # current year
        y_start = File.readline() # already include "\n"
        y_end = File.readline()
        # Write
        File.seek(0) # assign the pointer bact to the beginning.
        File.write(str(y+1) + '\n')
        File.write(y_start)
        File.write(y_end)
        # eliminate previous data that has not been overwrited
        File.truncate() 
        File.close()
        return None
        
    def writeRWInputFile(self, ObjSlotName, Data, Head, \
                         DataFolder="ABM_Output_toRW"):
        """Create input data file.

        Args:
            ObjSlotName (str): The ObjSlotName is the filename, 
            which should consist to the 
            name in control files.
            Data (list): List of the data, which the length 
            should consist with the 
            date length in "Head".
            Head (str): Pre-defined header for the input data 
            file. The header is 
            save in self.Header[<Head>].
            DataFolder (str): Data located folder. The default 
            is "ABM_Output_toRW".

        Returns:
            None
        """
        if os.path.isdir(DataFolder) and os.path.isabs(DataFolder):
            DataFolder_Path = DataFolder
        else:
            assert {DataFolder+"_Path"}.issubset(set(self.PATH.keys())), \
                self.logger.error("PathError Given DataFolder is not valid {}"\
                                  .format(DataFolder))
            DataFolder_Path = self.PATH[DataFolder+"_Path"]
            
        File = open(os.path.join(DataFolder_Path, ObjSlotName), "w")
        for item in list(Head):
            File.write("%s\n" % item)
        for item in list(Data):
            File.write("%s\n" % item)
        File.close()
        self.logger.info("{} is created at {}"\
                         .format(ObjSlotName, DataFolder_Path))
        return None
    
    def createFiles(self, FileContentDict, ABMpyFilename="ABM.py"):
        """Create files for RiverWare.
        
        This will create control files, batch files, year.txt, 
        and ABM.bat.

        Args:
            FileContentDict (dict): File content dictionary in the 
            setting.json.
            ABMpyFilename (str, optional): ABM model. 
            Defaults to "ABM.py".

        Returns:
            None
        """
        # Simulation period
        self.StartYear = FileContentDict["Simulation"]["StartYear"]
        self.EndYear = FileContentDict["Simulation"]["EndYear"]
        self.ABMoffsetYear = FileContentDict["Simulation"]["ABMoffsetYear"]
        self.FileContentDict = FileContentDict
        self.ABMpyFilename = ABMpyFilename
        
        ########## Add a check function for FileContentDict
        # Create control file
        for ctl in list(FileContentDict["ControlFile"].keys()):
            self.createControlFile(
                ctl,
                FileContentDict["ControlFile"][ctl]["DataFolder"],
                FileContentDict["ControlFile"][ctl]["ObjectList"],
                FileContentDict["ControlFile"][ctl]["Units"],
                FileContentDict["ControlFile"][ctl]["Scales"]
                )
        
        # Create batch file
        for b in list(FileContentDict["BatchFile"].keys()):
            FileContentDict["BatchFile"][b]["OtherActionList"][0] = \
                "SetRunInfo #RunInfo !InitDate {10-31-"+str(self.StartYear)\
                + " 24:00} !EndDate {12-31-"+str(self.EndYear)+" 24:00}"
            self.createBatchFile(
                b,
                FileContentDict["BatchFile"][b]["RWModelName"],
                FileContentDict["BatchFile"][b]["RWModelRuleName"],
                FileContentDict["BatchFile"][b]["OtherActionList"]
                )
            
        # Create year.txt
        self.createYeartxt(self.StartYear, self.EndYear, self.ABMoffsetYear)
        
        # Create ABM.bat with given argument provide WD for ABM.py
        ABMbatFilename = "ABM.bat" #ABMpyFilename[:-3]+".bat"
        File = open(os.path.join(self.PATH["ABM_Path"], ABMbatFilename), "w")
        # %~dp0  -> To run the bat or exe files at its directory.
        File.write("python %~dp0{} {}".format(ABMpyFilename, self.WD))  
        File.close()
        self.logger.info("{} is created at {}"\
                         .format(ABMbatFilename, 
                                 os.path.join(self.PATH["ABM_Path"], 
                                              ABMbatFilename)))
        return None
    
    def readRWOutputFile(self, Filename, DataFolder="RW_Output_toABM",
                         Freq="D"):
        """Read RW output to dataframe.

        Args:
            Filename (str): File name.
            DataFolder (str, optional): Data located folder.
            Defaults to "RW_Output_toABM".
            Freq (str, optional): Frequency of the data. 
            Defaults to "D".

        Returns:
            [DataFrame]: Dataframe.
        """

        if os.path.isdir(DataFolder) and os.path.isabs(DataFolder):
            DataFolder_Path = DataFolder
        else:
            assert {DataFolder+"_Path"}.issubset(set(self.PATH.keys())), \
                self.logger.error("PathError Given DataFolder is not valid {}"\
                                  .format(DataFolder))
            DataFolder_Path = self.PATH[DataFolder+"_Path"]
        
        # Read data
        DataPath = os.path.join(DataFolder_Path, Filename)
        assert os.path.isfile(DataPath),\
            self.logger.error("PathError Given file is not exist {}"\
                              .format(DataPath))
        File = open(DataPath, "r")  # Read mode
        TextFile = File.readlines()
        File.close()
        
        if "start_date" in TextFile[0]:
            StartDate = TextFile[0].split(" ")[1]
        else:
            StartDate = [i for i in TextFile[0:10] if "start_date" in i][0]\
                          .split(" ")[1].replace("-", "/")
            
        if "end_date" in TextFile[1]:
            EndDate = TextFile[1].split(" ")[1]
        else:
            EndDate = [i for i in TextFile[0:10] if "end_date" in i][0]\
                        .split(" ")[1].replace("-", "/")
        # Find index where data start
        Index = [i+1 for i, v in enumerate(TextFile[0:10]) if "#" in v][0]     
        Data = list(map(float, TextFile[Index:])) 
        
        df = pd.DataFrame()
        df[Filename] = Data
        rng = pd.date_range(start=StartDate, end=EndDate, freq=Freq)
        df.index = rng
        return df
    
    def createDoNothingBatFile(self, Path=None):
        """Create do nothing .bat.

        Args:
            Path (str, optional): Folder directory. Defaults to None.

        Returns:
            None
        """
        if Path is None: 
            Path = self.PATH["ABM_Path"]
        assert os.path.isdir(Path), \
            self.logger.error("PathError Given directory is not found {}"\
                              .format(Path))
        File = open(os.path.join(Path, "DoNothing.bat"), "w")
        File.write("@echo off\n")
        File.write("rem This bat executable doesn't do anything.")
        File.close()
        self.logger.info("Creat DoNothing.bat file at {}.".format(Path))
        return None
    
    def runPyRAMID(self, RiverwarePath, BatchFileName, ExecuteNow=True,
                   Log=True):
        """Execute PyRAMID through python code. 

        Args:
            RiverwarePath (str): Path of licensed Riverware.exe.
            BatchFileName (str): The name of the batchfile (Riverware).
            ExecuteNow (bool, optional): If false, all related files 
            will be created but not executed. 
             Defaults to True.
            Log (bool, optional): Create log file if ture. 
            Defaults to True.

        Returns:
            None
        """
        
        assert os.path.isfile(os.path.join(RiverwarePath, "riverware.exe")), \
            self.logger.error("PathError riverware.exe is not found in the "+\
                              "given RiverwarePath {}".format(RiverwarePath))
        
        # Create RunBatch.bat
        File = open(os.path.join(self.WD, "RunBatch.bat"), "w")
        File.write("cd {}\n".format(RiverwarePath.replace('\\','\\\\')))
        # File.write("riverware.exe --batch {} --log {}"\
        #         .format(os.path.join(self.PATH["BatchFiles_Path"], \
        #                              BatchFileName).replace('\\','\\\\'), \
        #                 os.path.join(self.WD, "PyRAMID.log")))
        File.write("riverware.exe --batch {} --log {}"\
                .format(os.path.join(self.PATH["BatchFiles_Path"], \
                                      BatchFileName).replace('\\','\\\\'), \
                        os.path.join(self.PATH["BatchFiles_Path"], \
                                      BatchFileName.split(".")[0]+".log")\
                            .replace('\\','\\\\')))
        File.close()
        self.logger.info("Creat RunBatch.bat file at {}."\
                         .format(os.path.join(self.WD, "RunBatch.bat")))
        
        # Run the model
        if ExecuteNow:
            CreateFileHandler = False
            StartTime = datetime.now()
            # Create the local log file for this run.
            if Log:
                #Create local log file and remove later on
                dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
                Logfile = 'RunBatch_{}.log'.format(dt_string)
                self.logger, self.fh = addLocalLogFile(Logfile, self.logger, \
                                                        self.WD)
                CreateFileHandler = True
                
                # Create global log file (Given exact same log file name) 
                # GlobalLogFile = os.path.join(self.WD, "PyRAMID.log")
                logger_RW = setLoggerForCustomizedFile(AbbrOfThisPyFile = "RW")
                # AddGlobalLogFile(GlobalLogFile, mode = 'a')
                            
            self.logger.info("Execute RunBatch.bat file.")
            print("\n\nRunning PyRAMID.........     \n(This will take time. "+\
                  "You can monitor the progress by checking log file in the "+\
                      "Batch folder.)")
            
            # Execute the RunBatch.bat using subprocess package 
            # https://docs.python.org/3/library/subprocess.html            
            cmd = os.path.join(self.WD, "RunBatch.bat").replace("\\","/")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            # This enable wait until finish command while allow the 
            # runtime communication.
            process.communicate() 
            #self.logger.info(process.stdout.read(1))
            try:
                while True:
                    out = process.stdout.readline()
                    out = out.decode('utf-8').replace('\n', '')
                    if out == '' and process.poll() is not None:
                        break
                    if out != '':
                        logger_RW.info(out)
                        sys.stdout.write(out)
                        sys.stdout.flush()
            except:
                self.logger.warning("Fail to print out RW log information. "+\
                                    "Please check the log file at Batch "+\
                                    "folder directly.")

            # Read out Riverware.log
            File = open(os.path.join(self.PATH["BatchFiles_Path"], \
                                  BatchFileName.split(".")[0]+".log"), "r")
            for x in File:
                self.logger.info(x)
            File.close()
            EndTime = datetime.now()
            Duration = EndTime - StartTime
            self.logger.info("\nFinish running RunBatch.bat file.\n"+\
                             "Duration: {}".format(Duration))
            # Remove the created file handler.
            if CreateFileHandler:    
                self.logger = removeLocalLogFile(self.logger, self.fh)
        return None
    
    def ObjSlotName2FileName(self, Name):
        """Convert RiverWare slots' name to file name.
        
        Convert ObjSlot name or a list of names from RW to 
        the filename, which RW replace the space and ":" to 
        "_".

        Args:
            Name (str/list): RiverWare slots' name (list).

        Returns:
            [str/list]: File name (list).
        """
        '''
        
        '''
        if isinstance(Name, str):
            Name = Name.replace(" ","_")
            Name = Name.replace(":","_")
            return Name 
        else:
            Namelist = [i.replace(" ","_") for i in Name]
            Namelist = [i.replace(":","_") for i in Namelist]
            return Namelist

    def collectRW_Final_Output2CSV(self, filename=None):
        """Collect RiverWare outputs into a CSV.

        Args:
            filename (str, optional): CSV file name. 
            Defaults to None resulting in FinalOutput.csv.

        Returns:
            None
        """
        # To make sure we have the following attributions.
        y = self.getYearfromYeartxt() 
        StartYear = self.StartYear
        EndYear = self.EndYear
        
        if filename is None:
            filename = os.path.join(self.WD, "FinalOutput.csv")
        Slotslist = os.listdir(self.PATH["RW_Final_Output_Path"])
        FinalOutput = pd.DataFrame()
        for i in Slotslist:
            df = self.readRWOutputFile(i, DataFolder="RW_Final_Output", \
                                       Freq="D")
            df= df.loc['{}-01-01'.format(StartYear):'{}-12-31'.format(EndYear)]
            FinalOutput = pd.concat([FinalOutput, df], axis=1)
        FinalOutput.to_csv(filename)
        self.logger.info("Results have been save at {}".format(filename))
        return None
    
    def readCSV(self, filename):
        """Read csv.

        Args:
            filename (str): CSV file name.

        Returns:
            [DataFrame]: Dataframe.
        """
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        return df

## Archive files
    def createArchiveFile(self, Filename, Collect=True):
        """Create txt archive files.

        Args:
            Filename (str): Complete path or just txt filename. If 
            users only provide txt filename, the archive files will 
            be stored at Archive folder under working directory.
            Collect (bool, optional): True, then the filename and 
            corresponding path will be saved at 
            ArchiveFilePath.pickle, which could be used by 
            ArchiveFiles2CSV(). Defaults to True.

        Returns:
            None
        """
        Filename = Filename.replace("\\", "/")
        directory = "/".join(Filename.split("/")[:-1])
        if os.path.isdir(directory):
            with open(Filename, 'w') as File: 
                pass
        else:
            if not os.path.exists(os.path.join(self.WD, "Archive")):
                os.makedirs(os.path.join(self.WD, "Archive"))
            Filename = os.path.join(self.WD, "Archive", Filename)\
                        .replace("\\", "/")
            with open(Filename, 'w') as File: 
                pass
            
        self.ArchiveFiles[Filename.split("/")[-1]] = Filename
        self.logger.info("Create archive file: {}".format(Filename))
        if Collect:
            ArchiveFilePath = os.path.join(self.WD, "ArchiveFilePath.pickle")
            with open(ArchiveFilePath, 'wb') as outfile:
                pickle.dump(self.ArchiveFiles, outfile)
        return None
    
    def archive(self, txt_filename, value, Collect=False):
        """Archive data. 
        
        If file is not exist, creat a new one.

        Args:
            txt_filename (str): .txt file name.
            value (float/str/list): Content to archive.
            Collect (bool, optional): True, then the filename and 
            corresponding path will be saved at 
            ArchiveFilePath.pickle, which could be used by 
            ArchiveFiles2CSV(). Defaults to False.

        Returns:
            None
        """
        Filename = txt_filename.replace("\\", "/").split("/")[-1]
        if os.path.exists(txt_filename) is False:
            self.createArchiveFile(txt_filename, Collect)
        if self.ArchiveFiles.get(Filename) is not None:
            with open(self.ArchiveFiles[Filename], 'a') as File: 
                File.write(str(value)+"\n")
        else:
            with open(txt_filename, 'a') as File: 
                File.write(str(value)+"\n")
        return None
    
    def ArchiveFiles2CSV(self):
        """Convert archived files in ArchiveFilePath.pickle to CSV

        Returns:
            [DataFrame]: Dataframe.
        """
        # Try to search the WD for ArchiveFilePath.pickle
        try: 
            ArchiveFilePath = os.path.join(self.WD, "ArchiveFilePath.pickle")
            with open(ArchiveFilePath, 'wb') as outfile:
                ArchiveFiles = pickle.load(outfile)     
            self.logger.info("Load ArchiveFilePath.pickle")
        except:
            ArchiveFiles = {}
        
        if ArchiveFiles == {}:
            self.logger.info("ArchiveFilePath.pickle is empty or not found.")
        else:
            Data = {}
            for file in ArchiveFiles:
                with open(ArchiveFiles[file], 'r') as File: 
                    Data[file.split(".")[0]] = [i.rstrip() \
                                                for i in File.readlines()]
            df = pd.DataFrame(dict([ (k,pd.Series(v)) \
                                    for k,v in Data.items() ]))
            df.to_csv(os.path.join(self.WD, "ArchiveData.csv"), index = False)
        return df