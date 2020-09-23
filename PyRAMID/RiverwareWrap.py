# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:50:14 2020
This file is the wraper for Riverware including creating control file, 
batch file, write input data files, read RW output files, and run PyRAMID. 
@author: CYLin
"""
import os
import pandas as pd
import sys
import shutil
import errno
import subprocess
import logging
from datetime import datetime
import pickle
from PyRAMID.Setting import ConsoleLogParm, MsglevelDict, AddLocalLogFile, RemoveLocalLogFile

class RiverwareWrap(object):
    def __init__(self, WD, copyFromRootForGA_RootPath = None, Msglevel = None):
        # Setup the log msg
        self.logger = logging.getLogger(__name__)
        if Msglevel is None: Msglevel = ConsoleLogParm['Msglevel']
        else:
            assert Msglevel in ['debug', 'info', 'warning', 'error'], print("ValueError Msglevel must be one of these [None, 'debug', 'info', 'warning', 'error'].")
            Msglevel = MsglevelDict[Msglevel]
        self.logger.setLevel(Msglevel)     
          
        
        # Folders' directories
        if copyFromRootForGA_RootPath is not None:      # For GA to work in the sub working folder.
            assert os.path.isdir(copyFromRootForGA_RootPath), self.logger.error("PathError The copyFromRootForGA_RootPath is not exist {}".format(copyFromRootForGA_RootPath))
            self.copyFromRootForGA_RootPath = copyFromRootForGA_RootPath
            if not os.path.exists(WD):
                os.makedirs(WD)
                self.logger.info("Sub-working folder has been created for GA at {}.".format(WD))
        else:                
            assert os.path.isdir(WD), self.logger.error("PathError The working directory is not exist {}".format(WD))
        self.WD = WD
        self.PATH = {"ABM_Path": os.path.join(WD, "ABM"),
                     "ABM_Output_toRW_Path": os.path.join(WD, "ABM_Output_toRW"),
                     "BatchFiles_Path": os.path.join(WD, "BatchFiles"),
                     "RW_Final_Output_Path": os.path.join(WD, "RW_Final_Output"),
                     "RW_Ini_Input_Path": os.path.join(WD, "RW_Ini_Input"),
                     "RW_Output_toABM_Path": os.path.join(WD, "RW_Output_toABM"),
                     "RWModel_Path": os.path.join(WD, "RWModel")}
        self.logger.info("Set working directory to {}".format(WD))
        
        if copyFromRootForGA_RootPath is not None:
            self.copyFolder(src = os.path.join(copyFromRootForGA_RootPath, "ABM"), dest = self.PATH["ABM_Path"])
            self.copyFolder(src = os.path.join(copyFromRootForGA_RootPath, "RWModel"), dest = self.PATH["RWModel_Path"])
            self.copyFolder(src = os.path.join(copyFromRootForGA_RootPath, "RW_Ini_Input"), dest = self.PATH["RW_Ini_Input_Path"])
            self.logger.info("ABM and RWModel folders have been copied to {}.".format(self.WD))
            self.resetDMIPath2mdlFile() # modify .mdl DMI path
            
        # Check the path exist. If not, make the path
        for path in list(self.PATH.keys()):
            if not os.path.exists(self.PATH[path]):
                os.makedirs(self.PATH[path])
                self.logger.info("{} folder has been created.".format(path[:-5]))
        self.logger.info("Subfolders' path:\n"+"\n".join(['{:^23} :  {}'.format(keys, values) for keys,values in self.PATH.items()]))
        #self.logger.info("Please put your Riverware related files in RWModel folder and ABM.py in ABM folder. After setting up your Riverware model with correct DMI, please continue this program to run the coupling code.")
        
        # Folder deletion lock => reduceStorageOfGivenFolder() can only be executed when it is a copied folder.
        self.FolderCanBeDeleted = False
        if copyFromRootForGA_RootPath is not None:
            self.FolderCanBeDeleted = True
        
        # ArchiveFile: Store historical values in txt EX par that is dynamically updated.
        try: 
            with open(os.path.join(self.WD, "ArchiveFilePath.pickle"), "rb") as f:
                self.ArchiveFiles = pickle.load(f)      # Path of created archive files.
            self.logger.info("Load ArchiveFilePath.pickle")
        except:
            self.ArchiveFiles = {}
            
        return None
    

    def copyFolder(self, src, dest):
        '''
        Cpoy the following src folder to dest. Please make sure the dest folder is not exist.
        Note this function will not copy .control and year.txt files
        '''
        try:
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns('*.control', 'year.txt'))
        except OSError as e:
            # If the error was caused because the source wasn't a directory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            if e.errno == errno.ENOTDIR:
                shutil.copy(src, dest)
            else:
                self.logger.error('PathError Directory not copied. Error: %s' % e)
        return None
    
    def reduceStorageOfGivenFolder(self, reduceLevel = "keep_result_only"):
        '''
        Be careful when using this function. This function will delete the whole folder.
        Only applicable for the copied folder.
        reduceLevel: 'all', 'keep_result_only', None
        '''
        if self.FolderCanBeDeleted:
            if reduceLevel == 'all':     
                # Whole working folder will be deleted. 
                shutil.rmtree(self.WD)
            elif reduceLevel == 'keep_result_only':
                # Only keep BatchFiles, ABM and RWModel folders
                shutil.rmtree(self.PATH["ABM_Output_toRW_Path"])
                shutil.rmtree(self.PATH["RW_Ini_Input_Path"])
                shutil.rmtree(self.PATH["RW_Output_toABM_Path"])
                shutil.rmtree(self.PATH["RWModel_Path"])               
        else:
            self.logger.warning("Folders can only be deleted when they are copied from the root WD.")
        
        return None
    
    def resetDMIPath2mdlFile(self):
        '''
        Note we will search through the RWModel folder for .mdl file. We expected to see a single .mdl file!
        '''
        findmdl = os.listdir(self.PATH["RWModel_Path"])
        findmdl = [i for i in findmdl if i[-4:] == ".mdl"]
        assert len(findmdl) == 1, self.logger.error('VAlueError We expect to see one .mdl file at {}.'.format(self.PATH["RWModel_Path"]))
        
        f = open(os.path.join(self.PATH["RWModel_Path"], findmdl[0]), "r+") # read and write
        # Read
        RWFile = f.read()
        oldpath = self.copyFromRootForGA_RootPath.replace("\\","/") + "/"
        newpath = self.WD.replace("\\","/") + "/"
        RWFile = RWFile.replace(oldpath, newpath)
        # Write
        f.seek(0) # assign the pointer bact to the beginning.
        f.write(RWFile)
        f.truncate() # eliminate previous data that has not been overwrited
        f.close()
        del RWFile # Release the memory.
        self.logger.info("DMI path setting in {} has been modified to {}.".format(findmdl[0], self.WD))
        return None
        
         
    def createControlFile(self, FileName, DataFolder, ObjectList = [], Units = None, Scales = "1.0"):
        '''
        Create control file for DMI in Riverware.

        Parameters
        ----------
        FileName : string
            The FileName should consist to the name using in DMI setting in the Riverware.
        DataFolder : string
            Data folder in the working directory or the absolute folder directory.
        ObjectList : list
            Data files that you want to read/write by DMI. EX ["Yakima River at Parker PARW.Gage Inflow"]
        Units : string/dict
            Units for each data file. If enter a string, all data will be considered as this assiged unit. Or a dictionary should be provided. ex: {"cfs": [obj1, obj2, obj4], "m": [obj3]} 
        Scales : string, optional
            Scale. Same setting logic as Units. The default is "1.0".

        Returns
        -------
        None.

        '''
        if os.path.isdir(DataFolder) and os.path.isabs(DataFolder):
            DataFolder_Path = DataFolder
        else:
            assert {DataFolder+"_Path"}.issubset(set(self.PATH.keys())), self.logger.error("PathError Given DataFolder is not valid {}".format(DataFolder))
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
                    
        f = open(os.path.join(self.PATH["RWModel_Path"], FileName), "w")
        for obj in ObjectList:
            f.write("{}: file={}/%o.%s units={} scale={} import=resize\n".format(obj, DataFolder_Path, Content[obj][0], Content[obj][1]))
        f.close()
        self.logger.info("{} is created at {}".format(FileName, self.PATH["RWModel_Path"]))
        return None
    
    def createBatchFile(self, FileName, RWModelName, RWModelRuleName, OtherActionList = ["StartController"]):
        # Form content dictionary
        f = open(os.path.join(self.PATH["BatchFiles_Path"], FileName), "w")
        f.write("OpenWorkspace {}\n".format(os.path.join(self.PATH["RWModel_Path"], RWModelName).replace('\\','\\\\')))
        f.write("LoadRules {}\n".format(os.path.join(self.PATH["RWModel_Path"], RWModelRuleName).replace('\\','\\\\')))
        # f.write("InvokeDMI {}\n".format("DMI_Ini_Input"))       # Read in external data
        for a in OtherActionList:
            f.write(a+"\n")    # Run RW model (Note the adaptive updata has to be set within RW model.)
        # f.write("InvokeDMI {}\n".format("DMI_Final_Output"))    # Output data
        f.write("CloseWorkspace")
        f.close()
        self.logger.info("{} is created at {}".format(FileName, self.PATH["BatchFiles_Path"]))
        return None
    
    def createYeartxt(self, StartYear, EndYear, ABMoffsetYear = 2):
        '''
        This year.txt is for ABM.py to read as reference. 
        Note that according to rule set in RW ABM might start at different year. Use ABMoffsetYear to specify the offset year for the ABM.
        '''
    
        self.StartYear = StartYear
        self.EndYear = EndYear
        ### Create txt for dynamically record the current year
        f = open(os.path.join(self.PATH["ABM_Path"], "year.txt"), "w")
        # This is the year that ABM start to simulate the diversion. According to our algorithm, we need at least a year record to run the algorithm.
        y = StartYear + ABMoffsetYear
        f.write(str(y) + '\n')
        f.write("Start_year: " + str(StartYear) + '\n')
        f.write("End_year: " + str(EndYear) + '\n')
        f.close()
        self.logger.info("year.txt is created at {}".format(os.path.join(self.PATH["ABM_Path"], "year.txt")))
        
    def getYearfromYeartxt(self):
        f = open(os.path.join(self.PATH["ABM_Path"], "year.txt"), "r") # read
        y = int(f.readline()) # current year
        y_start = int(f.readline().split()[1])   
        y_end = int(f.readline().split()[1])  
        f.close()
        
        self.StartYear = y_start
        self.EndYear = y_end
        
        return y 
    
    def addYear2Yeartxt(self):
        # https://stackoverflow.com/questions/6648493/how-to-open-a-file-for-both-reading-and-writing
    
        f = open(os.path.join(self.PATH["ABM_Path"], "year.txt"), "r+") # read and write
        # Read
        y = int(f.readline()) # current year
        y_start = f.readline()      # already include "\n"
        y_end = f.readline()
        # Write
        f.seek(0) # assign the pointer bact to the beginning.
        f.write(str(y+1) + '\n')
        f.write(y_start)
        f.write(y_end)
        f.truncate() # eliminate previous data that has not been overwrited
        f.close()
        return None
        
    def writeRWInputFile(self, ObjSlotName, Data, Head, DataFolder = "ABM_Output_toRW"):
        '''
        Create input data file.

        Parameters
        ----------
        ObjSlotName : string
            The ObjSlotName is the filename, which should consist to the name in control files.
        Data : list
            List of the data, which the length should consist with the date length in "Head".
        Head : string
            Pre-defined header for the input data file. The header is save in sefl.Header[<Head>].
        DataFolder : string
            Data located folder. The default is "ABM_Output_toRW".
        Returns
        -------
        None.

        '''
        if os.path.isdir(DataFolder) and os.path.isabs(DataFolder):
            DataFolder_Path = DataFolder
        else:
            assert {DataFolder+"_Path"}.issubset(set(self.PATH.keys())), self.logger.error("PathError Given DataFolder is not valid {}".format(DataFolder))
            DataFolder_Path = self.PATH[DataFolder+"_Path"]
            
        f = open(os.path.join(DataFolder_Path, ObjSlotName), "w")
        for item in list(Head):
            f.write("%s\n" % item)
        for item in list(Data):
            f.write("%s\n" % item)
        f.close()
        self.logger.info("{} is created at {}".format(ObjSlotName, DataFolder_Path))
        return None
    
    def createFiles(self, FileContentDict, ABMpyFilename = "ABM.py"):
        # Simulation period
        self.StartYear = FileContentDict["Simulation"]["StartYear"]
        self.EndYear = FileContentDict["Simulation"]["EndYear"]
        self.ABMoffsetYear = FileContentDict["Simulation"]["ABMoffsetYear"]
        self.FileContentDict = FileContentDict
        self.ABMpyFilename = ABMpyFilename
        
        ########## Add a check function for FileContentDict
        # Create control file
        for ctl in list(FileContentDict["ControlFile"].keys()):
            self.createControlFile(ctl, FileContentDict["ControlFile"][ctl]["DataFolder"], FileContentDict["ControlFile"][ctl]["ObjectList"], FileContentDict["ControlFile"][ctl]["Units"], FileContentDict["ControlFile"][ctl]["Scales"])
        
        # Create batch file
        for b in list(FileContentDict["BatchFile"].keys()):
            FileContentDict["BatchFile"][b]["OtherActionList"][0] = "SetRunInfo #RunInfo !InitDate {10-31-"+str(self.StartYear) +" 24:00} !EndDate {12-31-"+str(self.EndYear)+" 24:00}"
            self.createBatchFile(b, FileContentDict["BatchFile"][b]["RWModelName"], FileContentDict["BatchFile"][b]["RWModelRuleName"], FileContentDict["BatchFile"][b]["OtherActionList"])
            
        # Create year.txt
        self.createYeartxt(self.StartYear, self.EndYear, self.ABMoffsetYear)
        
        # Create ABM.bat with given argument provide WD for ABM.py
        ABMbatFilename = "ABM.bat" #ABMpyFilename[:-3]+".bat"
        f = open(os.path.join(self.PATH["ABM_Path"], ABMbatFilename), "w")
        f.write("python %~dp0{} {}".format(ABMpyFilename, self.WD))  # %~dp0  -> To run the bat or exe files at its directory.
        f.close()
        self.logger.info("{} is created at {}".format(ABMbatFilename, os.path.join(self.PATH["ABM_Path"], ABMbatFilename)))
        return None
    
    def readRWOutputFile(self, Filename, DataFolder = "RW_Output_toABM", Freq = "D"):
        '''
        Read RW output to dataframe.

        Parameters
        ----------
        Filename : string
            
        DataFolder : string, optional
            Data located folder.. The default is "RW_Output_toABM_Path".
        Freq : string, optional
            Frequency of the data. The default is "D".

        Returns
        -------
        df : DataFrame

        '''
        if os.path.isdir(DataFolder) and os.path.isabs(DataFolder):
            DataFolder_Path = DataFolder
        else:
            assert {DataFolder+"_Path"}.issubset(set(self.PATH.keys())), self.logger.error("PathError Given DataFolder is not valid {}".format(DataFolder))
            DataFolder_Path = self.PATH[DataFolder+"_Path"]
        
        # Read data
        path = os.path.join(DataFolder_Path, Filename)
        assert os.path.isfile(path), self.logger.error("PathError Given file is not exist {}".format(path))
        file = open(path, "r")  # Read mode
        text_file = file.readlines()
        file.close()
        
        if "start_date" in text_file[0]:
            start_date = text_file[0].split(" ")[1]
        else:
            start_date = [i for i in text_file[0:10] if "start_date" in i][0].split(" ")[1].replace("-", "/")
            
        if "end_date" in text_file[1]:
            end_date = text_file[1].split(" ")[1]
        else:
            end_date = [i for i in text_file[0:10] if "end_date" in i][0].split(" ")[1].replace("-", "/")
        index = [i+1 for i, v in enumerate(text_file[0:10]) if "#" in v][0]     # Find index where data start
        data = list(map(float, text_file[index:])) 
        
        df = pd.DataFrame()
        df[Filename] = data
        rng = pd.date_range(start = start_date, end = end_date, freq=Freq)
        df.index = rng
        return df
    
    def createDoNothingBatFile(self, path = None):
        if path is None: path = self.PATH["ABM_Path"]
        assert os.path.isdir(path), self.logger.error("PathError Given directory is not found {}".format(path))
        f = open(os.path.join(path, "DoNothing.bat"), "w")
        f.write("@echo off\n")
        f.write("rem This bat executable doesn't do anything.")
        f.close()
        self.logger.info("Creat DoNothing.bat file at {}.".format(path))
        return None
    
    def runPyRAMID(self, RiverwarePath, BatchFileName, ExecuteNow = True, SaveAllLog = True):
        '''
        Execute PyRAMID through python code. First, the program will create the executable .bat file. Then, run the .bat file by subprocess python package.
        All the run-time information will be stored in Main.log. The Riverware software run-time information will be stored at BatchFiles folder. 

        Parameters
        ----------
        RiverwarePath : string
            Path of licensed Riverware.exe.
        BatchFileName : string
            The name of the batchfile (Riverware).
        ExecuteNow : boolen, optional
            If false, all related files will be created but not executed. The default is True.

        Returns
        -------
        None.

        '''
        assert os.path.isfile(os.path.join(RiverwarePath, "riverware.exe")), self.logger.error("PathError riverware.exe is not found in the given RiverwarePath {}".format(RiverwarePath))
        
        # Create RunBatch.bat
        f = open(os.path.join(self.WD, "RunBatch.bat"), "w")
        f.write("cd {}\n".format(RiverwarePath.replace('\\','\\\\')))
        f.write("riverware.exe --batch {} --log {}".format(os.path.join(self.PATH["BatchFiles_Path"], BatchFileName).replace('\\','\\\\'), os.path.join(self.PATH["BatchFiles_Path"], BatchFileName.split(".")[0]+".log").replace('\\','\\\\')))
        f.close()
        self.logger.info("Creat RunBatch.bat file at {}.".format(os.path.join(self.WD, "RunBatch.bat")))
        
        # Run the model
        if ExecuteNow:
            start_time = datetime.now()
            # Create the local log file for this run.
            if SaveAllLog:
                dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
                Logfile = 'RunBatch_{}.log'.format(dt_string)
            else:
                Logfile = "RunBatch.log"
            self.logger, self.fh = AddLocalLogFile(Logfile, self.logger, self.WD)
            # fh = logging.FileHandler(Logfile, 'w') # Overwrite the file
            # fh.setLevel(logging.DEBUG)
            # formatter_fh = logging.Formatter('[%(asctime)s] %(name)s [%(levelname)s] %(message)s')
            # fh.setFormatter(formatter_fh)
            # self.logger.addHandler(fh)
            CreateFileHandler = True
                
            self.logger.info("Execute RunBatch.bat file.")
            print("\n\nRunning PyRAMID.........     \n(This will take time. You can monitor the progress by checking log file in the Batch folder.)")
            
            # Execute the RunBatch.bat using subprocess package https://docs.python.org/3/library/subprocess.html
            cmd = os.path.join(self.WD, "RunBatch.bat").replace("\\","/")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            #self.logger.info(process.stdout.read(1))
            try:
                for c in iter(lambda: process.stdout.read(1), b''):  # replace '' with b'' for Python 3
                    #self.logger.info(c)
                    sys.stdout.write(c)    # This will garentee to print out the msg generated from RW 
            except:
                self.logger.warning("Fail to print out RW log information. Please check the log file at Batch folder directly.")
            
            # Read out Riverware.log
            f = open(os.path.join(self.PATH["BatchFiles_Path"], BatchFileName.split(".")[0]+".log"), "r")
            for x in f:
                self.logger.info(x)
            f.close()
            end_time = datetime.now()
            duration = end_time - start_time
            self.logger.info("\nFinish running RunBatch.bat file.\nDuration: {}".format(duration))
            
            # Remove the created file handler.
            if CreateFileHandler:    
                self.logger = RemoveLocalLogFile(self.logger, self.fh)
        return None
    
    def ObjSlotName2FileName(self, Name):
        '''
        Convert ObjSlot name or a list of names from RW to the filename, which RW replace the space and ":" to "_".
        '''
        if isinstance(Name, str):
            Name = Name.replace(" ","_")
            Name = Name.replace(":","_")
            return Name 
        else:
            Namelist = [i.replace(" ","_") for i in Name]
            Namelist = [i.replace(":","_") for i in Namelist]
            return Namelist

    def collectRW_Final_Output2CSV(self, filename = None):
        y = self.getYearfromYeartxt() # To make sure we have the following attributions.
        StartYear = self.StartYear
        EndYear = self.EndYear
        
        if filename is None:
            filename = os.path.join(self.WD, "FinalOutput.csv")
        Slotslist = os.listdir(self.PATH["RW_Final_Output_Path"])
        FinalOutput = pd.DataFrame()
        for i in Slotslist:
            df = self.readRWOutputFile(i, DataFolder = "RW_Final_Output", Freq = "D")
            df = df.loc['{}-01-01'.format(StartYear):'{}-12-31'.format(EndYear)]
            FinalOutput = pd.concat([FinalOutput, df], axis = 1)
        FinalOutput.to_csv(filename)
        self.logger.info("Results have been save at {}".format(filename))
        return None
    
    def readCSV(self, filename):
        df = pd.read_csv(filename, index_col=0, parse_dates = True)
        return df

## Archive files
    def createArchiveFile(self, Filename):
        Filename = Filename.replace("\\", "/")
        directory = "/".join(Filename.split("/")[:-1])
        if os.path.isdir(directory):
            with open(Filename, 'w') as f: 
                pass
        else:
            
            if not os.path.exists(os.path.join(self.WD, "Archive")):
                os.makedirs(os.path.join(self.WD, "Archive"))
            Filename = os.path.join(self.WD, "Archive", Filename).replace("\\", "/")
            with open(Filename, 'w') as f: 
                pass
            
        self.ArchiveFiles[Filename.split("/")[-1]] = Filename
        self.logger.info("Create archive file: {}".format(Filename))
        with open(os.path.join(self.WD, "ArchiveFilePath.pickle"), 'wb') as outfile:
            pickle.dump(self.ArchiveFiles, outfile)
        return None
    
    def archive(self, txt_filename, value):
        with open(self.ArchiveFiles[txt_filename], 'a') as f: 
            f.write(str(value)+"\n")
            
    def ArchiveFiles2CSV(self):
        # Try to search the WD for ArchiveFilePath.pickle
        try: 
            with open(os.path.join(self.WD, "ArchiveFilePath.pickle"), "rb") as f:
                self.ArchiveFiles = pickle.load(f)      # Path of created archive files.
            self.logger.info("Load ArchiveFilePath.pickle")
        except:
            self.ArchiveFiles = {}
        
        
        if self.ArchiveFiles == {}:
            self.logger.info("ArchiveFiles is empty.")
        else:
            data = {}
            for file in self.ArchiveFiles:
                with open(self.ArchiveFiles[file], 'r') as f: 
                    data[file.split(".")[0]] = [i.rstrip() for i in f.readlines()]
            df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
            df.to_csv(os.path.join(self.WD, "ArchiveData.csv"),  index = False)
        return df