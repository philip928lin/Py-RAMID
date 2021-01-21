# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:12:57 2020

@author: CYLin
"""
import os
import logging
from datetime import datetime

# This is for the console message printing. 
ConsoleLogParm = {"Msglevel": logging.INFO,
            "MsgFormat": '[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
            "DateFormate": '%m/%d %I:%M:%S'}

# This log file will not be implemented by default.
FileLogParm = {"Msglevel": logging.INFO,
            "MsgFormat": '[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
            "DateFormate": None,
            "Filename":r".\RiverwareABM.log"}

# This log file will be created automatically when exercute runRiverwareABM() 
# in Riverware wrap.
FileLogParm_runRiverwareABM = {"Msglevel": logging.DEBUG,
            "MsgFormat": '[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
            "DateFormate": None,
            "Filename":"PyRAMID.log"}

MsglevelDict = {"debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR}



def AddGlobalLogFile(Filename = None, Directory = os.getcwd(), mode = 'w', \
                     Msglevel = None):
    if Filename is None:
        Filename = FileLogParm["Filename"]
    # else:
    #     Filename = os.path.join(Directory, Filename)
    # Create file handler at "root" level.
    logger = logging.getLogger("PyRAMID")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(Filename, mode) # 'w' Overwrite, 'a' append
    if Msglevel is not None:
        assert Msglevel in ['debug', 'info', 'warning', 'error'], \
            print("ValueError Msglevel must be one of these [None, 'debug', "+\
                  "'info', 'warning', 'error'].")
        fh.setLevel(MsglevelDict[Msglevel])
    else:
        fh.setLevel(FileLogParm["Msglevel"])
    formatter_fh = logging.Formatter(FileLogParm["MsgFormat"], \
                                     datefmt=FileLogParm["DateFormate"])
    fh.setFormatter(formatter_fh)
    logger.addHandler(fh)
    
    # print the following message with package hierarchical structures.
    logger = logging.getLogger(__name__) 
    logger.setLevel(logging.INFO)
    #logger.info("Global log file is created at {}"\
    #            .format(os.path.join(os.getcwd(),Filename[2:])))
    return None

def AddLocalLogFile(filename, logger, Directory, mode = 'w', Msglevel = None):
    '''
    This function is to add the local log file for modules within RiverwareABM
    package. 
    This function can be use for files outside of RiverwareABM package if the
    proper logger is given. To get the proper logger, we recommend user to run 
    setLoggerForCustomizedFile().
    '''
    Filename = os.path.join(Directory, filename)
    fh = logging.FileHandler(Filename, mode) # w: Overwrite the file, a: append
    if Msglevel is not None:
        assert Msglevel in ['debug', 'info', 'warning', 'error'], \
            print("ValueError Msglevel must be one of these [None, 'debug', "+\
                  "'info', 'warning', 'error'].")
        fh.setLevel(MsglevelDict[Msglevel])
    else:
        fh.setLevel(FileLogParm["Msglevel"])
    formatter_fh = logging.Formatter(FileLogParm["MsgFormat"], \
                                     datefmt=FileLogParm["DateFormate"])
    fh.setFormatter(formatter_fh)
    logger.addHandler(fh)
    return logger, fh

def RemoveLocalLogFile(logger, fh):
    '''
    This function is to remove the handler. 
    It is not designed to use outside of this package.
    '''
    logger.removeHandler(fh)
    return logger

def setLoggerForCustomizedFile(AbbrOfThisPyFile, Msglevel = None):
    '''
    This function help to get hierarchical logger under the root RiverwareABM 
    with the given abbreviation of your file.
    Msglevel:  None, 'debug', 'info', 'warning', 'error'
    '''
    # Add the hierarchical logger under the root RiverwareABM.
    logger = logging.getLogger("PyRAMID.{}".format(AbbrOfThisPyFile)) 
    if Msglevel is not None:
        assert Msglevel in ['debug', 'info', 'warning', 'error'], \
            print("ValueError Msglevel must be one of these [None, 'debug', "+\
                  "'info', 'warning', 'error'].")
        logger.setLevel(MsglevelDict[Msglevel])
    else:
        logger.setLevel(ConsoleLogParm["Msglevel"])
    return logger

def createFolderWithDatetime(WD, Folder):
    assert os.path.isdir(WD),\
        print("PathError Given directory not exists. {}".format(WD))
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    # This is use to store all calibration data
    WD_new = os.path.join(WD, Folder+"_{}".format(dt_string)) 
    os.makedirs(WD_new)
    print("Create {}".format(WD_new))
    return WD_new