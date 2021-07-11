import os
import logging
from datetime import datetime

# This is for the console message printing. 
ConsoleLogParm = {
    "MsgLevel": logging.INFO,
    "MsgFormat": '[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
    "DateFormate": '%m/%d %I:%M:%S'
    }

# This log file will not be implemented by default.
FileLogParm = {
    "MsgLevel": logging.INFO,
    "MsgFormat": '[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
    "DateFormate": None,
    "Filename":r".\RiverwareABM.log"
    }

# This log file will be created automatically when exercute  
# runPyRAMID() in RiverwareWrap.
FileLogParm_runRiverwareABM = {
    "MsgLevel": logging.DEBUG,
    "MsgFormat": '[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
    "DateFormate": None,
    "Filename":"PyRAMID.log"
    }

MsglevelDict = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR
    }

def addGlobalLogFile(Filename=None, Mode='w', MsgLevel=None):
    """Add a global log file.

    Args:
        Filename (str, optional): Log file name. Defaults to None.
        Mode (str, optional): txt mode. Defaults to 'w'.
        MsgLevel (str, optional): Message level. Defaults to None.

    Returns:
        None
    """
                     
    if Filename is None:
        Filename = FileLogParm["Filename"]
    # else:
    #     Filename = os.path.join(Directory, Filename)
    # Create file handler at "root" level.
    logger = logging.getLogger("PyRAMID")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(Filename, Mode) # 'w' Overwrite, 'a' append
    if MsgLevel is not None:
        assert MsgLevel in ['debug', 'info', 'warning', 'error'], \
            print("ValueError MsgLevel must be one of these [None, 'debug', "+\
                  "'info', 'warning', 'error'].")
        fh.setLevel(MsglevelDict[MsgLevel])
    else:
        fh.setLevel(FileLogParm["MsgLevel"])
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

def addLocalLogFile(filename, logger, Directory, Mode='w', MsgLevel=None):
    """Add local log file.
    
    This function is to add the local log file for modules within 
    RiverwareABMpackage. 
    This function can be use for files outside of RiverwareABM 
    package if the proper logger is given. To get the proper  
    logger, we recommend user to run setLoggerForCustomizedFile().

    Args:
        filename (str): Log file name.
        logger (object): logger object.
        Directory (str): Log file folder directory.
        Mode (str, optional): .txt mode. Defaults to 'w'.
        MsgLevel (str, optional): Message level. Defaults to None.

    Returns:
        [list]: A list contains logger and file handler.
    """

    Filename = os.path.join(Directory, filename)
    fh = logging.FileHandler(Filename, Mode) # w: Overwrite the file, a: append
    if MsgLevel is not None:
        assert MsgLevel in ['debug', 'info', 'warning', 'error'], \
            print("ValueError MsgLevel must be one of these [None, 'debug', "\
                  + "'info', 'warning', 'error'].")
        fh.setLevel(MsglevelDict[MsgLevel])
    else:
        fh.setLevel(FileLogParm["MsgLevel"])
    formatter_fh = logging.Formatter(FileLogParm["MsgFormat"], \
                                     datefmt=FileLogParm["DateFormate"])
    fh.setFormatter(formatter_fh)
    logger.addHandler(fh)
    return logger, fh

def removeLocalLogFile(logger, fh):
    """Remove file handler from given logger.

    Args:
        logger (object): logger object.
        fh (object): File handler object.

    Returns:
        object: logger object.
    """

    logger.removeHandler(fh)
    return logger

def setLoggerForCustomizedFile(AbbrOfThisPyFile, MsgLevel=None):
    """Set logger.
    
    This function help to get hierarchical logger under the 
    root RiverwareABM with the given abbreviation of your file.

    Args:
        AbbrOfThisPyFile (str): Abbreviation of .py file name.
        MsgLevel (str, optional): Message level. 'debug', 'info',
        'warning', 'error'. Defaults to None.

    Returns:
        object: logger
    """
    # Add the hierarchical logger under the root RiverwareABM.
    logger = logging.getLogger("PyRAMID.{}".format(AbbrOfThisPyFile)) 
    if MsgLevel is not None:
        assert MsgLevel in ['debug', 'info', 'warning', 'error'], \
            print("ValueError MsgLevel must be one of these [None, 'debug', "\
                  + "'info', 'warning', 'error'].")
        logger.setLevel(MsglevelDict[MsgLevel])
    else:
        logger.setLevel(ConsoleLogParm["MsgLevel"])
    return logger

def createFolderWithDatetime(WD, Folder):
    """Create folder with datetime under given WD.

    Args:
        WD (str): Working directory.
        Folder (str): Prefix of the folder name.

    Returns:
        [str]: Created folder's directory.
    """
    assert os.path.isdir(WD),\
        print("PathError Given directory not exists. {}".format(WD))
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    # This is use to store all calibration data
    WD_new = os.path.join(WD, Folder+"_{}".format(dt_string)) 
    os.makedirs(WD_new)
    print("Create {}".format(WD_new))
    return WD_new