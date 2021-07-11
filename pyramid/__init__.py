import json
import logging
from .riverware_wrap import RiverwareWrap
from .ga import GeneticAlgorithm, GADataConverter
from .setting import (addGlobalLogFile, 
                      setLoggerForCustomizedFile, 
                      createFolderWithDatetime,
                      ConsoleLogParm)

print("\n\nWelcome to Py-RAMID!\nA python package of a Riverware and "
      + "Agent-based Modeling Interface for Developers.\n")

logger = logging.getLogger(__name__)      # This is the root of logging.
logger.setLevel(ConsoleLogParm["MsgLevel"])

# Clear all existed handlers and  add new console handler by default.
logger.handlers.clear()
ch = logging.StreamHandler()
ch.setLevel(ConsoleLogParm["MsgLevel"])
formatter_ch = logging.Formatter(ConsoleLogParm["MsgFormat"], 
                                 datefmt=ConsoleLogParm["DateFormate"])
ch.setFormatter(formatter_ch)
logger.addHandler(ch)

def writeModelSetting2Json(filename, dictionary):
    data = json.dumps(dictionary, sort_keys=True, indent=4)
    with open(filename, "w") as f:
      f.write(data)
    return None

def readModelSettingFromJson(filename):
    with open(filename, "r") as f:
        dictionary = json.load(f)
        #print("Load JSON encoded data into Python dictionary")
    return dictionary