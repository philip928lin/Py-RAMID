# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:13:59 2020
This file is to help users import all defined classes by just typing 
< from RiverwareABM import RiverwareABM >.

Otherwise they can import indivdual class from the modular directly like what
we did in this file.

@author: CYLin
"""

from PyRAMID.RiverwareWrap import RiverwareWrap
from PyRAMID.DiversionAgent import DiversionAgent
from PyRAMID.Agent import *
from PyRAMID.AnalysisWrap import PyRAMIDAnalysis, ProspectFuncPlot
from PyRAMID.GA import GeneticAlgorithm, GADataConverter
from PyRAMID.Setting import AddGlobalLogFile, setLoggerForCustomizedFile, createFolderWithDatetime

import json
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