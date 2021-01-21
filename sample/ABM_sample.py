# -*- coding: utf-8 -*-
"""
Created on Thr Jan 21 23:24:49 2021
This is a sample of the general procedure for ABM.py design. 
The complete version of ABM.py for (Lin et al., 2021, submitted) is on the 
request to authors.

@author: C.Y. Lin at philip928lin@gmail.com
"""

import os
import sys
import pandas as pd
import numpy as np
from PyRAMID import PyRAMID

# Import ModelSetting
ModelSettingPath = r"C:\ModelSetting.json"
ModelSetting = PyRAMID.readModelSettingFromJson(ModelSettingPath)

def ABM(WD):
    # Initialize RiverwareWrap with WD
    RW_wrap = PyRAMID.RiverwareWrap(WD)
    
    #=========================================================================
    # Define agents
    #=========================================================================
    Agents = {}
    ##### Stage 1 ############################################################
    for i, v in enumerate(AgentName):      
        ## Initial agent 
        Agents[v] = PyRAMID.DiversionAgent(index = i, agentname = v, \
                                           Par = AgentPar[v], \
                                           SocialNormMatrix = SocialNormMatrix)
    
    
    ###### Collect neighbors' information ####################################
    # Note that the order of AgentsProbArr and SocialNormMatrix row order have  
    # to be consist.
    AgentsProbArr = []
    for v in AgentName:
        AgentsProbArr.append(Agents[v].CurrentValue["quantile"])
    AgentsProbArr = np.array(AgentsProbArr)    
    
    # Achive agents' original decisions
    if os.path.exists(os.path.join(WD,"AgentsProbsOrg.csv")) is False:
        RW_wrap.archive(os.path.join(WD,"AgentsProbsOrg.csv"),\
                        value = ",".join(["Year"]+AgentName))
    RW_wrap.archive(os.path.join(WD,"AgentsProbsOrg.csv"), \
                    value = "{},{}".format(y, \
                    ",".join([str(round(AgentsProbArr[i], 4)) \
                              for i in range(len(AgentName))])))    
    
    ##### Stage 2 ############################################################
    # Calculate AnnualRatio
    s = []; s2 = []; c = []; r = []; q = [] # For archiving
    for v in AgentName:
        ### strength = obv_flow - sim_flow
        ## Here we calculate strength using flow violation from last year   
        ActualFlow = FlowDatabase_annual.loc['{}-01-01'.format(y-1): \
                                             '{}-12-31'.format(y-1), \
                                                 DistrictFlow[v]].values
        SimFlow = FlowRW_annual.loc['{}-01-01'.format(y-1): \
                                    '{}-12-31'.format(y-1), \
                                        AgentSlots[v]["FlowSlots"]].values
        ## Average all downstream flow violation from the diversion point
        strength = np.nanmean((ActualFlow - SimFlow).sum(axis = 0))
        
        ## Run decision-making
        Stochastic = False
        Agents[v].makeDecision(AgentsProbArr, strength, Stochastic)
        
        ## Save RLCenter and Record
        AgentPar[v]['RLCenter'] = Agents[v].Par["RLCenter"]
        s.append(str(round(strength, 2)))
        s2.append(str(round(Agents[v].CurrentValue["Strength"], 4)))
        c.append(str(round(Agents[v].Par["RLCenter"], 4)))
        r.append(str(round(AnnualRatio[v], 4)))
        q.append(str(round(Agents[v].\
                           CurrentValue["quantile_social_adjusted"], 4)))
    # Achive strength, RLCenter, AnnualRatio
    AchiveItems = {"Strength":s, "StrengthSig":s2, "RLCenter":c, \
                   "AnnualRatio":r, "AgentsProbsAdj": q}
    for f in AchiveItems:
        if os.path.exists(os.path.join(WD, f + ".csv")) is False:
            RW_wrap.archive(os.path.join(WD, f + ".csv"),\
                            value = ",".join(["Year"]+AgentName))
        RW_wrap.archive(os.path.join(WD, f + ".csv"), \
                        value = "{},{}".format(y, \
                        ",".join(AchiveItems[f])))        
        
    ##### Stage 3 ############################################################
    # Calculate final daily diversion output
    AnnualRatio = pd.Series(AnnualRatio)        
    UpdatedDailyDiv = NormalizedDailyDiv*AnnualRatio
    # Add index
    rng = pd.date_range(start='{}-01-01'.format(y), \
                        end='{}-12-31'.format(y), freq='D')
    UpdatedDailyDiv.index = rng
    OutputDailyDiv.loc[rng, AgentName] = UpdatedDailyDiv.loc[rng, AgentName]  
        
    ##### Stage 4 ############################################################
    # Export RW inputs, save Diversion.csv and updated parameters
    OutputDailyDiv.to_csv(OutputDailyDivPath) # Form the diversion.csv for RW.
    AgentPar = pd.DataFrame(AgentPar)           
    AgentPar.to_csv(Par_path_ABMcopy)         # Save the updated parameters
    
    # Export to RW 
    for v in AgentName:
        ObjSlotName = "ABM_Diversion.ABM_{}_Wet".format(v)
        Data = OutputDailyDiv.loc[:,v]
        Head = ["start_date: {}-01-01 24:00".format(start_date),
                "end_date: {}-12-31 24:00".format(end_date),
                "timestep: 1 DAY",
                "units: cfs",
                "scale: 1.000000",
                "# Series Slot: {}_Wet".format(v)]
        RW_wrap.writeRWInputFile(ObjSlotName, Data, Head, \
                                 DataFolder = "ABM_Output_toRW")
    
    for v in AgentName:
        ObjSlotName = "ABM_Diversion.ABM_{}_Dry".format(v)
        Data = OutputDailyDiv.loc[:,v]
        Head = ["start_date: {}-01-01 24:00".format(start_date),
                "end_date: {}-12-31 24:00".format(end_date),
                "timestep: 1 DAY",
                "units: cfs",
                "scale: 1.000000",
                "# Series Slot: {}_Dry".format(v)]
        RW_wrap.writeRWInputFile(ObjSlotName, Data, Head, \
                                 DataFolder = "ABM_Output_toRW")
            
    logger.info("ABM {} done!".format(y))
            
    # Save year for next iteration
    RW_wrap.addYear2Yeartxt()


def main():
    # Get WD from the command line argument
    print(sys.argv)
    WD = sys.argv[1]
    logger = PyRAMID.setLoggerForCustomizedFile(AbbrOfThisPyFile = "ABM")
    assert os.path.isdir(WD), \
        logger.error("PathError Given command line argument is not a valid working directory. {}".format(sys.argv[1:]))
    
    # Run ABM
    ABM(WD)

if __name__ == "__main__":
    main()  
