# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:27:47 2020
This file is the agent moduler, which is designed according to the algorithm 
developed by Chung-Yi Lin.
@author: CYLin
"""
import math
import numpy as np
from scipy.stats import beta, norm, normaltest
import logging
from PyRAMID.Setting import ConsoleLogParm, MsglevelDict

class DiversionAgent(object):
    def __init__(self, index, agentname, Par = None, SocialNormMatrix = None, Msglevel = None):
        self.__name__ = "DiversionAgent"
        # Setup the log msg
        self.logger = logging.getLogger(__name__)
        if Msglevel is None: Msglevel = ConsoleLogParm['Msglevel']
        else:
            assert Msglevel in ['debug', 'info', 'warning', 'error'], print("ValueError Msglevel must be one of these [None, 'debug', 'info', 'warning', 'error'].")
            Msglevel = MsglevelDict[Msglevel]
        self.logger.setLevel(Msglevel)   

        # Check input eligibility
        assert isinstance(index, int), self.logger.error("TypeError index has to be an integer.")
        assert isinstance(agentname, str),  self.logger.error("TypeError agentname has to be a string.")
        
        # Setup agent information
        self.index = index
        self.agentname = agentname
        if SocialNormMatrix is None:
            self.logger.info(self.agentname +": "+"SocialNormMatrix is not given.")
        else:
            SocialNormMatrix = np.array(SocialNormMatrix)
            self.SocialNormMatrix = SocialNormMatrix    # Used in calSocialNormEffect()
        self.CurrentValue = {"AnnualdDivRatio": None,        # Result change in annual diversion
                             "MonthlydDiv": None,       # Result change in monthly diversion
                             "UpdateDailydDiv": None,   
                             "quantile_social_adjusted": None,
                             "p_beta": None,
                             "p_prospect": None,
                             "UpdateDailyDiv": None}    # Final updated daily diversion data
        
        # Default Par
        if Par is None:
            self.Par = {"BetaDiscreteResolution": 0.01, # Determine the resolution to discretize the beta distribution
                        "RLCenter": 0.5,                # Dynamic update with the stream flow violation data
                        "RLLearning_rate_alpha":0.5,    # Learning rate for stream flow violation
                        "RLScale": 1,                   # Normalize strength for RL
                        "ProspectAlpha": 0.7,           # Risk attitude => nonlinear function
                        "ProspectBeta": 1.5,            # Risk attitude => nonlinear function
                        "Prospectk": 1,                 # Exagurate the maleffect
                        "MaxdDivRatio": 500,                 # Historical Max change of flow violation/diversion
                        "dDivRatioScale": 1                  # Scale MaxdDivRatio if needed
                        }                 
            self.Summery(Log = True)
            self.logger.warning(self.agentname +": "+"Parmeter file is not given! The above default values have been adopted.")
        else:
            assert isinstance(Par, dict),  self.logger.error("TypeError "+self.agentname +": "+"Par must be a dictionary.") 
            keys = ['BetaDiscreteResolution', 'RLCenter', 'RLLearning_rate_alpha', 'RLScale', 'ProspectAlpha', 'ProspectBeta', 'Prospectk', 'MaxdDivRatio', 'dDivRatioScale']
            assert set(keys).issubset(set(Par.keys())), self.logger.error("KeyError "+self.agentname +": "+"Missing keys in Par dictionary.")
            # To make sure all the input are numerical by converting them.
            for k in keys:
                Par[k] = float(Par[k])
            self.Par = Par
            self.Summery(Log = True)  # Only show if MsgLevel = debug
            self.logger.info(self.agentname +": "+"Par has been loaded.\n")
            
        self.BetaN = None,  # Determine the certainty of the quantile value = len(Samples)
        return None
    
    def Summery(self, Log = False):
        if Log:
            self.logger.debug(self.agentname +": "+"Summery of the agent:\n========================================" + "\nIndex: {}".format(self.index) + "\nAgentname: {}".format(self.agentname) + "\n=============> Parameters <=============\n" + "\n".join(['{:^23} :  {}'.format(keys, values) for keys,values in self.Par.items()]) + "\n=============>  Results  <==============" + "\nAnnual change in diversion: {}".format(self.CurrentValue["AnnualdDivRatio"]) + "\nMonthly change in diversion: {}".format(self.CurrentValue["MonthlydDiv"]) + "\n========================================")
        else:
            print(self.agentname +": "+"Summery of the agent:\n========================================" + "\nIndex: {}".format(self.index) + "\nAgentname: {}".format(self.agentname) + "\n=============> Parameters <=============\n" + "\n".join(['{:^23} :  {}'.format(keys, values) for keys,values in self.Par.items()]) + "\n=============>  Results  <==============" + "\nAnnual change in diversion: {}".format(self.CurrentValue["AnnualdDivRatio"]) + "\nMonthly change in diversion: {}".format(self.CurrentValue["MonthlydDiv"]) + "\n========================================")
            
    def Normaltest(self, Samples, alpha = 0.05):
        try:
            k2, p = normaltest(Samples, axis = 0, nan_policy = "omit") # pvalue: A 2-sided chi squared probability for the hypothesis test.
            Normality = True if p >= alpha/2 and p <= 1-alpha/2 else False 
            if Normality is not True:
                self.logger.debug(self.agentname +": "+"Samples is rejected for normality, which pvalue = {} for a 2-sided chi squared probability. The samples are {}".format(p, Samples))
        except:
            self.logger.exception(self.agentname +": "+"message")
        
            
    def genQuantile(self, Samples, x):
        '''
        Calculate the quantile of x by fitting Samples with normal distribution.
        '''
        # Read in samples and fit sample to normal distribution. Then, withdraw the quantile value for x.
        Samples = np.array(Samples)
        assert isinstance(x, (int, float)),  self.logger.error("TypeError " + self.agentname +": "+"x need to be a scale value.") 
        assert len(Samples.shape) == 1, self.logger.error("TypeError " + self.agentname +": "+"Samples has to be a 1-d list or 1-d array.")
        
        Samples = Samples[~np.isnan(Samples)]
        mu, std = norm.fit(Samples)
        quantile = norm.cdf(x, loc=mu, scale=std)
                
        self.CurrentValue["quantile"] = quantile
        self.BetaN = len(Samples)                   # This controls the variance of Beta distribution.
        self.logger.debug(self.agentname +": "+"BetaN has been set to {}".format(self.BetaN))
        return quantile
    
    def makeDecision(self, AgentsProbArr, strength):
        '''
        Wrap the rest of the calculation in this function.
        Note genQuantile() must run for each agent first and gather AgentsProbArr in the order of agent.index.
        '''
        # $$$$$$$$$$$$$ Step 3 $$$$$$$$$$$$$ Social norm
        # Note the order of AgentsProbArr and SocialNormMatrix row order have to be consist.
        self.calSocialNormEffect(AgentsProbArr = AgentsProbArr)
        # $$$$$$$$$$$$$ Step 4 $$$$$$$$$$$$$ To Beta distribution
        ## Turn a single value into a distribution. The standard diviation depends on the length of the
        self.BetaUpdating()
        # $$$$$$$$$$$$$ Step 5 $$$$$$$$$$$$$ Adaptive adjustment using RL
        ## Here we calculate strength using flow violation from last year 
        self.RLUpdating(strength = strength, TransformType = "sigmoid")
        # $$$$$$$$$$$$$ Step 6 $$$$$$$$$$$$$ Agent's risk attitude
        self.mappingWithProspectWeightFunction()
        # $$$$$$$$$$$$$ Step 7 $$$$$$$$$$$$$ CAlculate expected annual change of diversion (cfs-yr)
        self.dDiversion()
        # $$$$$$$$$$$$$ Step 8 $$$$$$$$$$$$$ Disaggregate into daily diversion data
        #self.Disaggregate2Daily(OriginalDailyDiv = OriginalDailyDiv)  
        return None
    
    def calSocialNormEffect(self, AgentsProbArr):
        '''
        The index of the AgentsProbArr must conrepond to agent.index and the row index of the SocialNormMatrix.
        '''
        # Blend agents own experience with neighbors'.
        # AgentsProbArr is an array of all agents quentile values sorted with their index.
        # SocialNormMatrix is a matrix defining weight to belief others and links. Row i col i is the agent i's weight to belief itself. Other place in row i are binery values indicating whether to consider agent j's opinion.
        SocialNormMatrix = self.SocialNormMatrix
        AgentsProbArr = np.array(AgentsProbArr)
        assert isinstance(AgentsProbArr, (np.ndarray, list)), self.logger.error("TypeError " + self.agentname +": "+"AgentsProbArr has to be a 1-d list or a 1-d array.")
        assert SocialNormMatrix.shape == (len(AgentsProbArr), len(AgentsProbArr)), self.logger.error("ValueError " + self.agentname +": "+"Dimension is inconsist between AgentsProbArr and SocialNormMatrix.")
  
        BeliefSelf = SocialNormMatrix[self.index, self.index]
        SumofLinks = sum(SocialNormMatrix[self.index,:])-BeliefSelf
        if SumofLinks == 0:
            assert BeliefSelf == 1,  self.logger.error("ValueError " + self.agentname +": "+"BeliefSelf must be 1 since no social links for this agent.")
            WeightArray = np.zeros(len(AgentsProbArr))
        else:
            WeightArray = SocialNormMatrix[self.index,:]*(1-BeliefSelf)/SumofLinks
        WeightArray[self.index] = BeliefSelf
        
        self.CurrentValue["quantile_social_adjusted"] = sum(AgentsProbArr*WeightArray)
        self.logger.debug(self.agentname +": "+"quantile_social_adjusted = {}".format(self.CurrentValue["quantile_social_adjusted"]))
        return None
    
    def BetaUpdating(self):
        '''
        Turn a single quatile value to a distribution using Beta updating.
        '''
        # When we don’t know anything, the probability of landing head is uniformly distributed. This is a special case of Beta, and is parametrized as Beta(⍺=1, β=1).
        N = self.BetaN
        resolution_beta = self.Par["BetaDiscreteResolution"]
        if self.CurrentValue.get("quantile_social_adjested") is None:
            P = self.CurrentValue["quantile"]
            self.logger.debug(self.agentname +": "+"No quantile_social_adjested => no social norm effect.")
        else:
            P = self.CurrentValue["quantile_social_adjested"]
        # Beta baysian updating 
        a, b = 1 + P*N, 1 + (1-P)*N
        rv = beta(a, b)        
        # Discretize beta distribution
        p_beta = rv.cdf(np.arange(0, 1+resolution_beta, resolution_beta))
        p_beta = [p_beta[i+1] - p_beta[i] for i in range(len(p_beta)-1)]
        
        self.CurrentValue["p_beta"] = p_beta
        self.logger.debug(self.agentname +": " + "Beta Bayesian updating is done.")
        return None  
    
    def RLUpdating(self, strength, TransformType = "sigmoid"):
        '''
        Dynamically update the mapping values according to strength, which is defined by flow violation.
        strength = sim_flow - obv_flow 
        strength = obv_diversion - sim_diversion
        TransformType = "sigmoid" or "linear"
        '''
        # Note that the scale variation should within +- 0.5 (Can be scaled by historical max and min values or calibrated.)
        assert isinstance(strength, (int, float)), self.logger.error(self.agentname +": "+"strength has to be a scalar.")
        center = self.Par["RLCenter"]
        scale = self.Par["RLScale"]
        learning_rate_alpha = self.Par["RLLearning_rate_alpha"]
        
        # Normalize the strength into -0.5 ~ 0.5 by either linear function or sigmoid function 
        strength = strength/scale  
        if TransformType == "linear": # Linear
            if strength > 0.5:
                strength = 0.5
            elif strength < -0.5:
                strength = -0.5
        elif TransformType == "sigmoid":
            if strength < 0:    # To avoid overflow issue
                strength = 1 - 1/(1+math.exp(strength)) - 0.5
            else:
                strength = 1/(1+math.exp(-strength)) - 0.5       
            
            
        # RL updating (center)       
        if strength >= 0:   # increase center => decrease Div 
            center_updated = center + strength*learning_rate_alpha*(1-center)
        else:               # decrease center => increase Div
            center_updated = center + strength*learning_rate_alpha*center  
            
        self.Par["RLCenter"] = center_updated  # Assign the new center.
        self.logger.debug(self.agentname +": "+"Update RLCenter to {}".format(self.Par["RLCenter"]))
        return None 
    
    def mappingWithProspectWeightFunction(self):
        '''
        Map beta distribution by propect function, which represents agent's risk attitude. 
        To plot the prospect function, use AnalysisWrap.ProspectFuncPlot()
        '''
        alpha = self.Par["ProspectAlpha"]
        beta = self.Par["ProspectBeta"]
        k = self.Par["Prospectk"]
        center = self.Par["RLCenter"]
        p = self.CurrentValue["p_beta"]
        
        if isinstance(p, (int, float)):
            p = np.array([p])
        else:
            p = np.array(p)
            
        p_new = p.copy()  # CDF
        # Scaled prospect function
        p_new_p = (p[p>=center]-center)/(1-center)
        p_new_n = (p[p<center]-center)/(center)
        
        p_new[p_new>=center] = (p_new_p**alpha)*(1-center) + center
        p_new[p_new<center] = ( -k*(-np.sign(p_new_n)*(np.abs(p_new_n))**beta) )*center + center
        
        self.CurrentValue["p_prospect"] = p_new
        self.logger.debug(self.agentname +": "+"Finish risk attitude mapping using prospect function.")
        return None
    
    def dDiversion(self):
        '''
        Calculate the expected/random value of the amount of the cahnge in annual diversion from mapped beta distribution.
        MaxdDivRatio define the maximum diversion change between two consecutive years.
        '''
        scale = self.Par["dDivRatioScale"]
        MaxdDivRatio = self.Par["MaxdDivRatio"]
        p_new = self.CurrentValue["p_prospect"]  # CDF
        center = self.Par["RLCenter"]
        interval = 2 / (len(p_new) - 1)
        dDivList = (  np.arange(-1,1 + interval, interval) - (center-0.5)*2  )*MaxdDivRatio*scale
        dDivRatioExpected = sum(dDivList*p_new)  # Expected value
        if dDivRatioExpected <= -1:
            self.logger.error("ValueError " + self.agentname +": " + "The dDivRatioExpected = {} <-1, which will result in negative diversion value. We reset to -0.9".format(dDivRatioExpected))
            dDivRatioExpected = -0.9
        
        self.CurrentValue["AnnualdDivRatio"] = dDivRatioExpected
        self.logger.debug(self.agentname +": "+"AnnualdDivRatio = {}".format(self.CurrentValue["AnnualdDivRatio"] ))
        return None
    
    # def Disaggregate2Daily(self, OriginalDailyDiv):
    #     '''
    #     Disaggregate annual dDiversion to monthly, then to daily.
    #     '''
    #     # Disaggregate the annual change into monthly change using historical standard variation in each month.
    #     assert isinstance(OriginalDailyDiv, (list, np.ndarray)), self.logger.error("TypeError " + self.agentname +": "+"OriginalDailyDiv needs to be list or array.")
    #     OriginalDailyDiv = np.array(OriginalDailyDiv).flatten()
    #     assert len(OriginalDailyDiv) == 365, self.logger.error("IndexError " + self.agentname +": "+"Length of OriginalDailyDiv must be 365.")
        
    #     dDivRatioExpected = self.CurrentValue["AnnualdDivRatio"]
    #     DailydDiv = np.array(OriginalDailyDiv*dDivRatioExpected)
    #     UpdateDiv = OriginalDailyDiv*(dDivRatioExpected+1)
        
    #     assert len(UpdateDiv[UpdateDiv<0]) == 0, self.logger.error("ValueError " + self.agentname +": " + "The daily Diversion has negative values. {}".format(UpdateDiv))
        
    #     self.CurrentValue["UpdateDailydDiv"] = DailydDiv
    #     self.CurrentValue["UpdateDailyDiv"] = UpdateDiv
    #     self.logger.debug(self.agentname +": "+"UpdateDailyDiv has been calculated.")        
        
    #     return None


#%%
        #div_month = np.array( [sum(div_day[DayinMonthAcc[i]:DayinMonthAcc[i+1]]) for i in range(12) ] )
        #div_day = np.zeros(365)
        #div_day[np.where(OriginalDailyDiv > 0)] = 1
                # for i, v in enumerate(DayinMonth):
                #     div_day_inMonth = div_day[DayinMonthAcc[i]:DayinMonthAcc[i+1]]
                #     if div_month[i] == 0:
                #         DailydDiv = DailydDiv + list(0*div_day_inMonth)
                #     else:
                #         DailydDiv = DailydDiv + list(MonthlydDiv[i]/div_month[i]*div_day_inMonth)
