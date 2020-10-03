# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:24:37 2020

@author: CYLin
"""
import os
import pandas as pd
import numpy as np

def readHydromet(FileName, GageName, ColumnNameDict = {"AF": "Storage (acre-feet)", "QD": "Discharge (cfs)", "QJ": "Canal_Discharge (cfs)"}):
    assert os.path.isfile(FileName), "The directory is not exist {}".format(FileName)
    # If it shows error, probably the "NO RECORD" is in the data file
    df = pd.read_csv(FileName, sep = "\s+", parse_dates = ["DATE"], index_col = "DATE")
    df[df == "MISSING"] = np.nan
    df[df == "NO_RECORD"] = np.nan
    df = df.astype('float')
    df[df < -90] = np.nan
    ColumnName = [ColumnNameDict.get(c) for c in list(df)]
    ColumnName = ["Data" if c is None else c for c in ColumnName ]
    ColumnName = [GageName + "_" + c for c in ColumnName]
    df.columns = ColumnName
    return df

def calWthData(DataPath = r"C:\Users\Philip\OneDrive\Lehigh\0_Yakima\DATA\crb_monthly_met\crb", 
               GridSettingFile = r"C:\Users\Philip\OneDrive\Lehigh\0_Yakima\DATA\crb_monthly_met/Yakima_Wth_Grid.csv"):
    df_coor = pd.read_csv(GridSettingFile)
    
    RegionName = list(set(df_coor["Name"]))
    df = pd.DataFrame()
    for r in RegionName:
        WthName = list(df_coor[df_coor["Name"] == r]["WthFileName"])
        Percentage = df_coor[df_coor["Name"] == r]["PERCENTAGE"]
        Percentage = list(Percentage/sum(Percentage))
        print(Percentage)
        df_temp = pd.DataFrame()
        for i, f in enumerate(WthName):
            df_f = pd.read_csv(os.path.join(DataPath, f), sep="\s+", header = None)
            df_f.columns = ["Year", "Month", "Prep", "Tmax", "Tmin", "Wind"]
            df_temp[f] = df_f["Prep"]*Percentage[i]
        df[r] = df_temp.sum(axis = 1)
            
    rng = pd.date_range(start = "1949-1", end = "2011-1", freq = "M")
    df.index = rng
    return df

# df = calWthData()
# df.to_csv(r"C:\Users\Philip\OneDrive\Lehigh\0_Yakima\DATA/EdMaurerPrepYakima.csv")
# df_year = df.resample("Y").sum()
# df_year.to_csv(r"C:\Users\Philip\OneDrive\Lehigh\0_Yakima\DATA/EdMaurerPrepYakima_year.csv")


def readGenCSV(FileName):
        df = pd.read_csv(FileName, index_col = 0, parse_dates = True)
        return df
# =============================================================================    

import matplotlib.pyplot as plt
from scipy import stats

class PyRAMIDAnalysis(object):
    def __init__(self, x_obv, y_sim, name = None, x_label = "Obv", y_label = "Sim", indicators = "All"):
        x_obv = np.array(x_obv).flatten(); y_sim = np.array(y_sim).flatten() # Make sure the data type in 1-D array
        # Can delete
        assert isinstance(x_obv, (list, np.ndarray)), "x_obv has to be 1-D list or array. x_obv = {}".format(x_obv)
        assert isinstance(y_sim, (list, np.ndarray)), "y_sim has to be 1-D list or array. y_sim = {}".format(y_sim)
        assert x_obv.shape == y_sim.shape, "Dimension of x_obv and y_sim are not consist.{} & {}".format(x_obv.shape, y_sim.shape)
        
        self.x_obv = x_obv
        self.y_sim = y_sim
        if name is None:
            self.name = ""
        else:
            self.name = name
            
        self.x_label = x_label
        self.y_label = y_label
        
        self.indicators = self.calIndicators()
        if indicators != "All" and isinstance(indicators, list):
            self.indicators = self.indicators[indicators]
               
        return None
    
    def calIndicators(self):
        '''
        r   : Correlation of correlation
        r2  : Coefficient of determination
        rmse: Root mean square error
        NSE : Nash–Sutcliffe efficiency
        CP  : Correlation of persistence
        RSR : RMSE-observations standard deviation ratio 
        KGE : Kling–Gupta efficiency
        
        Returns
        -------
        indicators : dict
        '''
        x_obv = self.x_obv
        y_sim = self.y_sim
        mask = ~np.isnan(x_obv) & ~np.isnan(y_sim)  # Mask to ignore nan
        
        # Indicator calculation
        mu_sim = np.nanmean(y_sim); mu_obv = np.nanmean(x_obv)
        sig_sim = np.nanstd(y_sim); sig_obv = np.nanstd(x_obv)
        
        indicators = {}
        indicators["r"] = np.corrcoef(x_obv[mask], y_sim[mask])[0,1]
        indicators["r2"] = indicators["r"]**2
        indicators["rmse"] = np.nanmean((x_obv[mask]-y_sim[mask])**2)**0.5
        indicators["NSE"] = 1 - np.nansum((x_obv[mask]-y_sim[mask])**2)/np.nansum((x_obv[mask]-mu_obv)**2) # Nash
        indicators["CP"] = 1 - np.nansum((x_obv[1:]-y_sim[1:])**2)/np.nansum((x_obv[1:]-x_obv[:-1])**2)
        indicators["RSR"] = indicators["rmse"]/sig_obv
        indicators["KGE"] = 1 - ((indicators["r"]-1)**2 + (sig_sim/sig_obv - 1)**2 + (mu_sim/mu_obv - 1)**2)**0.5
        print(self.name)
        print("\n".join(['{:^10} :  {}'.format(keys, values) for keys,values in indicators.items()]))
        
        self.indicators = indicators
        # r = np.corrcoef(x_obv[mask], y_sim[mask])[0,1]
        # r2 = r**2
        # rmse = np.nanmean((x_obv[mask]-y_sim[mask])**2)**0.5
        # NSE = 1 - np.nansum((x_obv[mask]-y_sim[mask])**2)/np.nansum((x_obv[mask]-np.nanmean(x_obv[mask]))**2) # Nash
        # CP = 1 - np.nansum((x_obv[1:]-y_sim[1:])**2)/np.nansum((x_obv[1:]-x_obv[:-1])**2)
        # RSR = rmse/np.nanstd(x_obv)
        
        return indicators
    
    def RegPlot(self, Title = None, xyLabal = None, SameXYLimit = True):
        
        x_obv = self.x_obv
        y_sim = self.y_sim
        
        if Title is None:
            Title = "Regression" + self.name
        else:
            Title = Title + self.name
        
        if xyLabal is None:
            x_label = self.x_label; y_label = self.y_label
        else:
            x_label = xyLabal[0]; y_label = xyLabal[1]
            
        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(Title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Regression calculation and plot
        mask = ~np.isnan(x_obv) & ~np.isnan(y_sim)  # Mask to ignore nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_obv[mask], y_sim[mask]) # Calculate the regression line
        line = slope*x_obv+intercept                # For plotting regression line
        ax.plot(x_obv, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope, intercept))
        # end
        
        # # Indicator calculation
        # r = np.corrcoef(x_obv[mask], y_sim[mask])[0,1]
        # r2 = r**2
        # rmse = np.nanmean((x_obv[mask]-y_sim[mask])**2)**0.5
        # CE = 1 - np.nansum((x_obv[mask]-y_sim[mask])**2)/np.nansum((x_obv[mask]-np.nanmean(x_obv[mask]))**2) # Nash
        # CP = 1 - np.nansum((x_obv[1:]-y_sim[1:])**2)/np.nansum((x_obv[1:]-x_obv[:-1])**2)
        
        # Plot data point
        ax.scatter(x_obv, y_sim, color="k", s=3.5)
        ax.legend(fontsize=9, loc = 'upper right')
        if SameXYLimit:
            Max = max([np.nanmax(x_obv),np.nanmax(y_sim)]); Min = min([np.nanmin(x_obv),np.nanmin(y_sim)])
            ax.set_xlim(Min,Max)
            ax.set_ylim(Min,Max)
            
        # PLot indicators
        Name = {"r": "$r$",
                "r2":"$r^2$",
                "rmse":"RMSE",
                "NSE": "NSE",
                "CP": "CP",
                "RSR": "RSR",
                "KGE": "KGE"}
        string = "\n".join(['{:^10} :  {}'.format(Name[keys], values) for keys,values in self.indicators.items()])
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.annotate(string, xy= (0.05, 0.95), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, fontsize=9, bbox = props)       
        plt.show()
        return ax
    
    def TimeseriesPlot(self, x = None, Title = None, xyLabal = None, ax = None):
        if x is not None:
            assert isinstance(x, (list, np.ndarray)), print("x has to be list, or array")
        x_obv = self.x_obv
        y_sim = self.y_sim
        
        if Title is None:
            Title = "Timeseries" + self.name
        else:
            Title = Title + self.name
        
        if xyLabal is None:
            x_label = "Time"; y_label = "Value"
        else:
            x_label = xyLabal[0]; y_label = xyLabal[1]
        
        if x is None:
            x = np.arange(0,len(x_obv))
        else:
            assert len(x) == len(x_obv), print("Input length of x is not corresponding to the length of data.")
            # try:
            #     x = pd.date_range(start=x[0], end=x[1])
            # except:
            #     x = x
        
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, x_obv, label = self.x_label)
        ax.plot(x, y_sim, linestyle='dashed', label = self.y_label)
        #ax.bar(x, np.nan_to_num(y_obv-y_sim), label = "Hydromet - YAKRW", color = "red")
        ax.legend(fontsize=9)
        ax.set_title(Title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        #ax.set_xticks(pd.date_range(start='1/1/1966', end='12/31/2005'))
        if ax is None:
            plt.show()
        return ax
    
    
def ProspectFuncPlot(alpha, beta, center = 0.5, k = 1, Title = None, xyLabal = None):
        
    if Title is None:
        Title = "Prospect Function" 
    else:
        Title = Title 
    
    if xyLabal is None:
        x_label = "Original Input"; y_label = "Adjusted Output"
    else:
        x_label = xyLabal[0]; y_label = xyLabal[1]
    
    p = np.arange(0,1.001,0.001)
    p_new = p.copy()
    
    p_new_p = (p[p>=center]-center)/(1-center)
    p_new_n = (p[p<center]-center)/(center)
    
    p_new[p_new>=center] = (p_new_p**alpha)*(1-center) + center
    p_new[p_new<center] = ( -k*(-np.sign(p_new_n)*(np.abs(p_new_n))**beta) )*center + center
    

    # Create dots
    x_dots = np.arange(0,1.025,0.025)
    y_dots = x_dots.copy()
    
    y_dots_p = (x_dots[x_dots>=center]-center)/(1-center)
    y_dots_n = (x_dots[x_dots<center]-center)/(center)
    
    y_dots[y_dots>=center] = (y_dots_p**alpha)*(1-center) + center
    y_dots[y_dots<center] = ( -k*(-np.sign(y_dots_n)*(np.abs(y_dots_n))**beta) )*center + center
    

    fig, ax = plt.subplots()
    ax.plot(p, p_new)
    ax.scatter(x_dots, [0]*len(x_dots), s = 5, color = "black", marker = "x")
    ax.scatter([0]*len(x_dots), y_dots, s = 5, color = "black", marker = "x")
    #ax.legend(fontsize=9)
    ax.set_title(Title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    plt.show()
    return ax