import numpy as np
import pandas as pd


# Data load
T2 = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
T6 = pd.read_excel('./Excel_Files/T6.xls',sheet_name='T6_data')
U18 = pd.read_excel('./Excel_Files/U18.xls',sheet_name='U18_data')




# %% 
# Resample Logs
resample = 0
if resample ==1:
        
    dep = np.arange(min(T2.DEPT),max(T2.DEPT),0.25) 
    T2_rs = pd.DataFrame(columns=[T2.columns])
    T2_rs.iloc[:,0] = dep
    for i in range(len(T2.columns)):
        if i == len(T2.columns)-1:
            T2_rs.iloc[:,i] = T2.WELL[0]
        else:
            f = interpolate.interp1d(T2.DEPT, T2.iloc[:,i])
            T2_rs.iloc[:,i] =f(dep)
    T2_rs.dropna(inplace=True)
    T2_rs.to_excel("./Excel_Files/T2_rs.xls", sheet_name='T2_data')    

    dep = np.arange(min(T6.DEPT),max(T6.DEPT),0.25) 
    T6_rs = pd.DataFrame(columns=[T6.columns])
    T6_rs.iloc[:,0] = dep
    for i in range(len(T6.columns)):
        if i == len(T6.columns)-1:
            T6_rs.iloc[:,i] = T6.WELL[0]
        else:
            f = interpolate.interp1d(T6.DEPT, T6.iloc[:,i])
            T6_rs.iloc[:,i] =f(dep)
    T6_rs.dropna(inplace=True)
    T6_rs.to_excel("./Excel_Files/T6_rs.xls", sheet_name='T6_data') 

    dep = np.arange(min(U18.TDEP),max(U18.TDEP),0.25) 
    U18_rs = pd.DataFrame(columns=[U18.columns])
    U18_rs.iloc[:,0] = dep
    for i in range(len(U18.columns)):
        if i == len(U18.columns)-1:
            U18_rs.iloc[:,i] = U18.WELL[0]
        else:
            f = interpolate.interp1d(U18.TDEP, U18.iloc[:,i])
            U18_rs.iloc[:,i] =f(dep)
    U18_rs.dropna(inplace=True)
    U18_rs.to_excel("./Excel_Files/U18_rs.xls", sheet_name='U18_data') 
