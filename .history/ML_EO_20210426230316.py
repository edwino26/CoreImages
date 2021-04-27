# # lectura de los xlsx del procesamiento de imagenes 
# y la informacion obtenida de los registros por pozo
# %%
import numpy as np
import pandas as pd
import os
import os.path
import matplotlib.pyplot as plt  # GRAPHS
import glob
import seaborn as sns
import missingno as msno



# ===============================================

#%% ====================== Main Dataframe Computation =====================

def round_depth(number):
    return round(number * 2, 0) / 2  #Use 2 to round to closest 0.5

# Data load
T2 = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
T6 = pd.read_excel('./Excel_Files/T6.xls',sheet_name='T6_data')
U18 = pd.read_excel('./Excel_Files/U18.xls',sheet_name='U18_data')
T2_img = pd.read_excel('./Excel_Files/Processed_Images_T2.xls',sheet_name='T2').rename(columns={"DEPTH": "DEPT"})
T6_img = pd.read_excel('./Excel_Files/Processed_Images_T6.xls',sheet_name='T6').rename(columns={"DEPTH": "DEPT"})
U18_img = pd.read_excel('./Excel_Files/Processed_Images_U18.xls',sheet_name='U18').rename(columns={"DEPTH": "DEPT"})


# Conditioning before joining well dataset pairs
T2['DEPT'] = T2['DEPTH'].astype(float).apply(lambda x: round_depth(x))
T2_img['DEPT'] = T2_img['DEPT'].astype(float).apply(lambda x: round_depth(x))
T6['DEPT'] = T6['DEPTH'].astype(float).apply(lambda x: round_depth(x))
T6_img['DEPT'] = T6_img['DEPT'].astype(float).apply(lambda x: round_depth(x))
U18['DEPT'] = U18['TDEP'].astype(float).apply(lambda x: round_depth(x))
U18_img['DEPT'] = U18_img['DEPT'].astype(float).apply(lambda x: round_depth(x))

# Join only depths present at both datasets in a pair
T2 = T2_img.merge(T2, on='DEPT', how='inner').dropna()
T6 = T6_img.merge(T6, on='DEPT', how='inner').dropna()
U18 = U18_img.merge(U18, on='DEPT', how='inner').dropna()

# Specific curves to be used in ML algorithm
logset = ['DEPT', 'GR_EDTC', 'RHOZ', 'AT90', 'DTCO', 'NPHI', 'WELL', 'GRAY']

df = T2[logset].append(T6[logset]).append(U18[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "RT", "RHOZ": "RHOB", "WELL": "Well"}).set_index('Well')
df.reset_index()


# ===================== Descriptive Statistics =====================
print(df.groupby(['Well']).describe())

# %%
#sns.heatmap(df.drop(['DEPT'], axis=1).isnull())
#sns.heatmap(df.isnull(), cbar=False)
# %%
#msno.matrix(df)
# %%
#msno.heatmap(df)
    

# %%
