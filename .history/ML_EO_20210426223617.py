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

#%% =================================
# ===============================================


def round_depth(number):
    return round(number * 2, 0) / 2  #Use to to round to closest 0.5


T2 = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
T6 = pd.read_excel('./Excel_Files/T6.xls',sheet_name='T6_data')
U18 = pd.read_excel('./Excel_Files/U18.xls',sheet_name='U18_data')

T2_img = pd.read_excel('./Excel_Files/Processed_Images_T2.xls',sheet_name='T2').rename(columns={"DEPTH": "DEPT"})
T6_img = pd.read_excel('./Excel_Files/Processed_Images_T6.xls',sheet_name='T6').rename(columns={"DEPTH": "DEPT"})
U18_img = pd.read_excel('./Excel_Files/Processed_Images_U18.xls',sheet_name='U18').rename(columns={"DEPTH": "DEPT"})



T2['DEPT'] = T2['DEPTH'].astype(float)
T2_img['DEPT'] = T2_img['DEPT'].astype(float)
T2['DEPT'] = T2['DEPT'].apply(lambda x: round_depth(x))
T2_img['DEPT'] = T2_img['DEPT'].apply(lambda x: round_depth(x))
T2.sort_values(by=['DEPT'], inplace=True)

T6['DEPT'] = T6['DEPTH'].astype(float)
T6_img['DEPT'] = T6_img['DEPT'].astype(float)
T6['DEPT'] = T6['DEPT'].apply(lambda x: round_depth(x))
T6_img['DEPT'] = T6_img['DEPT'].apply(lambda x: round_depth(x))
T6.sort_values(by=['DEPT'], inplace=True)


T2 = T2_img.merge(T2, on='DEPT', how='inner').dropna()
T6 = T6_img.merge(T6, on='DEPT', how='inner').dropna()

logset = ['DEPT', 'GR_EDTC', 'RHOZ', 'AT90', 'DTCO', 'NPHI', 'WELL', 'GRAY']

df = T2[logset].append(T6[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "RT", "RHOZ": "RHOB"}).set_index('WELL')

# a = T2['DEPT'][602]
# print(a)
# b =T2_img['DEPT'][0]
# print(b)
# if a==b:
#     print("EQUALLLLLLLLLLLLLLLLLLLLL")


print('===========JOINT BELOW =========================')
print(dflogs)


# %%
sns.heatmap(df.drop(['DEPT'], axis=1).isnull())
sns.heatmap(df.isnull(), cbar=False)
# %%
msno.matrix(df)
# %%
msno.heatmap(df)
    

# %%
