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




#%%
T2 = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
T6 = pd.read_excel('./Excel_Files/T6.xls',sheet_name='T6_data')
U18 = pd.read_excel('./Excel_Files/U18.xls',sheet_name='U18_data')

Img_T2 = pd.read_excel('./Excel_Files/Processed_Images_T2.xls',sheet_name='T2')
Img_T6 = pd.read_excel('./Excel_Files/Processed_Images_T6.xls',sheet_name='T6')
Img_U18 = pd.read_excel('./Excel_Files/Processed_Images_U18.xls',sheet_name='U18')


#Get subIm array in which PHOTO =1 to get rid of incorrect value ointerpolation between images
Img_U18 = g_U18[Img_U18['PHOTO'].apply(lambda x: 1 if x.is_integer())]


dep= np.arange(200,1350,0.5)
f = interpolate.interp1d(df4['DEPTH'], df4['RHOZ'])
RHOZ_new = f(dep)


# u18=pd.read_excel('./Excel_Files/U18.xls',sheet_name='U18_data')
# df= pd.DataFrame()

logset = ['DEPT', 'GR_EDTC', 'RHOZ', 'AT90', 'DTCO', 'NPHI', 'WELL']

df1 = T2[logset].append(T6[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "RT", "RHOZ": "RHOB"})
#.set_index("WELL")





#print(df)
sns.heatmap(df.drop(['WELL', 'DEPTH', 'DEPT'], axis=1).isnull())

#sns.heatmap(df.isnull(), cbar=False)

# %%
msno.matrix(df)
# %%
msno.heatmap(df)

    
# %%
