# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import math
import lasio
import matplotlib.pyplot as plt  # GRAPHS
import glob
from matplotlib import rcParams

# %%

las1= lasio.read('./LAS/U18/U18_GR.las')
df1= las1.df()
df1.reset_index(inplace=True)
df1 = df1[['GR_EDTC', 'TDEP']]

las2 = lasio.read('./LAS/U18/U18_AT90_NPHI.las')
df2 = las2.df()
df2.reset_index(inplace=True)
df2 = df2[['AT90','NPHI','TDEP',]]

las3 = lasio.read('./LAS/U18/U18_DTCO.las')
df3= las3.df()
df3.reset_index(inplace=True)
df3 = df3[['DTCO', 'TDEP']]

# REUNI LOS 3 LAS EN 1 SOLO 
frames = [df1, df2, df3]
result = pd.merge(df1,df2)
df5=pd.merge(result,df3)

# %%

# EXCEL DE DATA U18.XLSX TIENE LOS DATOS TAL CUAL ENVIADOS POR VALENTINA SIN CAMBIOS 
U18_xl =pd.read_excel('./data U18.xlsx',sheet_name = 'DATA')
df4=U18_xl[['DEPTH','RHOZ']]

# array con nueva tabla para TDEP (prof) con paso de 0.5
dep= np.arange(200,1350,0.5)



## interpolacion de valores del RHOZ
# df4.set_index(df4.DEPTH)
#df = df4.interpolate()
##print("Interpolated DataFrame:")
#df6=pd.merge(df5,df)

#df7= df6.loc[(df6.TDEP>=700) &  (df6.TDEP<=1100)]
#print(df7.head)


# %%
with pd.ExcelWriter('U18_info1.xlsx') as writer:  
    df5.to_excel(writer, sheet_name='U181_frames')
   
