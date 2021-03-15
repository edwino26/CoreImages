# # lectura de los xlsx del procesamiento de imagenes 
# y la informacion obtenida de los registros por pozo
# %%
import numpy as np
import pandas as pd
import lasio
import os
import os.path
import matplotlib.pyplot as plt  # GRAPHS
import glob

#%%
t2=pd.read_excel('./Excel_Files/T2.xlsm',sheet_name='T2_data')
Img_t2=pd.read_excel('./Processed_Images_T2.xlsx',sheet_name='T2')
t6=pd.read_excel('./T6.xlsx',sheet_name='T6_data')
Img_t6=pd.read_excel('./Processed_Images_T6.xlsx',sheet_name='T6')
u18=pd.read_excel('./U18.xlsx',sheet_name='U18_data')
df= pd.DataFrame()
df = df.append((t2,t6,u18), ignore_index=False)
df1 = Img_t2[['DEPTH', 'GRAY','PHOTO']]
print(df1)
# %%
with pd.ExcelWriter('All_Wells.xlsx') as writer:  
    df1.to_excel(writer, sheet_name='Wells_data')

    