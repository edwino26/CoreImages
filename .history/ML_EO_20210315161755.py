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


#%%
T2 = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
Img_T2 = pd.read_excel('./Excel_Files/Processed_Images_T2.xls',sheet_name='T2')
T6 = pd.read_excel('./Excel_Files/T6.xls',sheet_name='T6_data')
Img_T6 = pd.read_excel('./Excel_Files/Processed_Images_T6.xls',sheet_name='T6')

# u18=pd.read_excel('./Excel_Files/U18.xls',sheet_name='U18_data')
# df= pd.DataFrame()

df = df.append((T2,T6), ignore_index=False)
# df1 = Img_t2[['DEPTH', 'GRAY','PHOTO']]
print(t2)

# %%
with pd.ExcelWriter('All_Wells.xls') as writer:  
    df1.to_excel(writer, sheet_name='Wells_data')

    