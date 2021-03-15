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

df = T2.append(T6)

print(df)
sns.heatmap(df.drop(['WELL', 'DEPTH', 'DEPT'), axis=1))

# %%


    