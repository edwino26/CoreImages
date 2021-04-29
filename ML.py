# # lectura de los xlsx del procesamiento de imagenes 
# y la informacion obtenida de los registros por pozo
# %%
import numpy as np
import pandas as pd
import lasio
import cv2
import os
import matplotlib.pyplot as plt  # GRAPHS
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#%%
#T2
t2=pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
t2_img= pd.read_excel('./Excel_Files/Processed_Images_T2.xls',sheet_name='T2')
df = t2_img[['DEPTH','GRAY','PHOTO']]
df.set_index('DEPTH',inplace=True)
df1 = t2[['DEPT','DEPTH',"GR_EDTC", "RHOZ","AT90","NPHI","DTCO",'WELL']]
df1.set_index('DEPT',inplace=True)
df2=pd.DataFrame(df1.join(df,how='inner'))
#df2.head(300)

#T6
t6=pd.read_excel('./Excel_Files/T6.xls',sheet_name='T6_data')
t6_img= pd.read_excel('./Excel_Files/Processed_Images_T6.xls',sheet_name='T6')
df3 = t6_img[['DEPTH','GRAY','PHOTO']]
df3.set_index('DEPTH',inplace=True)
df4 = t6[['DEPT','DEPTH',"GR_EDTC", "RHOZ","AT90","NPHI","DTCO",'WELL']]
df4.set_index('DEPT',inplace=True)
df5=pd.DataFrame(df4.join(df3,how='inner'))
# df5.head(300)

#df5 bajo df2
vertical_stack = pd.concat([df2, df5], axis=0)
print(vertical_stack)
print(vertical_stack.head())
with pd.ExcelWriter('ML-Wells.xlsx') as writer:  
    vertical_stack.to_excel(writer, sheet_name='Wells_data')
   
