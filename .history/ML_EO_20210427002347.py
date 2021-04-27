# # lectura de los xlsx del procesamiento de imagenes 
# y la informacion obtenida de los registros por pozo
# %%
import numpy as np
import pandas as pd
import os
import os.path
import matplotlib.pyplot as plt  # GRAPHS
from mlxtend.plotting import scatterplotmatrix
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

# U18 had too many images compared to other 2 wells. Take a smaller set to have a more balanced dataset
U18 = U18[U18['DEPT']>771]
U18 = U18[U18['DEPT']<925]

# Join only depths present at both datasets within a pair
T2 = T2_img.merge(T2, on='DEPT', how='inner').dropna()
T6 = T6_img.merge(T6, on='DEPT', how='inner').dropna()
U18 = U18_img.merge(U18, on='DEPT', how='inner').dropna()

# Specific curves to be used in ML algorithm
logset = ['DEPT', 'GR_EDTC', 'RHOZ', 'AT90', 'DTCO', 'NPHI', 'WELL', 'GRAY']

#df = T2[logset].append(T6[logset]).append(U18[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "RT", "RHOZ": "RHOB", "WELL": "Well"}).set_index('Well')
df = T2[logset].append(T6[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "RT", "RHOZ": "RHOB", "WELL": "Well"}).set_index('Well')

df.reset_index(inplace=True)


# %%
# ===================== Descriptive Statistics and Plots =====================
print(df.groupby(['Well']).median())
print(df.groupby(['Well']).count())
print(df.groupby(['Well']).min())
print(df.groupby(['Well']).max())

# Show GR distribution among wells
fig,ax = plt.subplots()
hatches = ('\\', '//', '..')         # fill pattern
alpha_v = 0.9
for (i, d),hatch in zip(df.groupby('Well'), hatches):
    d['GR'].hist(alpha=alpha_v, ax=ax, label=i, hatch=hatch)
    alpha_v -= 0.3
ax.legend()

#Show how the gray scale varies among each photo
fig,ax = plt.subplots()
hatches = ('\\', '//', '..')         # fill pattern
alpha_v = 0.9
for (i, d),hatch in zip(df.groupby('Well'), hatches):
    d['GRAY'].hist(alpha=alpha_v, ax=ax, label=i, hatch=hatch)
    alpha_v -= 0.3
ax.legend()

# Expected correlations between variables
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(df['GRAY'])
axs[1, 0].plot(df['GR'], df['GRAY'], linestyle='None', markersize=4, marker='o')
axs[0, 1].plot(df['RHOB'], df['GRAY'], linestyle='None', markersize=4, marker='o')
axs[1, 1].hist2d(df['RT'], df['GRAY'])
plt.show()

# Matrix Plot
scatterplotmatrix(df, figsize=(10, 8))
plt.tight_layout()
plt.show()

#
variables= ['GR', 'RHOB', 'RT', 'DTCO', 'NPHI', 'GRAY']
fig, axes = scatterplotmatrix(df[df['Well']=='T2'].drop(['Well', 'DEPT'], axis=1).values, figsize=(10, 8), alpha=0.5)
fig, axes = scatterplotmatrix(df[df['Well']=='T6'].drop(['Well', 'DEPT'], axis=1).values, fig_axes=(fig, axes), alpha=0.5)
fig, axes = scatterplotmatrix(df[df['Well']=='U18'].drop(['Well', 'DEPT'], axis=1).values, fig_axes=(fig, axes), alpha=0.5, names=variables)
plt.tight_layout()
plt.show()


# %%
#sns.heatmap(df.drop(['DEPT'], axis=1).isnull())
#sns.heatmap(df.isnull(), cbar=False)
# %%
#msno.matrix(df)
# %%
#msno.heatmap(df)
    

# %%
