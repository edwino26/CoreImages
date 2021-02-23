# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# %%
('pip3 install dlisio')


# %%
import dlisio
import numpy as np
import pandas as pd
import math
import lasio
import matplotlib.pyplot as plt  # GRAPHS
import glob
from matplotlib import rcParams
from pandas import ExcelWriter #este va servir para exportar achivos a excel 
import matplotlib.pyplot as plt
from pandas import ExcelWriter #este va servir para exportar achivos a excel 
import xlrd


# %%

las2= lasio.read('./LAS/U18/U18_GR.las')
df2= las2.df()
df2.reset_index(inplace=True)
df2 = df2[['GR_EDTC', "TDEP"]]

las3 = lasio.read('./LAS/U18/U18_AT90_NPHI.las')
df3 = las3.df()
df3.reset_index(inplace=True)
df3 = df3[['AT90', 'TDEP']]

datos =pd.read_excel('./U18_x/U18_RHOZ1.xlsx' ,sheet_name = 'RHOZ')
datos1 = datos[["RHOZ","DEPTH"]]
print(datos1)

las5 = lasio.read('./LAS/U18/U18_AT90_NPHI.las')
df5 = las5.df()
df5.reset_index(inplace=True)
df5 = df5[['NPHI', "TDEP"]]


las6 = lasio.read('./LAS/U18/U18_DTCO.las')
df6 = las6.df()
df6.reset_index(inplace=True)
df6 = df6[['DTCO', "TDEP"]]


# %%

dt = 700
bt=1020

plt.figure(figsize=(15,9))
plt.subplot(171)
plt.plot(df2.GR_EDTC,df2.TDEP,'g', lw=0.5)
plt.ylabel('DEPTH')
plt.title('GR_EDTC ')
plt.xlabel('Gamma Ray ')
plt.axis([20, 130, dt,bt])
plt.gca().invert_yaxis()
plt.grid(True)
plt.hlines(y=710, xmin=0, xmax=130)
plt.hlines(y=1010, xmin=0, xmax=130)


plt.subplot(172)
plt.plot(df3.AT90, df3.TDEP, 'c',lw=0.5)
plt.title('AT90 ')
plt.xlabel('Resistivity')
plt.axis([10, 980, dt,bt])
plt.xscale('log')
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=710, xmin=0, xmax=980)
plt.hlines(y=1010, xmin=0, xmax=980)

plt.subplot(173)
plt.plot( datos1.RHOZ,datos1.DEPTH,'black',lw=0.5)
plt.title('RHOZ')
plt.xlabel('Standard \n Resolution \n Formation \n Density') #\n ( G/C3)'  DENTRO DEL PARENTESIS
plt.axis([2.25, 2.65,dt,bt])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=710, xmin=0, xmax=130)
plt.hlines(y=1010, xmin=0, xmax=130)


plt.subplot(174)
plt.plot(df5.NPHI, df5.TDEP, 'purple',lw=0.5)
plt.title('NPHI')
plt.xlabel('Thermal \n Neutron \n Porosity')
plt.axis([0.1, 0.8, dt,bt])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=710, xmin=0, xmax=130)
plt.hlines(y=1010, xmin=0, xmax=130)

plt.subplot(175)
plt.plot(df6.DTCO,df6.TDEP,'red',lw=0.5)
plt.title('DTCO ')
plt.xlabel('Delta-T \n Compressional ')
plt.axis([60,125, dt,bt])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=710, xmin=0, xmax=130)
plt.hlines(y=1010, xmin=0, xmax=130)



plt.suptitle('Umiat 18 WELL LOGS Alaska'+ las2.well['STAT']['value'])



plt.show()


# %%



