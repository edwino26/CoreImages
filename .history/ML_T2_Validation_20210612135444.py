#T2 TEST DATA
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import interpolate
from scipy.integrate import simps
from numpy import trapz
from sklearn.metrics import mean_squared_error

# %%

#Load Stack
UVStack = pd.read_excel('./ML_Results/T2_test/ImgStack.xls')
ImgStackk = UVStack.copy().to_numpy()

# %%

def integrate(y_vals, h):
    i = 1
    total = y_vals[0] + y_vals[-1]
    for y in y_vals[1:-1]:
        if i % 2 == 0:
            total += 2 * y
        else:
            total += 4 * y
        i += 1
    return total * (h / 3.0)


# %% Load and resample "results" (res) file

sub = pd.read_excel('./ML_Results/T2_test/sub.xls')
res = pd.read_excel('./ML_Results/T2_test/Results.xls')
res = res[res.Well == 'T2']
res.sort_values(by=['DEPT'])
res.drop(['Unnamed: 0', 'Set'], axis=1, inplace=True)
res.reset_index(inplace=True, drop=True)

dep = np.arange(min(res.DEPT), max(res.DEPT),0.5) #res is not at 0.5 thanks to balancing
res_rs = pd.DataFrame(columns=[res.columns])
res_rs.DEPT = dep

for i in range(len(res.columns)):
    if i != 8:
        f = interpolate.interp1d(res.DEPT, res.iloc[:,i])
        res_rs.iloc[:,i] =f(dep)
    else:
        res_rs.iloc[:,i] = res.Well[0]  
#T2_rs.dropna(inplace=True)

res = res_rs.copy()
difference = res.DEPT.diff()
difference.describe()


# %%
TT = pd.read_excel('./ML_Results/Train_Test_Results.xls')


istr = 0
iend = 42344
dplot_o = 3671
dplot_n = 3750
shading = 'bone'

# %% Load Log Calculations

T2_x = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
T2_x = T2_x[['DEPTH','GR_EDTC','RHOZ','AT90','NPHI','Vsh','Vclay','grain_density','porosity',
                   'RW2','Sw_a','Sw_a1','Sw_p','Sw_p1','SwWS','Swsim','Swsim1','PAY_archie',
                    'PAY_poupon','PAY_waxman','PAY_simandoux']]


# %%


T2_rs = pd.DataFrame(columns=[T2_x.columns])
T2_rs.iloc[:,0] = dep

for i in range(len(T2_x.columns)):
    f = interpolate.interp1d(T2_x.DEPTH, T2_x.iloc[:,i])
    T2_rs.iloc[:,i] =f(dep)
#T2_rs.dropna(inplace=True)

T2_x = T2_rs.copy()
difference_T2 = T2_x.DEPTH.diff()
difference.describe()

# %%
plt.figure()
plt.subplot2grid((1, 10), (0, 0), colspan=3)
plt.plot(sub['GRAY'], sub['DEPTH'], 'mediumseagreen', linewidth=0.5);
plt.axis([50, 250, dplot_o, dplot_n]);
plt.gca().invert_yaxis();
plt.fill_between(sub['GRAY'], 0, sub['DEPTH'], facecolor='green', alpha=0.5)
plt.xlabel('Gray Scale RGB')

plt.subplot2grid((1, 10), (0, 3), colspan=7)
plt.imshow(ImgStackk[istr:iend,80:120], aspect='auto', origin='upper', extent=[0,1,dplot_n,dplot_o], cmap=shading);
plt.axis([0, 1, dplot_o, dplot_n]);
plt.gca().invert_yaxis()
plt.xlabel('Processed Image')
plt.colorbar()
p_50 = np.percentile(sub['DEPTH'], 50)
plt.yticks([]); plt.xticks([])
plt.subplots_adjust(wspace = 20, left = 0.1, right = 0.9, bottom = 0.1, top = 0.9)
plt.show()
# %%
CORE =pd.read_excel('./CORE/CORE.xlsx',sheet_name='XRD')
mask = CORE.Well.isin(['T2'])
T2_Core = CORE[mask]
prof=T2_Core['Depth']
clays=T2_Core['Clays']

xls1 = pd.read_excel ('./CORE/CORE.xlsx', sheet_name='Saturation')
mask = xls1.Well.isin(['T2'])
T2_sat = xls1[mask]
long=T2_sat  ['Depth']
poro=T2_sat  ['PHIT']
grain=T2_sat  ['RHOG']
sw_core=T2_sat  ['Sw']
klinkenberg = T2_sat ['K']

minimo=grain.min()
maximo=grain.max()
c=2.65
d=2.75
norm=(((grain-minimo)*(d-c)/(maximo-minimo))+c)

xls2 = pd.read_excel ('./CORE/CORE.xlsx', sheet_name='Gamma')
mask = xls2.Well.isin(['T2'])
T2_GR = xls2[mask]
h=T2_GR['Depth']
cg1=T2_GR['GR_Scaled']

# %%
# ~~~~~~~~~~~~~~~~~~ Plot Results ~~~~~~~~~~~~~~~~~~~~~~
ct = 0
top= dplot_o 
bottom= dplot_n 
no_plots = 9

ct+=1
plt.figure(figsize=(13,9))
plt.subplot(1,no_plots,ct)
plt.plot (T2_x.GR_EDTC,T2_x.DEPTH,'g', lw=3)
#plt.fill_between(T2_x.GR_EDTC.values.reshape(-1), T2_x.DEPTH.values.reshape(-1), y2=0,color='g', alpha=0.8)
plt.title('$Gamma Ray$',fontsize=8)
plt.axis([40,130,top,bottom])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('Gamma Ray ',fontsize=6)
plt.gca().invert_yaxis()
plt.grid(True)
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

ct+=1
plt.subplot(1,no_plots,ct)
plt.plot (T2_x.PAY_poupon,T2_x.DEPTH,'r',lw=0.5)
h_P = integrate(T2_x.PAY_poupon.values, 0.5)
plt.title('$PAY Poupon$',fontsize=8)
plt.fill_between(T2_x.PAY_poupon.values.reshape(-1),T2_x.DEPTH.values.reshape(-1), color='r', alpha=0.8)
plt.axis([0.01,0.0101,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

#Waxman-Smits
ct+=1
plt.subplot(1,no_plots,ct)
plt.plot (T2_x.PAY_waxman,T2_x.DEPTH,'g',lw=0.5)
h_WS = integrate(T2_x.PAY_waxman.values, 0.5)

plt.title('$PAY Waxman$',fontsize=8)
plt.fill_between(T2_x.PAY_waxman.values.reshape(-1),T2_x.DEPTH.values.reshape(-1), color='g', alpha=0.8)
plt.axis([0.01,0.0101,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

#Simandoux
ct+=1
plt.subplot(1,no_plots,ct)
plt.plot (T2_x.PAY_simandoux,T2_x.DEPTH,'y',lw=0.5)
h_S = integrate(T2_x.PAY_simandoux.values, 0.5)

plt.title('$PAY Simandoux$',fontsize=8)
plt.fill_between(T2_x.PAY_simandoux.values.reshape(-1),T2_x.DEPTH.values.reshape(-1), color='y', alpha=0.8)
plt.axis([0.01,0.0101,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

ct+=1   #RGB Gray from Image
plt.subplot(1,no_plots,ct)
plt.plot(sub['GRAY'], sub['DEPTH'], 'mediumseagreen', linewidth=0.5);
plt.axis([50, 250, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
#plt.title('$Core Img$',fontsize=8)
plt.gca().invert_yaxis();
plt.gca().yaxis.set_visible(False)
plt.fill_between(sub['GRAY'], 0, sub['DEPTH'], facecolor='green', alpha=0.5)
plt.xlabel('Gray Scale RGB', fontsize=7)

ct+=1  # True UV from Image
plt.subplot(1,no_plots,ct, facecolor='#302f43')
corte= 170
PAY_Gray_scale = res['GRAY'].copy()
PAY_Gray_scale.GRAY[PAY_Gray_scale.GRAY<corte] = 0
PAY_Gray_scale.GRAY[PAY_Gray_scale.GRAY>=corte] = 1
h_TRUE_UV = integrate(PAY_Gray_scale.values, 0.5)
plt.plot (PAY_Gray_scale,res.DEPT,'#7d8d9c',lw=0.5)
plt.title('$OBJETIVO (suavizado-a-2.5ft)$',fontsize=10) 
plt.fill_between(PAY_Gray_scale.values.reshape(-1),res.DEPT.values.reshape(-1), color='#7d8d9c', alpha=0.8)
plt.axis([0.01,0.0101,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 


ct+=1
plt.subplot(1,no_plots,ct)
plt.imshow(ImgStackk[istr:iend,80:120], aspect='auto', origin='upper', extent=[0,1,dplot_n,dplot_o], cmap=shading);
plt.axis([0, 1, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel('Stacked UV Photos', fontsize=7)
plt.colorbar()
p_50 = np.percentile(sub['DEPTH'], 50)
plt.yticks([]); plt.xticks([])

ct+=1
plt.subplot(1,no_plots,ct)
plt.plot (res['RandomForest'],res.DEPT,'r',lw=1)
plt.plot (res.GRAY,res.DEPT,'k',lw=0.5)
plt.title('ML: GRIS',fontsize=12)
plt.axis([0,2,top,bottom])
plt.xticks(fontsize=8)
plt.xlabel('RandomForest',fontsize=7)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(0, 255) 
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

ct+=1
plt.subplot(1,no_plots,ct, facecolor='#302f43')
PAY_Gray_scale2 = res['RandomForest'].copy().rename(columns={'RandomForest':'GRAY'})
PAY_Gray_scale2.GRAY[PAY_Gray_scale2.GRAY<corte] = 0
PAY_Gray_scale2.GRAY[PAY_Gray_scale2.GRAY>=corte] = 1
h_ML = integrate(PAY_Gray_scale2.values, 0.5)
plt.plot (PAY_Gray_scale2, res.DEPT,'#7d8d9c',lw=0.5)
plt.title('$RESULTADO$',fontsize=8)
plt.fill_between(PAY_Gray_scale2.values.reshape(-1),res.DEPT.values.reshape(-1), color='#7d8d9c', alpha=0.8)
plt.axis([0.01,0.0101,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 



plt.suptitle('Pozo T2: Comparación Final')
plt.show()

# %%
plt.figure(figsize=(10,9))
plt.subplot(1,1,1)
plt.plot(res.GRAY, res['RandomForest'], 'ko')
plt.plot(res.GRAY, res.GRAY, 'r')
plt.xlim(0, 255) 
plt.ylim(0, 255) 
plt.xlabel('Valor en Escala de Gris Suavizado a res. de Registros',fontsize=17)
plt.ylabel('Predicción de Escala de Gris usando Random Forest',fontsize=17)
plt.show()

# %%  Erro Calculation

# T2_x.PAY_poupon,T2_x.DEPTH
# T2_x.PAY_waxman
# T2_x.PAY_simandoux




# %%
pay = pd.DataFrame(columns=['Poupon', 'Waxman_Smits', 'Simandoux', 'Machine_L', 'True_UV'], index=['ft','RMSE'])

pay.loc['ft', 'Poupon'] = float(h_P.round(2))
pay.loc['ft', 'Waxman_Smits'] = float(h_WS.round(2))
pay.loc['ft', 'Simandoux'] = float(h_S.round(2))
pay.loc['ft', 'Machine_L'] = float(h_ML.round(2))
pay.loc['ft', 'True_UV'] = float(h_TRUE_UV.round(2))

pay.loc['RMSE', 'Poupon'] = pay.iloc[0,0] - pay.iloc[0,4]
pay.loc['RMSE', 'Waxman_Smits'] = pay.iloc[0,1] - pay.iloc[0,4]
pay.loc['RMSE', 'Simandoux'] = pay.iloc[0,2] - pay.iloc[0,4]
pay.loc['RMSE', 'Machine_L'] = pay.iloc[0,3] - pay.iloc[0,4]
pay.loc['RMSE', 'True_UV'] = pay.iloc[0,4] - pay.iloc[0,4]

pay.head()

# %%
y_colors = ['g', 'b']*5 # <-- this concatenates the list to itself 5 times.
my_colors = [(0.5,0.4,0.5), (0.75, 0.75, 0.25)]*5 # <-- make two custom RGBs and repeat/alternate them over all the bar elements.

my_colors = [(6/255, 184/255, 112/255)]*5



# %%
payN = pay.T.copy()
payN.reset_index(inplace=True)
payN.rename(columns={'index':'Método'}, inplace=True)

plt.figure()
payN.plot.bar(x='Método', y='RMSE', rot=0, color=my_colors)
plt.ylabel('Diferencia en pies de roca productiva calculada')
plt.title('Error Comparado con calculo directo de Imágen UV')
# %%
