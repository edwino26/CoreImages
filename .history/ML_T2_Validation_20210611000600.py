#T2 TEST DATA
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import interpolate
from scipy.integrate import simps
from numpy import trapz


# %%

#Load Stack
UVStack = pd.read_excel('./ML_Results/T2_test/ImgStack.xls')
ImgStackk = UVStack.copy().to_numpy()


# %%
sub = pd.read_excel('./ML_Results/T2_test/sub.xls')
res = pd.read_excel('./ML_Results/T2_test/Results.xls')
res = res[res.Well == 'T2']
res.sort_values(by=['DEPT'])
res.drop(['Unnamed: 0'], axis=1, inplace=True)
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

res_rs.head()

# %%

res = res[res.DEPT.mod(0.5) == 0]
res.drop_duplicates(subset='DEPT', keep="last", inplace=True)




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

dep = np.arange(min(res.DEPT), max(res.DEPT),0.5) 
T2_rs = pd.DataFrame(columns=[T2_x.columns])
T2_rs.iloc[:,0] = dep

for i in range(len(T2_x.columns)):
    f = interpolate.interp1d(T2_x.DEPTH, T2_x.iloc[:,i])
    T2_rs.iloc[:,i] =f(dep)
#T2_rs.dropna(inplace=True)

T2_x = T2_rs.copy()


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
plt.figure(figsize=(10,9))
plt.subplot(1,no_plots,ct)
plt.plot (T2_x.GR_EDTC,T2_x.DEPTH,'g',cg1,(h+3),'c.',lw=0.5)
plt.title('$GR/ Core.GR $',fontsize=8)
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
plt.title('$PAY_P$',fontsize=8)
plt.fill_between(T2_x.PAY_poupon,T2_x.DEPTH, color='r', alpha=0.8)
plt.axis([0,0.001,top,bottom])
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
plt.title('$PAY_W$',fontsize=8)
plt.fill_between(T2_x.PAY_waxman,T2_x.DEPTH, color='g', alpha=0.8)
plt.axis([0,0.001,top,bottom])
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
plt.title('$PAY_S$',fontsize=8)
plt.fill_between(T2_x.PAY_simandoux,T2_x.DEPTH, color='y', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

ct+=1
plt.subplot(1,no_plots,ct)
plt.plot(sub['GRAY'], sub['DEPTH'], 'mediumseagreen', linewidth=0.5);
plt.axis([50, 250, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
plt.title('$Core Img$',fontsize=8)
plt.gca().invert_yaxis();
plt.gca().yaxis.set_visible(False)
plt.fill_between(sub['GRAY'], 0, sub['DEPTH'], facecolor='green', alpha=0.5)
plt.xlabel('Gray Scale RGB', fontsize=7)

ct+=1
corte= 140
PAY_Gray_scale = res['GRAY'].apply(lambda x: 1 if x<corte else 0)
plt.subplot(1,no_plots,ct)
plt.plot (PAY_Gray_scale,res.DEPT,'c',lw=0.5)
plt.title('$PAY-GS$',fontsize=8)
plt.fill_between(PAY_Gray_scale,res.DEPT, color='c', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 
plt.xlabel('Resolution to Log Scale',fontsize=7)

ct+=1
plt.subplot(1,no_plots,ct)
plt.imshow(ImgStackk[istr:iend,80:120], aspect='auto', origin='upper', extent=[0,1,dplot_n,dplot_o], cmap=shading);
plt.axis([0, 1, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel('Processed \n Image', fontsize=7)
plt.colorbar()
p_50 = np.percentile(sub['DEPTH'], 50)
plt.yticks([]); plt.xticks([])

ct+=1
plt.subplot(1,no_plots,ct)
plt.plot (res['RandomForest'],res.DEPT,'r',lw=1)
plt.plot (res.GRAY,res.DEPT,'k',lw=0.5)
plt.title('Machine Learning',fontsize=8)
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
PAY_Gray_scale = res['RandomForest'].apply(lambda x: 1 if x<corte else 0)
plt.subplot(1,no_plots,ct)
plt.plot (res.DEPT,'c',lw=0.5)
plt.title('$Validations$',fontsize=8)
plt.fill_between(PAY_Gray_scale,res.DEPT, color='c', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 



plt.suptitle('Tinmiaq-2 Method Comparison')
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



# %%
rmse = pd.DataFrame(columns=['Poupon', 'Waxman-Smits', 'Simandooux', 'Machine Learning'])

rmse['Poupon'] = mean_squared_error(y_test, y_pred_test, squared=False)





# %%
