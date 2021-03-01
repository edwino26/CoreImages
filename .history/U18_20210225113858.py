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

# REUNI LOS 3 LAS EN UN SOLO DF 
frames = [df1, df2, df3]
result = pd.merge(df1,df2)
df5=pd.merge(result,df3)
# ubicar los datos para mostrar en tabla entre 700ft a 1100 ft
df= df5.loc[(df5.TDEP>=700) &  (df5.TDEP<=1100)]
print(df.head)

# %%
# NO TIENE CAMBIOS AUN 
U18_xl =pd.read_excel('./data U18.xlsx',sheet_name = 'DATA')
U18_xl= U18_xl[['DEPTH','RHOZ']]
print(U18_xl)
# %%
top=650
bottom=1200
temp=((0.0198*df5.TDEP)+ 26.921) 
v= 400000
b=0.88
tsup = 25 #F 
WS=18000
RWs= (v/tsup/WS)**b
tf=temp
Kt1=6.77
df5['RW']=(RWs*(tsup+Kt1))/(temp+Kt1)
df5['Vsh'] = (df5.GR_EDTC - 10) / (156 - 10)
df5['Vclay']=((0.65)*df5.Vsh) 

mud_density=1.13835   #en g/cc
rhoss=2.70  # g/cc
rhosh=2.75
df5['grain_density']=((df5.Vsh*rhosh)+(1-df5.Vsh)*rhoss)
df5['porosity']=(df1.grain_density-(U18_xl.RHOZ))/(df5.grain_density-mud_density)
# %%
# %%
CORE =pd.read_excel('./CORE/CORE.xlsx',sheet_name='XRD')
mask = CORE.Well.isin(['U18'])
U18_Core = CORE[mask]
prof=U18_Core['Depth']
clays=U18_Core['Clays']

xls1 = pd.read_excel ('./CORE/CORE.xlsx', sheet_name='Saturation')
mask = xls1.Well.isin(['U18'])
U18_sat = xls1[mask]
long=U18_sat  ['Depth']
poro=U18_sat  ['PHIT']
grain=U18_sat  ['RHOG']
sw_core=U18_sat  ['Sw']
klinkenberg =U18_sat ['K']

minimo=grain.min()
maximo=grain.max()
c=2.65
d=2.75
norm=(((grain-minimo)*(d-c)/(maximo-minimo))+c)

xls2 = pd.read_excel ('./CORE/CORE.xlsx', sheet_name='Gamma')
mask = xls2.Well.isin(['U18'])
U18_GR = xls2[mask]
h=U18_GR['Depth']
cg1=U18_GR['GR_Scaled']

plt.hist(clays,bins=50,facecolor='y',alpha=0.75,ec='black', label="Vclay")
plt.title('Histogram-Vclay')
plt.xlabel('%Vclay')
plt.ylabel('Frecuency')
plt.legend()
# %%
# %%
# EL RHOZ ESTA SIN CAMBIOS 
dt = 200
bt= 1450
plt.figure(figsize=(15,9))
plt.subplot(171)
plt.plot(df5.GR_EDTC,df5.TDEP,'g',lw=0.5)
plt.title('$GR$')
plt.axis([40, 130, dt,bt])
plt.xlabel('Gamma Ray ')
plt.gca().invert_yaxis()
plt.grid(True)

plt.subplot(172)
plt.plot(df5.AT90,df5.TDEP,lw=0.5)
plt.axis([10, 700, dt,bt])
plt.title('$AT90$')
plt.xlabel('Resistivity')
plt.gca().invert_yaxis()
plt.xscale('log')
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(173)
plt.plot(U18_xl.RHOZ,U18_xl.DEPTH,'red',lw=0.5)
plt.axis([2.25, 2.65, dt,bt])
plt.title('$RHOZ$')
plt.xlabel('Standard \n Resolution \n Formation \n Density') 
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(174)
plt.plot(df5.NPHI,df5.TDEP,'purple',lw=0.5)
plt.axis([0.6, 0.1, dt,bt])
plt.title('$NPHI$')
plt.xlabel('Thermal \n Neutron \n Porosity',fontsize=8)
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(175)
plt.plot(df5.DTCO,df5.TDEP,'r',lw=0.5)
plt.title('$DTCO$')
plt.xlabel('Delta-T \n Compressional ')
plt.axis([60,125, dt,bt])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(176)
plt.plot(temp,df5.TDEP,'c')
plt.axis([20, 65, dt,bt])
plt.gca().invert_yaxis()
plt.title('$TEMP$')
plt.xlabel('Temperature')
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(177)
plt.plot(df5.RW,df5.TDEP,'blue',lw=0.5)
plt.title('$RW$')
plt.axis([0.4, 0.85, dt,bt])
plt.xlabel('Water \n Resistivity')
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)


plt.suptitle('U18_WELL LOGS_')



plt.show()
# %%
## SW_Archie
## SW=((a*Rw)/(Rt*(Por)^m))^(1/n)
a=1
m=2
n=2
Rw=df1.RW
Rt=df2.AT90
Phi=df1.porosity
F = (a / (Phi**m))
df5['Sw_a']  = (F *Rw/Rt)**(1/n)
df5['Sw_a1']= df5['Sw_a'].apply(lambda x: 1 if x>1 else x)
df5['Sw_a1'] = df5['Sw_a1'].replace(np.nan, 1)
dfSh = df5[df5['Vsh']>0.5]
Rsh= 5.2523
# %%
# WATER SATURATION POUPON
# TERM1= 1/RT - VSH/RSH
term1=(1/df5.AT90)-(df5.Vsh/Rsh)
## TERM2 = F*RW
term2=(F*df5.RW)
## TERM3 = (1-vsh)
term3=(1-df5.Vsh)
## SW_POUPON = ((TERM1*TERM2)/TERM3))^(1/N)
df5['Sw_p']=((term1*term2)/term3)**(1/n)
df5['Sw_p1']= df5['Sw_p'].apply(lambda x: 1 if x >1 else x)
df5['Sw_p1'] = df5['Sw_p1'].replace(np.nan, 1)
# %%
# WAXMAN-SMITS CEC method (does not require VCL) but requires core measurements of CEC 
TempC = (temp-32)/1.8
df5['SwWS'] = df5['Sw_p1']
CEC_av = 5
# ===== Waxman Smits Iterations. Reference:  Well Logging for Earth Scientists, Page 663-667 
for i in range(len(Rt)):
    error = 1000
    count1 = 0
    phit = Phi.iloc[i]
    
    if math.isnan(phit):
        df1['SwWS'][i] = 1
    else:
        Qv = rhosh*(1-phit)*CEC_av/phit/100    # Old Method
        Bcond = 3.83*(1-0.83*np.exp(-0.5/Rw.iloc[i]))  #  Waxman and Thomas, 1974
        BQv = Qv*Bcond
        E = (phit**m)/a
        Ct = 1/Rt.iloc[i]
        Cw = 1/Rw.iloc[i]
        x0 = df5.iloc[i]['Sw_a1']
        Swguess = x0

        while count1 <= 100 and error > 0.0001:
            count1 = count1+1        
            g =  E*Cw*(Swguess**n) + E*BQv*(Swguess**(n-1)) - Ct
            error = g
            gp = n*E*Cw*(Swguess**(n-1)) + (n-1)*E*BQv*(Swguess**(n-2))
           # print(df_1['SwWS'][i-1])
            df5['SwWS'].iloc[i] = Swguess-g/gp
            Swguess = df5['SwWS'].iloc[i]      


# %%
# SIMANDOUX (1963) for shaly-sandy formations, used with saline fm waters Equation solved for n=2
# Input parameters:
    #Rw - water resistivity
    #Rt - true resistivity
    #Phi - porosity
    #Rsh - shale resistivity
    # a - tortuosity factor
    # m - cementation exponent
    # n - saturation exponent
    # Vsh - Volume of shale
## CRAIN'S EQUATION 
#c=(1-df_1.Vsh)*a*(RWs)/(Phi**m)
#d=c*df_1.Vsh/(2*Rsh)
#e=c/Rt
#SWS=((c**2+d)**0.5-d)**(2/n)
df5['Swsim']=((a*Rw)/(2*(Phi**m)))*(((df5.Vsh/Rsh)**2+((4*Phi**m)/(a*Rw*Rt)))**(1/2)-(df5.Vsh/Rsh))
df5['Swsim1'] = df5['Swsim'].replace(np.nan, 1)
df5.head(2000)

# %%
plt.figure(figsize=(15,9))
plt.subplot(191)
plt.plot (df5.GR_EDTC,df5.TDEP,'g',cg1,h,'c.',lw=0.5)
plt.title('$GR/ Core.GR $')
plt.axis([20, 130, top,bottom])
plt.xlabel('Gamma Ray ')
plt.gca().invert_yaxis()
plt.grid(True)

plt.subplot(192)
plt.title('Vsh')
plt.plot (df5.Vsh,df5.TDEP,'black',lw=0.5)
plt.axis([0,1, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(193)
plt.title('$Vclay/Vclay Core$')
plt.plot (df5.Vclay,df5.TDEP,'m',clays,prof,'ro',lw=0.5)
plt.axis([0,1, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(194)
plt.title('Porosity \n  Core Por.')
plt.plot (df5.porosity,U18_xl.DEPTH,'m',poro,long,'c*',lw=0.5)
plt.axis([0, 0.4, top,bottom])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(195)
plt.title('Grain density \n Core GD')
plt.plot (df5.grain_density,df5.TDEP,'y',norm,long,'g>',lw=0.5)
plt.axis([2.64, 2.76, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

#Basic Archie
plt.subplot(196)
plt.plot (df5.Sw_a1,df5.TDEP,'c',sw_core,long,'m.',lw=0.5)
plt.title('$SW_A$')
plt.axis([0,1.1,top,bottom])
plt.xlabel('Water \n Saturation_A')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

#Poupon Laminated Model
plt.subplot(197)
plt.plot (df5.Sw_p1,df5.TDEP,'r',sw_core,long,'m.',lw=0.5)
plt.title('$SW_P$')
plt.axis([0,1.5,top,bottom])
plt.xlabel('Water \n Saturation_P')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

#Waxman-Smits
plt.subplot(198)
plt.plot (df5.SwWS,df5.TDEP,'g',sw_core,long,'m.',lw=0.5)
plt.title('$SW_W$')
plt.axis([0,5,top,bottom])
plt.xlabel('Water \n Saturation_Waxman')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

#Simandoux
plt.subplot(199)
plt.plot (df5.Swsim1,df5.TDEP   ,'y',sw_core,long,'m.',lw=0.5)
plt.title('$SW_S$')
plt.axis([0,2,top,bottom])
plt.xlabel('Water \n Saturation_Sim')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

plt.show() 

# %%
with pd.ExcelWriter('U18.xlsx') as writer:  
    df1.to_excel(writer, sheet_name='U18_DF1')
    df2.to_excel(writer, sheet_name='U18_DF2')
    df3.to_excel(writer, sheet_name='U18_DF3')
    U18_xl.to_excel(writer, sheet_name='U18_XL')