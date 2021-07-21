# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import math
import lasio
from scipy import interpolate
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

U18_xl =pd.read_excel('./Excel_Files/U18_test.xls',sheet_name = 'U18_data')
df4=U18_xl[['DEPTH','RHOZ']]

#MERGE> to combine the LAS files into 1 df 
result = pd.merge(df1,df2, on= 'TDEP',how='left')
result.set_index('TDEP', inplace=True)
df5=pd.merge(result,df3,on= 'TDEP',how='left')
df5.set_index('TDEP', inplace=True)
# %%
# array con nueva tabla para TDEP (prof) con paso de 0.5
dep= np.arange(200,1350,0.5)
f = interpolate.interp1d(df4['DEPTH'], df4['RHOZ'])
RHOZ_new = f(dep)

plt.plot(df4['DEPTH'], df4['RHOZ'], 'o', dep, RHOZ_new, '-')
plt.show()

df6= pd.DataFrame(RHOZ_new,dep, columns=['RHOZ'])
df=pd.DataFrame(df5.join(df6,how='inner',on='TDEP'))


# %%
TDEP= df.index
top=650
bottom=1200
temp=((0.0198*TDEP)+ 26.921) 
v= 400000
b=0.88
tsup = 25 #F 
WS=18000
RWs= (v/tsup/WS)**b
tf=temp
Kt1=6.77
df['RW']=(RWs*(tsup+Kt1))/(temp+Kt1)
df['Vsh'] = (df.GR_EDTC - 10) / (156 - 10)
df['Vclay']=((0.65)*df.Vsh) 
mud_density=1.13835   #en g/cc
rhoss=2.70  # g/cc
rhosh=2.75
df['grain_density']=((df.Vsh*rhosh)+(1-df.Vsh)*rhoss)
df['porosity']=(df.grain_density-(df.RHOZ))/(df.grain_density-mud_density)
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
dt = 200
bt= 1350
plt.figure(figsize=(15,9))
plt.subplot(171)
plt.plot(df.GR_EDTC,TDEP,'g',lw=0.5)
plt.title('$GR$')
plt.axis([20, 130, dt,bt])
plt.xlabel('Gamma Ray ')
plt.gca().invert_yaxis()
plt.grid(True)

plt.subplot(172)
plt.plot(df.AT90,TDEP,lw=0.5)
plt.axis([10, 800, dt,bt])
plt.title('$AT90$')
plt.xlabel('Resistivity')
plt.gca().invert_yaxis()
plt.xscale('log')
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(173)
plt.plot(df.RHOZ,TDEP,'red',lw=0.5)
plt.axis([2.25, 2.65, dt,bt])
plt.title('$RHOZ$')
plt.xlabel('Standard \n Resolution \n Formation \n Density') 
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(174)
plt.plot(df.NPHI,TDEP,'purple',lw=0.5)
plt.axis([0.6, 0.1, dt,bt])
plt.title('$NPHI$')
plt.xlabel('Thermal \n Neutron \n Porosity',fontsize=8)
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(175)
plt.plot(df.DTCO,TDEP,'r',lw=0.5)
plt.title('$DTCO$')
plt.xlabel('Delta-T \n Compressional ')
plt.axis([60,125, dt,bt])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(176)
plt.plot(temp,TDEP,'c')
plt.axis([20, 65, dt,bt])
plt.gca().invert_yaxis()
plt.title('$TEMP$')
plt.xlabel('Temperature')
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(177)
plt.plot(df.RW,TDEP,'blue',lw=0.5)
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
Rw=df.RW
Rt=df.AT90
Phi=df.porosity
F = (a / (Phi**m))
df['Sw_a']  = (F *Rw/Rt)**(1/n)
df['Sw_a1']= df['Sw_a'].apply(lambda x: 1 if x>1 else x)
df['Sw_a1'] = df['Sw_a1'].replace(np.nan, 1)
dfSh = df[df['Vsh']>0.5]
Rsh = 50
#Sw_Poupon
# TERM1= 1/RT - VSH/RSH
term1=(1/df.AT90)-(df.Vsh/Rsh)
## TERM2 = F*RW
term2=(F*df.RW)
## TERM3 = (1-vsh)
term3=(1-df.Vsh)
## SW_POUPON = ((TERM1*TERM2)/TERM3))^(1/N)
df['Sw_p']=((term1*term2)/term3)**(1/n)
df['Sw_p1']= df['Sw_p'].apply(lambda x: 1 if x >1 else x)
df['Sw_p1'] = df['Sw_p1'].replace(np.nan, 1)


# %%
# WAXMAN-SMITS CEC method (does not require VCL) but requires core measurements of CEC 
TempC = (temp-32)/1.8
df['SwWS'] = df['Sw_p1']
CEC_av = 5
# ===== Waxman Smits Iterations. Reference:  Well Logging for Earth Scientists, Page 663-667 
for i in range(len(Rt)):
    error = 1000
    count1 = 0
    phit = Phi.iloc[i]
    
    if math.isnan(phit):
        df['SwWS'][i] = 1
    else:
        Qv = rhosh*(1-phit)*CEC_av/phit/100    # Old Method
        Bcond = 3.83*(1-0.83*np.exp(-0.5/Rw.iloc[i]))  #  Waxman and Thomas, 1974
        BQv = Qv*Bcond
        E = (phit**m)/a
        Ct = 1/Rt.iloc[i]
        Cw = 1/Rw.iloc[i]
        x0 = df.iloc[i]['Sw_a1']
        Swguess = x0

        while count1 <= 100 and error > 0.0001:
            count1 = count1+1        
            g =  E*Cw*(Swguess**n) + E*BQv*(Swguess**(n-1)) - Ct
            error = g
            gp = n*E*Cw*(Swguess**(n-1)) + (n-1)*E*BQv*(Swguess**(n-2))
           # print(df_1['SwWS'][i-1])
            df['SwWS'].iloc[i] = Swguess-g/gp
            Swguess = df['SwWS'].iloc[i]      
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
df['Swsim']=((a*Rw)/(2*(Phi**m)))*(((df.Vsh/Rsh)**2+((4*Phi**m)/(a*Rw*Rt)))**(1/2)-(df.Vsh/Rsh))
df['Swsim1'] = df['Swsim'].replace(np.nan, 1)
df.head(2000)

# %%
plt.figure(figsize=(15,9))
plt.subplot(191)
plt.plot (df.GR_EDTC,TDEP,'g',cg1,h,'c.',lw=0.5)
plt.title('$GR/ Core.GR $')
plt.axis([20, 130, top,bottom])
plt.xlabel('Gamma Ray ')
plt.gca().invert_yaxis()
plt.grid(True)

plt.subplot(192)
plt.title('Vsh')
plt.plot (df.Vsh,TDEP,'black',lw=0.5)
plt.axis([0,1, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(193)
plt.title('$Vclay/Vclay Core$')
plt.plot (df.Vclay,TDEP,'m',clays,prof,'ro',lw=0.5)
plt.axis([0,1, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(194)
plt.title('Porosity \n  Core Por.')
plt.plot (df.porosity,TDEP,'m',poro,long,'c*',lw=0.5)
plt.axis([0, 0.3, top,bottom])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(195)
plt.title('Grain density \n Core GD')
plt.plot (df.grain_density,TDEP,'y',norm,long,'g>',lw=0.5)
plt.axis([2.64, 2.76, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

#Basic Archie
plt.subplot(196)
plt.plot (df.Sw_a1,TDEP,'c',sw_core,long,'m.',lw=0.5)
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
plt.plot (df.Sw_p1,TDEP,'r',sw_core,long,'m.',lw=0.5)
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
plt.plot (df.SwWS,TDEP,'g',sw_core,long,'m.',lw=0.5)
plt.title('$SW_W$')
plt.axis([0,1,top,bottom])
plt.xlabel('Water \n Saturation_Waxman')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

#Simandoux
plt.subplot(199)
plt.plot (df.Swsim1,TDEP,'y',sw_core,long,'m.',lw=0.5)
plt.title('$SW_S$')
plt.axis([0,1,top,bottom])
plt.xlabel('Water \n Saturation_Sim')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

plt.show() 
# %%






















corte=0.5
df['PAY_archie']=df.Sw_a1.apply(lambda x: 1 if x<corte else 0)
df['PAY_poupon']=df.Sw_p1.apply(lambda x: 1 if x<corte else 0)
df['PAY_waxman']=df.SwWS.apply(lambda x: 1 if x<corte else 0)
df['PAY_simandoux']=df.Swsim1.apply(lambda x: 1 if x<corte else 0)




plt.figure(figsize=(15,9))
plt.subplot(191)
#Basic Archie
plt.plot (df.Sw_a1,TDEP,'c',lw=0.5)
plt.title('$SW_A$')
plt.axis([0,1,top,bottom])
plt.xlabel('Sw_Archie')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlim(1, 0)

plt.subplot(192)
plt.plot (df.PAY_archie,TDEP,'c',lw=0.5)
plt.title('$PAY_A$')
plt.fill_between(df.PAY_archie,TDEP, color='c', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 


#Poupon Laminated Model
plt.subplot(193)
plt.plot (df.Sw_p1,TDEP,'r',lw=0.5)
plt.title('$SW_P$')
plt.axis([0,1.5,top,bottom])
plt.xlabel('Sw_Poupon')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.xlim(1, 0)

plt.subplot(194)
plt.plot (df.PAY_poupon,TDEP,'r',lw=0.5)
plt.title('$PAY_P$')
plt.fill_between(df.PAY_poupon,TDEP, color='r', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 


#Waxman-Smits
plt.subplot(195)
plt.plot (df.SwWS,TDEP,'g',lw=0.5)
plt.title('$SW_W$')
plt.axis([0,5,top,bottom])
plt.xlabel('Sw_Waxman')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.xlim(1, 0)

plt.subplot(196)
plt.plot (df.PAY_waxman,TDEP,'g',lw=0.5)
plt.title('$PAY_W$')
plt.fill_between(df.PAY_waxman,TDEP, color='g', alpha=0.8)
plt.axis([0, 5,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 

#Simandoux
plt.subplot(197)
plt.plot (df.Swsim1,TDEP,'y',lw=0.5)
plt.title('$SW_S$')
plt.axis([0,2,top,bottom])
plt.xlabel('Sw_Simandoux')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.xlim(1, 0) 

plt.subplot(198)
plt.plot (df.PAY_simandoux,TDEP,'y',lw=0.5)
plt.title('$PAY_S$')
plt.fill_between(df.PAY_simandoux,TDEP, color='y', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 

plt.show()

# %%
df.insert(21, "WELL", 'U18')
df.head()
with pd.ExcelWriter('U18_TEST.xlsx') as writer:  
    df.to_excel(writer, sheet_name='U18_data')
    

# %%
