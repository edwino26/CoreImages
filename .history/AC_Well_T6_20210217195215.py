# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import math
import lasio
import matplotlib.pyplot as plt  # GRAPHS
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import glob

# %%    
las_T6= lasio.read('./LAS/T6/T6_Logs.las')
las_T6_S= lasio.read('./LAS/T6/T6_Sonic.las')

# %%
df_1 = las_T6.df()
df_1 = df_1[["GR_EDTC", "RHOZ", "DEPTH","AT90","NPHI"]]
df_1['Vsh'] = (df_1.GR_EDTC - 40) / (160 - 40)
df_1['Vclay']=((0.65)*df_1.Vsh) 

mud_density=1.14434   #en g/cc
rhoss=2.65  # g/cc
rhosh=2.8
df_1['grain_density']=((df_1.Vsh*rhosh)+(1-df_1.Vsh)*rhoss)
df_1['porosity']=(df_1.grain_density-df_1.RHOZ)/(df_1.grain_density-mud_density)

df_2 = las_T6_S.df()
Depth= las_T6_S.index
df_2 = df_2[['DTCO']]


# %%
top=3370
bottom=3525
dt = 2370
bt=4140
temp=((0.0159*df_1.DEPTH)+ 39.505) 
v= 400000
b=0.88
tsup= 25 #F
WS=14000
RWs= (v/tsup/WS)**b
tf=temp
Kt1=6.77
df_1['RW2']=(RWs*(tsup+Kt1))/(temp+Kt1)
print(RWs)


# %%
plt.figure(figsize=(15,9))
plt.subplot(171)
plt.plot(df_1.GR_EDTC,df_1.DEPTH,'g',lw=0.5)
plt.title('$GR$')
plt.axis([40, 130, dt,bt])
plt.xlabel('Gamma Ray ')
plt.gca().invert_yaxis()
plt.grid(True)
plt.hlines(y=2090, xmin=0, xmax=130)
plt.hlines(y=2906, xmin=0, xmax=130)

plt.hlines(y=2906, xmin=0, xmax=130)
plt.hlines(y=3373, xmin=0, xmax=130)

plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)

plt.subplot(172)
plt.plot(df_1.AT90,df_1.DEPTH,lw=0.5)
plt.axis([2, 40, dt,bt])
plt.title('$AT90$')
plt.xlabel('Resistivity')
plt.gca().invert_yaxis()
plt.xscale('log')
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(173)
plt.plot(df_1.RHOZ,df_1.DEPTH,'red',lw=0.5)
plt.axis([1.9, 2.65, dt,bt])
plt.title('$RHOZ$')
plt.xlabel('Standard \n Resolution \n Formation \n Density') #\n ( G/C3)'  DENTRO DEL PARENTESIS
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(174)
plt.plot(df_1.NPHI,df_1.DEPTH,'purple',lw=0.5)
plt.axis([0.6, 0.1, dt,bt])
plt.title('$NPHI$')
plt.xlabel('Thermal \n Neutron \n Porosity')
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(175)
plt.plot(df_2.DTCO,Depth,'r',lw=0.5)
plt.title('$DTCO$')
plt.xlabel('Delta-T \n Compressional ')
plt.axis([60,125, dt,bt])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(176)
plt.plot(temp,df_1.DEPTH,'c')
plt.axis([75, 106, dt,bt])
plt.gca().invert_yaxis()
plt.title('$TEMP$')
plt.xlabel('Temperature')
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(177)
plt.plot(df_1.RW2,df_1.DEPTH,'blue',lw=0.5)
plt.title('$RW$')
plt.axis([0.32, 0.43, dt,bt])
plt.xlabel('Water \n Resistivity')
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)



plt.suptitle('Tinmiaq-6_WELL LOGS_'+ las_T6.well['STAT']['value'])



plt.show()


# %%
plt.figure(figsize=(15,9))
plt.subplot(171)
plt.plot(df_1.GR_EDTC,df_1.DEPTH,'g',lw=0.5)
plt.title('$GR$')
plt.axis([40, 130, top,bottom])
plt.xlabel('Gamma Ray ')
plt.gca().invert_yaxis()
plt.grid(True)
plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)


plt.subplot(172)
plt.plot(df_1.AT90,df_1.DEPTH,lw=0.5)
plt.axis([2, 40, top,bottom])
plt.title('$AT90$')
plt.xlabel('Resistivity')
plt.gca().invert_yaxis()
plt.xscale('log')
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)

plt.subplot(173)
plt.plot(df_1.RHOZ,df_1.DEPTH,'red',lw=0.5)
plt.axis([1.6, 2.65,top,bottom])
plt.title('$RHOZ$')
plt.xlabel('Standard \n Resolution \n Formation \n Density') #\n ( G/C3)'  DENTRO DEL PARENTESIS
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)

plt.subplot(174)
plt.plot(df_1.NPHI,df_1.DEPTH,'purple',lw=0.5)
plt.axis([0.6, 0.1,top,bottom])
plt.title('$NPHI$')
plt.xlabel('Thermal \n Neutron \n Porosity')
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)

plt.subplot(175)
plt.plot(df_2.DTCO,Depth,'r',lw=0.5)
plt.title('$DTCO$')
plt.xlabel('Delta-T \n Compressional ')
plt.axis([60,125, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)

plt.subplot(176)
plt.plot(temp,df_1.DEPTH,'c')
plt.axis([80, 105,top,bottom])
plt.gca().invert_yaxis()
plt.title('$TEMP$')
plt.xlabel('Temperature')
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)

plt.subplot(177)
plt.plot(df_1.RW2,df_1.DEPTH,'blue',lw=0.5)
plt.title('$RW$')
plt.axis([0.32, 0.43, top,bottom])
plt.xlabel('Water \n Resistivity')
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3373, xmin=0, xmax=130)
plt.hlines(y=3520, xmin=0, xmax=130)





plt.suptitle('Tinmiaq-6_WELL LOGS_'+ las_T6.well['STAT']['value'])



plt.show()


# %%
CORE =pd.read_excel('./CORE/CORE.xlsx',sheet_name='XRD')
mask = CORE.Well.isin(['T6'])
T6_Core = CORE[mask]
prof=T6_Core['Depth']
clays=T6_Core['Clays']

xls1 = pd.read_excel ('./CORE/CORE.xlsx', sheet_name='Saturation')
mask = xls1.Well.isin(['T6'])
T6_sat = xls1[mask]
long=T6_sat  ['Depth']
poro=T6_sat  ['PHIT']
grain=T6_sat  ['RHOG']
sw_core=T6_sat  ['Sw']
klinkenberg = T6_sat ['K']

minimo=grain.min()
maximo=grain.max()
c=2.65
d=2.75
norm=(((grain-minimo)*(d-c)/(maximo-minimo))+c)

xls2 = pd.read_excel ('./CORE/CORE.xlsx', sheet_name='Gamma')
mask = xls2.Well.isin(['T6'])
T6_GR = xls2[mask]
h=T6_GR['Depth']
cg1=T6_GR['GR_Scaled']

plt.hist(clays,bins=50,facecolor='y',alpha=0.75,ec='black', label="Vclay")
plt.title('Histogram-Vclay')
plt.xlabel('Vclay')
plt.ylabel('Frecuency')
plt.legend()


# %%
## SW_Archie
## SW=((a*Rw)/(Rt*(Por)^m))^(1/n)
## Rt= df_1.AT90
## Rw= df_1.RW2
a=1
m=2
n=2
Rw=df_1.RW2
Rt=df_1.AT90
Phi=df_1.porosity
F = (a / (Phi**m))
df_1['Sw_a']  = (F *Rw/Rt)**(1/n)
df_1['Sw_a1']= df_1['Sw_a'].apply(lambda x: 1 if x>1 else x)
df_1['Sw_a1'] = df_1['Sw_a1'].replace(np.nan, 1)


# %%
dfSh = df_1[df_1['Vsh']>0.5]
Rsh = np.percentile(dfSh['AT90'],20)
print(Rsh)


# %%
## TERM1= 1/RT - VSH/RSH
df_1['term1']=(1/df_1.AT90)-(df_1.Vsh/Rsh)
## TERM2 = F*RW
term2=(F*df_1.RW2)
## TERM3 = (1-vsh)
term3=(1-df_1.Vsh)
## SW_POUPON = ((TERM1*TERM2)/TERM3))^(1/N)
df_1['Sw_p']=((df_1.term1*term2)/term3)**(1/n)
df_1['Sw_p1']= df_1['Sw_p'].apply(lambda x: 1 if x >1 else x)
df_1['Sw_p1'] = df_1['Sw_p1'].replace(np.nan, 1)


# %%
# WAXMAN-SMITS CEC method (does not require VCL) but requires core measurements of CEC 
TempC = (temp-32)/1.8
df_1['SwWS'] = df_1['Sw_p1']
CEC_av = 5
# ===== Waxman Smits Iterations. Reference:  Well Logging for Earth Scientists, Page 663-667 
for i in range(len(Rt)):
    error = 1000
    count1 = 0
    phit = Phi.iloc[i]
    
    if math.isnan(phit):
        df_1['SwWS'][i] = 1
    else:
        Qv = rhosh*(1-phit)*CEC_av/phit/100    # Old Method
        Bcond = 3.83*(1-0.83*np.exp(-0.5/Rw.iloc[i]))  #  Waxman and Thomas, 1974
        BQv = Qv*Bcond
        E = (phit**m)/a
        Ct = 1/Rt.iloc[i]
        Cw = 1/Rw.iloc[i]
        x0 = df_1.iloc[i]['Sw_a1']
        Swguess = x0

        while count1 <= 100 and error > 0.0001:
            count1 = count1+1        
            g =  E*Cw*(Swguess**n) + E*BQv*(Swguess**(n-1)) - Ct
            error = g
            gp = n*E*Cw*(Swguess**(n-1)) + (n-1)*E*BQv*(Swguess**(n-2))
           # print(df_1['SwWS'][i-1])
            df_1['SwWS'].iloc[i] = Swguess-g/gp
            Swguess = df_1['SwWS'].iloc[i]      


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
df_1['Swsim']=((a*Rw)/(2*(Phi**m)))*(((df_1.Vsh/Rsh)**2+((4*Phi**m)/(a*Rw*Rt)))**(1/2)-(df_1.Vsh/Rsh))
df_1['Swsim1'] = df_1['Swsim'].replace(np.nan, 1)
df_1.head(100)


# %%
plt.figure(figsize=(15,9))
plt.subplot(191)
plt.plot (df_1.GR_EDTC,df_1.DEPTH,'g',cg1,h,'c.',lw=0.5)
plt.title('$GR/ Core.GR $')
plt.axis([40,130,top,bottom])
plt.xlabel('Gamma Ray ')
plt.gca().invert_yaxis()
plt.grid(True)


plt.subplot(192)
plt.title('Vsh')
plt.plot (df_1.Vsh,df_1.DEPTH,'black',lw=0.5)
plt.axis([0,1,top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(193)
plt.title('$Vclay/Vclay Core$')
plt.plot (df_1.Vclay,df_1.DEPTH,'m',clays,prof,'ro',lw=0.5)
plt.axis([0,1, top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(194)
plt.title('Porosity \n  Core Por.')
plt.plot (df_1.porosity,df_1.DEPTH,'m',poro,long,'c*',lw=0.5)
plt.axis([0, 0.4,top,bottom])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(195)
plt.title('Grain density \n Core GD')
plt.plot (df_1.grain_density,df_1.DEPTH,'y',norm,long,'g>',lw=0.5)
plt.axis([2.64, 2.76,top,bottom])
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

#Basic Archie
plt.subplot(196)
plt.plot (df_1.Sw_a1,df_1.DEPTH,'c',sw_core,long,'m.',lw=0.5)
plt.title('$SW_A$')
plt.axis([0,1.1,top,bottom])
plt.xlabel('Sw_Archie')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

#Poupon Laminated Model
plt.subplot(197)
plt.plot (df_1.Sw_p1,df_1.DEPTH,'r',sw_core,long,'m.',lw=0.5)
plt.title('$SW_P$')
plt.axis([0,1.5,top,bottom])
plt.xlabel('Sw_Poupon')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

#Waxman-Smits
plt.subplot(198)
plt.plot (df_1.SwWS,df_1.DEPTH,'g',sw_core,long,'m.',lw=0.5)
plt.title('$SW_W$')
plt.axis([0,5,top,bottom])
plt.xlabel('Sw_Waxman')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

#Simandoux
plt.subplot(199)
plt.plot (df_1.Swsim1,df_1.DEPTH,'y',sw_core,long,'m.',lw=0.5)
plt.title('$SW_S$')
plt.axis([0,2,top,bottom])
plt.xlabel('Sw_Simandoux')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0) 
# %%
corte=0.5
df_1['PAY_archie']=df_1.Sw_a1.apply(lambda x: 1 if x<corte else 0)
df_1['PAY_poupon']=df_1.Sw_p1.apply(lambda x: 1 if x<corte else 0)
df_1['PAY_waxman']=df_1.SwWS.apply(lambda x: 1 if x<corte else 0)
df_1['PAY_simandoux']=df_1.Swsim1.apply(lambda x: 1 if x<corte else 0)


plt.figure(figsize=(15,9))
plt.subplot(191)
#Basic Archie
plt.plot (df_1.Sw_a1,df_1.DEPTH,'c',lw=0.5)
plt.title('$SW_A$')
plt.axis([0,1,top,bottom])
plt.xlabel('Sw_Archie')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.grid(True)
plt.xlim(1, 0)

plt.subplot(192)
plt.plot (df_1.PAY_archie,df_1.DEPTH,'c',lw=0.5)
plt.title('$PAY_A$')
plt.fill_between(df_1.PAY_archie,df_1.DEPTH, color='c', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 


#Poupon Laminated Model
plt.subplot(193)
plt.plot (df_1.Sw_p1,df_1.DEPTH,'r',lw=0.5)
plt.title('$SW_P$')
plt.axis([0,1.5,top,bottom])
plt.xlabel('Sw_Poupon')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

plt.subplot(194)
plt.plot (df_1.PAY_poupon,df_1.DEPTH,'r',lw=0.5)
plt.title('$PAY_P$')
plt.fill_between(df_1.PAY_poupon,df_1.DEPTH, color='r', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 


#Waxman-Smits
plt.subplot(195)
plt.plot (df_1.SwWS,df_1.DEPTH,'g',lw=0.5)
plt.title('$SW_W$')
plt.axis([0,5,top,bottom])
plt.xlabel('Sw_Waxman')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)

plt.subplot(196)
plt.plot (df_1.PAY_waxman,df_1.DEPTH,'g',lw=0.5)
plt.title('$PAY_W$')
plt.fill_between(df_1.PAY_waxman,df_1.DEPTH, color='g', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 

#Simandoux
plt.subplot(197)
plt.plot (df_1.Swsim1,df_1.DEPTH,'y',lw=0.5)
plt.title('$SW_S$')
plt.axis([0,2,top,bottom])
plt.xlabel('Sw_Simandoux')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0) 

plt.subplot(198)
plt.plot (df_1.PAY_simandoux,df_1.DEPTH,'y',lw=0.5)
plt.title('$PAY_S$')
plt.fill_between(df_1.PAY_simandoux,df_1.DEPTH, color='y', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 

plt.show()
# %%
mask3 = xls1.Well.isin(['T2'])
T2_sat = xls1[mask3]
poro1=T2_sat['PHIT']
sw_core1=T2_sat['Sw']
klinkenberg1 = T2_sat['K']

mask4 = xls1.Well.isin(['U18'])
U18_sat = xls1[mask4]
poro2=U18_sat['PHIT']
sw_core2=U18_sat['Sw']
klinkenberg2 = U18_sat['K']

n = 10
ticks = range(n)
colors = plt.cm.get_cmap('winter_r',n)(ticks)
lcmap = plt.matplotlib.colors.ListedColormap(colors)
plt.figure(figsize=(19,5))
plt.subplot(121)
plt.scatter(poro,klinkenberg, c=sw_core, s=30, alpha=15, edgecolor='none', cmap=lcmap,marker="o")
plt.scatter(poro1,klinkenberg1, c=sw_core1, s=40, alpha=90, edgecolor='none', cmap=lcmap,marker='^')
plt.scatter(poro2,klinkenberg2, c=sw_core2, s=60, alpha=30, edgecolor='none', cmap=lcmap,marker="x")
plt.colorbar(ticks=ticks,label='Sw_core ')
plt.clim(0,1)


circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=10, label='T2')
plus = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=10, label='U18')
triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=10, label='T6')

plt.legend(handles=[circle, triangle,plus],loc='lower right')

plt.title('Core permeability & porosity')
plt.ylabel('Core Permeability (mD)')
plt.yscale('log')
plt.xlabel('Core Porosity')
plt.grid(True)

plt.show()


# %%

n = 10
ticks = range(n)
colors = plt.cm.get_cmap('summer',n)(ticks)
lcmap = plt.matplotlib.colors.ListedColormap(colors)
plt.figure(figsize=(19,5))
plt.subplot(121)
plt.scatter(poro,sw_core, c=np.log(klinkenberg), s=40, alpha=20, edgecolor='none', cmap=lcmap,marker="o")
plt.scatter(poro1,sw_core1,c=np.log(klinkenberg1) , s=80, alpha=10, edgecolor='none', cmap=lcmap,marker='^')
plt.scatter(poro2,sw_core2, c=np.log(klinkenberg2), s=60, alpha=80, edgecolor='none', cmap=lcmap,marker="x")
plt.colorbar(ticks=ticks,label='Core Perm. (mD)')
plt.clim(0,1)


circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=10, label='T2')
triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=10, label='T6')
plus = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=10, label='U18')

plt.legend(handles=[circle, triangle, plus],loc='lower right')

plt.title('Core permeability & porosity')
plt.ylabel('Sw  v/v')
plt.xlabel('Core Porosity, v/v')
plt.grid(True)
plt.show()



# %%
pay=sw_core.apply(lambda x: 1 if x<0.5 else 0)
pay1=sw_core1.apply(lambda x: 1 if x<0.5 else 0)
pay2=sw_core2.apply(lambda x: 1 if x<0.5 else 0)
n =2
ticks = range(n)
colors = plt.cm.get_cmap('winter',n)(ticks)
lcmap = plt.matplotlib.colors.ListedColormap(colors)


plt.figure(figsize=(19,5))
plt.subplot(121)
plt.scatter(poro,klinkenberg, c=pay, s=40, alpha=20, edgecolor='none', cmap=lcmap,marker="o")
plt.scatter(poro1,klinkenberg1, c=pay1,s=80, alpha=10, edgecolor='none', cmap=lcmap,marker='^')
plt.scatter(poro2,klinkenberg2, c=pay2, s=60, alpha=80, edgecolor='none', cmap=lcmap,marker="x")
plt.colorbar(ticks=ticks, label='Pay')
plt.clim(0,1)


circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=10, label='T2')
plus = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=10, label='U18')
triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=10, label='T6')

plt.legend(handles=[circle,triangle,plus],loc='lower right')

plt.title('Core permeability & porosity')
plt.ylabel('Core Permeability (mD)')
plt.yscale('log')
plt.xlabel('Core Porosity v/v')
plt.grid(True)
plt.show()
# %%

# %%
