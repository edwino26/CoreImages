#T2 VALIDATION
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# %%

#Load Stack
UVStack = pd.read_excel('./ML_Results/T2_validation/ImgStack.xls')
ImgStackk = UVStack.copy().to_numpy()
sub = pd.read_excel('./ML_Results/T2_validation/sub.xls')
res = pd.read_excel('./ML_Results/T2_validation/Results.xls')
TT = pd.read_excel('./ML_Results/T2_validation/Train_Test_Results.xls')

istr = 0
iend = 42344
dplot_o = 3671
dplot_n = 3750
shading = 'bone'


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
# Plot Results

T2_x = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
T2_x = T2_x[['DEPTH','GR_EDTC','RHOZ','AT90','NPHI','Vsh','Vclay','grain_density','porosity',
                   'RW2','Sw_a','Sw_a1','Sw_p','Sw_p1','SwWS','Swsim','Swsim1','PAY_archie',
                    'PAY_poupon','PAY_waxman','PAY_simandoux']]


plt.figure(figsize=(10,9))
plt.subplot(1,14,1)
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

plt.subplot(1,14,2)
plt.plot(T2_x.AT90,T2_x.DEPTH,lw=0.5)
plt.axis([2, 40, top,bottom])
plt.xticks(fontsize=8)
plt.title('$AT90$',fontsize=8)
plt.xlabel('Resistivity',fontsize=7)
plt.gca().invert_yaxis()
plt.xscale('log')
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

plt.subplot(1,14,3)
plt.plot(T2_x.RHOZ,T2_x.DEPTH,'red',lw=0.5)
plt.axis([2, 2.65,top,bottom])
plt.xticks(fontsize=8)
plt.title('$RHOZ$',fontsize=8)
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

plt.subplot(1,14,4)
plt.plot(T2_x.NPHI,T2_x.DEPTH,'purple',lw=0.5)
plt.axis([0.6, 0.1,top,bottom])
plt.xticks(fontsize=8)
plt.title('$NPHI$',fontsize=8)
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)



#Poupon Laminated Model
plt.subplot(1,14,5)
plt.plot (T2_x.Sw_p1,T2_x.DEPTH,'r',lw=0.5)
plt.title('$SW_P$',fontsize=8)
plt.axis([0,1.5,top,bottom])
plt.xticks(fontsize=8)
plt.xlabel('Sw_Poupon',fontsize=7)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

plt.subplot(1,14,6)
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
plt.subplot(1,14,7)
plt.plot (T2_x.SwWS,T2_x.DEPTH,'g',lw=0.5)
plt.title('$SW_W$',fontsize=8)
plt.axis([0,5,top,bottom])
plt.xticks(fontsize=8)
plt.xlabel('Sw_Waxman',fontsize=7)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0)
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

plt.subplot(1,14,8)
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
plt.subplot(1,14,9)
plt.plot (T2_x.Swsim1,T2_x.DEPTH,'y',lw=0.5)
plt.title('$SW_S$',fontsize=8)
plt.axis([0,2,top,bottom])
plt.xticks(fontsize=8)
plt.xlabel('Sw_Simandoux',fontsize=7)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)
plt.xlim(1, 0) 
plt.hlines(y=3665.65, xmin=0, xmax=130)
plt.hlines(y=3889.5, xmin=0, xmax=130)

plt.subplot(1,14,10)
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

plt.subplot(1,14,11)
plt.plot(sub['GRAY'], sub['DEPTH'], 'mediumseagreen', linewidth=0.5);
plt.axis([50, 250, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
plt.title('$Core Img$',fontsize=8)
plt.gca().invert_yaxis();
plt.gca().yaxis.set_visible(False)
plt.fill_between(sub['GRAY'], 0, sub['DEPTH'], facecolor='green', alpha=0.5)
plt.xlabel('Gray Scale RGB', fontsize=7)

corte= 170
PAY_Gray_scale = sub['GRAY'].apply(lambda x: 1 if x<corte else 0)
plt.subplot(1,14,12)
plt.plot (PAY_Gray_scale,sub['DEPTH'],'c',lw=0.5)
plt.title('$PAY-GS$',fontsize=8)
plt.fill_between(PAY_Gray_scale,sub['DEPTH'], color='c', alpha=0.8)
plt.axis([0,0.001,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 

plt.subplot(1,14,13)
plt.imshow(ImgStack[istr:iend,80:120], aspect='auto', origin='upper', extent=[0,1,dplot_n,dplot_o], cmap=shading);
plt.axis([0, 1, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel('Processed \n Image', fontsize=7)
plt.colorbar()
p_50 = np.percentile(sub['DEPTH'], 50)
plt.yticks([]); plt.xticks([])

plt.suptitle('Tinmiaq-2 WELL')
plt.show()