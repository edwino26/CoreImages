
# %%
import glob
import cv2
import lasio
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import interpolate



# %%
Well = 'T2'
cores_per_image = 6
uvFiles = glob.glob('./Photos/T2/*.jpg')
filedos = []

for l in range(len(uvFiles)):
    filedos.append(int(uvFiles[l][12:16]))
indices = [index for index, value in sorted(enumerate(filedos), reverse=False, key=lambda x: x[1]) if value > 1]
uvFiles = [uvFiles[i] for i in indices]


a = []
b = []
DEPTH = []
GRAY = []
PHOTO = []
dos = []
dns = []
doo = 0
dnn = 1
lg =0
ImgStack = []


# %%
def oneventlbuttondown(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 10, (0, 0, 255), thickness=-1)
        #        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
core_length = 3


for k in range(len(uvFiles)): #Loop through various files containing images
    vc = []  #Reset stacked photo when switching from one photo to the other one


    fname = uvFiles[k][12:28]
    # Picture path
    img = cv2.imread(uvFiles[k])
    do = int(fname[0:4])
    dn = int(fname[5:9])    

    dos.append(do)
    dns.append(dn)

    for i in range(cores_per_image):
        if k == 0 and i == 0:

            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("output", 400, 300)
            cv2.setMouseCallback("image", oneventlbuttondown)
            cv2.imshow("image", img)
            print(
                'Click 1) left upper corner 2) right lower corner in leftmost core and 3) leftupper corner in second core')
            cv2.waitKey(0)
            y = b[0];
            x = a[0];
            dy = b[1] - b[0];
            dx = a[1] - a[0]
            gap = a[2] - a[1]

        if i == 0:
            x = a[0]; y = b[0];
        if i == 3:
            midgap = gap * 4
        else:
            midgap = 0
        if i > 0: 
            x = x + (dx + gap) + midgap

        crop_img = img[y:y + dy, x:x + dx]
        
        if i == 0:# and k == 0:
            vc = crop_img
        else:
            vc = cv2.vconcat([vc, crop_img])
            
        vc_gray = cv2.cvtColor(vc, cv2.COLOR_BGR2GRAY)   
        crop_name = str(int(fname[0:4]) + (core_length * i)) + ".jpg"

    if k == 0:
        ImgStack = vc_gray
    else:
        ImgStack = np.concatenate((ImgStack, vc_gray), axis =0)

    
    path = os.path.join(os.path.relpath('./Cropped/Cropped_T2', start=os.curdir), crop_name)
    #cv2.imwrite(path, crop_img)
    concat_name = fname[0:4] + "-" + fname[5:9] + ".jpg"
    path = os.path.join(os.path.relpath('./Cropped/Cropped_T2', start=os.curdir), concat_name)
    #cv2.imwrite(path, vc)
    p = vc.shape

    if k == len(uvFiles)-1:
        doo = min(dos)
        dnn = max(dns)

    img_log = np.average(vc_gray[:, 20:100], axis=1)
    depths = np.arange(do, dn, (dn - do) / len(img_log))
    photo_number = np.full_like(depths, 1)*(k+1)

    DEPTH.extend(depths.tolist())
    GRAY.extend(img_log.tolist())
    PHOTO.extend(photo_number.tolist())


# %%

d = {'DEPTH': DEPTH, 'GRAY': GRAY}
sub = pd.DataFrame(np.array(DEPTH).T, columns = ['DEPTH'])
sub['GRAY'] = np.array(GRAY).T
sub['PHOTO'] = np.array(PHOTO).T


dep= np.arange(doo,dnn,0.25)
sub_S = pd.DataFrame(dep.T, columns = ['DEPTH'])
sub_S['DEPTH'] = dep
f = interpolate.interp1d(sub['DEPTH'], sub['GRAY'])
sub_S['GRAY'] = f(dep)
p = interpolate.interp1d(sub['DEPTH'], sub['PHOTO'])
sub_S['PHOTO'] = p(dep)

#sub_S.to_excel("./Excel_Files/Processed_Images_T2.xls",sheet_name=Well) 
#sub.to_excel("./ML_Results/T2_validation/sub.xls",sheet_name=Well) 

# %%
dplot_o = 3671
dplot_n = 3750
shading = 'bone'

istr = int(ImgStack.shape[0]*(dplot_o - doo)/(dnn-doo))
iend = int(ImgStack.shape[0]*(dplot_n - doo)/(dnn-doo))

ImgStackDF = pd.DataFrame(ImgStack)
#Save Image Stach to Dataframe
#ImgStackDF.to_excel("./ML_Results/T2_validation/ImgStack.xls",sheet_name=Well) 

plt.figure()
plt.subplot2grid((1, 10), (0, 0), colspan=3)
plt.plot(sub['GRAY'], sub['DEPTH'], 'mediumseagreen', linewidth=0.5);
plt.axis([50, 250, dplot_o, dplot_n]);
plt.gca().invert_yaxis();
plt.fill_between(sub['GRAY'], 0, sub['DEPTH'], facecolor='green', alpha=0.5)
plt.xlabel('Gray Scale RGB')

plt.subplot2grid((1, 10), (0, 3), colspan=7)
plt.imshow(ImgStack[istr:iend,80:120], aspect='auto', origin='upper', extent=[0,1,dplot_n,dplot_o], cmap=shading);
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
UVStack = pd.read_excel('./ML_Results/T2_validation/ImgStack.xls')
UVStack.to_numpy()


# %%
top= dplot_o 
bottom= dplot_n 
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
     

# %%
