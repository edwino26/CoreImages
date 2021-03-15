
# %%
import glob
import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Core Depth conversions necessary:
# 710-728 => 710.00 - 724.95
# 728-746 => 741.90 - 756.95
# 746-764 => 771.00 - 786.05
# 764-782 => 803.85 - 818.9
# 782-800 => 864.75 - 879.75
# 800-818 => 925.8  - 940.75
# 818-836 => 986.20 - 1001.10

# 804-822 => 803.85 - 818.9
# 822-840 => 864.75 - 879.75
# 840-858 => 925.8  - 940.75

# %%
Well = 'U18'
cores_per_image = 6
uvFiles = glob.glob('./Photos/U18/TIF/*.tif')
print(uvFiles)
filedos = []

for l in range(len(uvFiles)):
    
    filedos.append(int(uvFiles[l][17:20]))
    print(filedos)
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


    fname = uvFiles[k][13:27]
    # Picture path
    img = cv2.imread(uvFiles[k])
    do = int(fname[4:7])
    dn = int(fname[8:11])  
    
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
            midgap = gap * 2 ##Adjust size of gap. 4 used for T2/T6
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
        crop_name = str(int(fname[4:7]) + (core_length * i)) + ".tif"

    if k == 0:
        ImgStack = vc_gray
    else:
        ImgStack = np.concatenate((ImgStack, vc_gray), axis =0)

    
    path = os.path.join(os.path.relpath('./Cropped/Cropped_U18', start=os.curdir), crop_name)
    cv2.imwrite(path, crop_img)
    concat_name = fname[4:7] + "-" + fname[8:11] + ".tif"
    path = os.path.join(os.path.relpath('./Cropped/Cropped_U18', start=os.curdir), concat_name)
    cv2.imwrite(path, vc)
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
sub.to_excel("Processed_Images_U18.xlsx",sheet_name=Well) 
 
# %%
dplot_o = 710
dplot_n = 836
shading = 'bone'

istr = int(ImgStack.shape[0]*(dplot_o - doo)/(dnn-doo))
iend = int(ImgStack.shape[0]*(dplot_n - doo)/(dnn-doo))

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


# ------------------------------------------------ END OF IMAGE PROCESSING -----------------------------------

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
klinkenberg = U18_sat ['K']

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

# %%
top= dplot_o 
bottom= dplot_n 
U18_x = pd.read_excel('./U18.xlsx',sheet_name='U18_data')
U18_x = U18_x[['TDEP','GR_EDTC','RHOZ','AT90','NPHI','Vsh','Vclay','grain_density','porosity',
                   'RW','Sw_a','Sw_a1','Sw_p','Sw_p1','SwWS','Swsim','Swsim1','PAY_archie',
                    'PAY_poupon','PAY_waxman','PAY_simandoux']]




plt.figure(figsize=(9,9))
plt.subplot(1,14,1)
plt.plot (U18_x.GR_EDTC,U18_x.TDEP,'g',cg1,(h+3),'c.',lw=0.5)
plt.title('$GR/ Core.GR $',fontsize=8)
plt.axis([20,130,top,bottom])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('Gamma Ray ',fontsize=6)
plt.gca().invert_yaxis()
plt.grid(True)

plt.subplot(1,14,2)
plt.plot(U18_x.AT90,U18_x.TDEP,lw=0.5)
plt.axis([10, 800, top,bottom])
plt.xticks(fontsize=8)
plt.title('$AT90$',fontsize=8)
plt.xlabel('Resistivity',fontsize=7)
plt.gca().invert_yaxis()
plt.xscale('log')
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(1,14,3)
plt.plot(U18_x.RHOZ,U18_x.TDEP,'red',lw=0.5)
plt.axis([2.2, 2.75,top,bottom])
plt.xticks(fontsize=8)
plt.title('$RHOZ$',fontsize=8)
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

plt.subplot(1,14,4)
plt.plot(U18_x.NPHI,U18_x.TDEP,'purple',lw=0.5)
plt.axis([0.6, 0.1,top,bottom])
plt.xticks(fontsize=8)
plt.title('$NPHI$',fontsize=8)
plt.gca().invert_yaxis()
plt.gca().yaxis.set_visible(False)
plt.grid(True)

#Poupon Laminated Model
plt.subplot(1,14,5)
plt.plot (U18_x.Sw_p1,U18_x.TDEP,'r',lw=0.5)
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
plt.plot (U18_x.PAY_poupon,U18_x.TDEP,'r',lw=0.5)
plt.title('$PAY_P$',fontsize=8)
plt.fill_between(U18_x.PAY_poupon,U18_x.TDEP, color='r', alpha=0.8)
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
plt.plot (U18_x.SwWS,U18_x.TDEP,'g',lw=0.5)
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
plt.plot (U18_x.PAY_waxman,U18_x.TDEP,'g',lw=0.5)
plt.title('$PAY_W$',fontsize=8)
plt.fill_between(U18_x.PAY_waxman,U18_x.TDEP, color='g', alpha=0.8)
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
plt.plot (U18_x.Swsim1,U18_x.TDEP,'y',lw=0.5)
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
plt.plot (U18_x.PAY_simandoux,U18_x.TDEP,'y',lw=0.5)
plt.title('$PAY_S$',fontsize=8)
plt.fill_between(U18_x.PAY_simandoux,U18_x.TDEP, color='y', alpha=0.8)
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

corte= 130
PAY_Gray_scale = sub['GRAY'].apply(lambda x: 1 if x<corte else 0)
plt.subplot(1,14,12)
plt.plot (PAY_Gray_scale,sub['DEPTH'],'c',lw=0.5)
plt.title('$PAY-GS$',fontsize=8)
plt.fill_between(PAY_Gray_scale,sub['DEPTH'], color='c', alpha=0.9)
plt.axis([0,0.001,top,bottom])
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()   
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(True) 

plt.subplot(1,14,13)
im = plt.imshow(ImgStack[istr:iend,80:120], aspect='auto', origin='upper', extent=[0,1,dplot_n,dplot_o], cmap=shading);
plt.axis([0, 1, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel('Processed \n Image', fontsize=7)
plt.colorbar(im)
p_50 = np.percentile(sub['DEPTH'], 50)
plt.yticks([]); plt.xticks([])

plt.suptitle('Umiat-18 WELL',fontsize=10)
plt.show()

