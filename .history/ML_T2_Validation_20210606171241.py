#T2 VALIDATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

#Load Stack
UVStack = pd.read_excel('./ML_Results/T2_validation/ImgStack.xls')
ImgStackk = UVStack.copy().to_numpy()
sub = pd.read_excel('./ML_Results/T2_validation/sub.xls')

istr = 0
iend = 42344
dplot_o = 3671
dplot_n = 3750
shading = 'bone'


# %%
plt.figure(figsize=(10,9))

plt.subplot(1,6,1)
plt.imshow(ImgStackk[istr:iend,80:120], aspect='auto', origin='upper', extent=[0,1,dplot_n,dplot_o], cmap=shading);
plt.axis([0, 1, dplot_o, dplot_n]);
plt.xticks(fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel('Processed \n Image', fontsize=7)
plt.colorbar()
p_50 = np.percentile(sub['DEPTH'], 50)
plt.yticks([]); plt.xticks([])

plt.suptitle('Tinmiaq-2 WELL')
plt.show()