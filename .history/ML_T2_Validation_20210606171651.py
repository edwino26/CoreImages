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

#Load Model and Validate holdout Well    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)
score = model.score(X_val, y_val)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = model.predict(X_val)
Ypredict = pd.DataFrame(Ypredict, columns=['Gray_pred'])
df_val['Gray_pred'] = Ypredict 
df_val.head()