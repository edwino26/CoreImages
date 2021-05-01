# # lectura de los xlsx del procesamiento de imagenes 
# y la informacion obtenida de los registros por pozo
# %%
import numpy as np
import pandas as pd
import os
import os.path
import matplotlib.pyplot as plt  # GRAPHS
from mlxtend.plotting import scatterplotmatrix
import glob
import seaborn as sns
import missingno as msno
import math

#Clustering packages
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# ===============================================

#%% ====================== Main Dataframe Computation =====================

def round_depth(number):
    return round(number * 2, 0) / 2  #Use 2 to round to closest 0.5

# Data load
T2 = pd.read_excel('./Excel_Files/T2.xls',sheet_name='T2_data')
T6 = pd.read_excel('./Excel_Files/T6.xls',sheet_name='T6_data')
U18 = pd.read_excel('./Excel_Files/U18.xls',sheet_name='U18_data')
T2_img = pd.read_excel('./Excel_Files/Processed_Images_T2.xls',sheet_name='T2').rename(columns={"DEPTH": "DEPT"})
T6_img = pd.read_excel('./Excel_Files/Processed_Images_T6.xls',sheet_name='T6').rename(columns={"DEPTH": "DEPT"})
U18_img = pd.read_excel('./Excel_Files/Processed_Images_U18.xls',sheet_name='U18').rename(columns={"DEPTH": "DEPT"})


# Conditioning before joining well dataset pairs
T2['DEPT'] = T2['DEPTH'].astype(float).apply(lambda x: round_depth(x))
T2_img['DEPT'] = T2_img['DEPT'].astype(float).apply(lambda x: round_depth(x))
T6['DEPT'] = T6['DEPTH'].astype(float).apply(lambda x: round_depth(x))
T6_img['DEPT'] = T6_img['DEPT'].astype(float).apply(lambda x: round_depth(x))
U18['DEPT'] = U18['TDEP'].astype(float).apply(lambda x: round_depth(x))
U18_img['DEPT'] = U18_img['DEPT'].astype(float).apply(lambda x: round_depth(x))

# U18 had too many images compared to other 2 wells. Take a smaller set to have a more balanced dataset
U18 = U18[U18['DEPT']>771]
U18 = U18[U18['DEPT']<925]

# Join only depths present at both datasets within a pair
T2 = T2_img.merge(T2, on='DEPT', how='inner').dropna()
T6 = T6_img.merge(T6, on='DEPT', how='inner').dropna()
U18 = U18_img.merge(U18, on='DEPT', how='inner').dropna()

# Specific curves to be used in ML algorithm
logset = ['DEPT', 'GR_EDTC', 'RHOZ', 'AT90', 'DTCO', 'NPHI', 'WELL', 'GRAY']

#df = T2[logset].append(T6[logset]).append(U18[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "RT", "RHOZ": "RHOB", "WELL": "Well"}).set_index('Well')
df = T2[logset].append(T6[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "logRT", "RHOZ": "RHOB", "WELL": "Well"}).set_index('Well')
df['logRT'] = df['logRT'].apply(lambda x: math.log10(x))
df.reset_index(inplace=True)


# %%
# ===================== Descriptive Statistics and Plots =====================
print(df.groupby(['Well']).median())
print(df.groupby(['Well']).count())
print(df.groupby(['Well']).min())
print(df.groupby(['Well']).max())

# Show GR distribution among wells
fig,ax = plt.subplots()
hatches = ('\\', '//', '..')         # fill pattern
alpha_v = 0.9
for (i, d),hatch in zip(df.groupby('Well'), hatches):
    d['GR'].hist(alpha=alpha_v, ax=ax, label=i, hatch=hatch)
    alpha_v -= 0.3
ax.legend()

#Show how the gray scale varies among each photo
fig,ax = plt.subplots()
hatches = ('\\', '//', '..')         # fill pattern
alpha_v = 0.9
for (i, d),hatch in zip(df.groupby('Well'), hatches):
    d['GRAY'].hist(alpha=alpha_v, ax=ax, label=i, hatch=hatch)
    alpha_v -= 0.3
ax.legend()

# Expected correlations between variables
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(df['GRAY'])
axs[1, 0].plot(df['GR'], df['GRAY'], linestyle='None', markersize=4, marker='o')
axs[0, 1].plot(df['RHOB'], df['GRAY'], linestyle='None', markersize=4, marker='o')
axs[1, 1].hist2d(df['logRT'], df['GRAY'])
plt.show()

# Matrix Plot
variables= ['GR', 'RHOB', 'logRT', 'DTCO', 'NPHI', 'GRAY']
fig, axes = scatterplotmatrix(df[df['Well']=='T2'].drop(['Well', 'DEPT'], axis=1).values, figsize=(10, 8), alpha=0.5)
fig, axes = scatterplotmatrix(df[df['Well']=='T6'].drop(['Well', 'DEPT'], axis=1).values, fig_axes=(fig, axes), alpha=0.5, names=variables) 
#fig, axes = scatterplotmatrix(df[df['Well']=='U18'].drop(['Well', 'DEPT'], axis=1).values, fig_axes=(fig, axes), alpha=0.5, names=variables)
plt.tight_layout()
plt.show()

#With SB

sns.set_theme(style="ticks")
sns.pairplot(df, kind="kde")

# %% =========================  Machine Learning ===============================
print(df.shape)
df['Pay'] = df['GRAY'].apply(lambda x: 1 if x> 170 else 0)
# %%

data = df.drop(['Well', 'DEPT'], axis=1).copy()
data.head(100)
# %%
train, test = train_test_split(data, test_size=0.2)
scaler = StandardScaler()
test.shape
train.shape

option = 'Gray' #or 'Pay'
if option == 'Gray':
    target_col = 5
else:
    target_col = 6

X = train.iloc[:, [0,1,2,3,4]]
y = train.iloc[:, [target_col]]   #5 is GRAY, #6 is Pay

X_test = test.iloc[:, [0,1,2,3,4]]
y_test = test.iloc[:, [target_col]]  

# %% Preprocessing

#FInd scaling parameters based on training data only
scaler = StandardScaler()
scaler.fit(X)
print(scaler.mean_)

#Fit scaling to both train and test data
X = scaler.transform(X)
X_test = scaler.transform(X_test)


#Scaling the target is not necessary as mentioned by several online references

# %% -------------- Linear Model: LASSO, L1 regularization --------------
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_test, y_test)
print(clf.coef_, clf.intercept_)
y_pred = clf.predict(X_test)
fig = plt.figure()

plt.subplot(121)
plt.plot(y, y_pred, 'o')
plt.plot(y, y,  'blue')

ax = fig.add_subplot(122)
plt.plot(y_test, y_pred, 'o')
plt.plot(y_test, y_test,  'blue')
plt.suptitle('Training vs Test Data')
ax.text(0.95, 0.01, str(round(clf.score(X_test, y_test),2)),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
plt.show()
#-------------------------------- End Lasso --------------------------


# %%
model = MLPRegressor(random_state=1, max_iter =500, hidden_layer_sizes=(100, 20, 100, 20, 100, 20, 100, 20), validation_fraction=0)._fit(X,y)


model.predict(X_test)
print(model.score(X_test, y_test))

print(model.get_params(True))
y_pred = model.predict(X_test)
mean_squared_error(y_true, y_pred)

fig = plt.figure()
plt.subplot(121)
plt.plot(y, model.predict(X), 'o')
plt.plot(y, y,  'blue')
ax = fig.add_subplot(122)
plt.plot(y_test, y_pred, 'o')
plt.plot(y_test, y_test,  'blue')
plt.suptitle('Training vs Test Data')
ax.text(0.95, 0.01, str(round(model.score(X_test, y_test),2)),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
plt.show()
# %%
