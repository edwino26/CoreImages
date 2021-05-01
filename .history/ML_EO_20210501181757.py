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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier


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

# %% =========================  Machine Learning: PROCESSING ===============================
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

# Scaling
#Find scaling parameters based on training data only
scaler = StandardScaler()
scaler.fit(X)
print(scaler.mean_)
X = scaler.transform(X)
X_test = scaler.transform(X_test)
# Note: Scaling the target is nor necessary, nor advised
#  ====================================================================
#  ================  Machine Learning: MODELING =======================
#  ====================================================================
print(df.shape)
# %% -------------- Linear Model: LASSO, L1 regularization --------------
rgr = linear_model.Lasso(alpha=0.3)
rgr.fit(X, y)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
print(rgr.coef_, rgr.intercept_)
mse = mean_squared_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.05, 'MSE = '+str(round(mse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].text(1.2, 0.1, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()
#-------------------------------- End Lasso --------------------------
# %% -------------- Linear Model: ElasticNet, L1+L2 regularization --------------
rgr= linear_model.ElasticNet(alpha=0.5, l1_ratio=0.1, random_state = 5, selection='random')
rgr.fit(X, y)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
print(rgr.coef_, rgr.intercept_)
mse = mean_squared_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.05, 'MSE = '+str(round(mse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].text(1.2, 0.1, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()
#-------------------------------- End ElasticNet --------------------------
#--------------------------------------------------------------------------
# %% ------------------- Linear Model: Ridge Regression -------------------
rgr= linear_model.Ridge(alpha=0.5, solver='auto')
rgr.fit(X, y)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
print(rgr.coef_, rgr.intercept_)
mse = mean_squared_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.05, 'MSE = '+str(round(mse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].text(1.2, 0.1, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()
#--------------------------------- End Ridge ------------------------------
#--------------------------------------------------------------------------
# %% ------------------- Support Vector Machines: SVR ---------------------
rgr= SVR(C= 150, epsilon=0.2)
rgr.fit(X, y)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.05, 'MSE = '+str(round(mse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].text(1.2, 0.1, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()
#--------------------------------- End SVR --------------------------------

#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o--o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o-o-o-o-o-o-o-  ENSEMBLE METHODS -o-o-o-o-o-o-o-o-o-o-o-o-o-o
# Ref: https://scikit-learn.org/stable/modules/ensemble.html

# %% ------------------- Averaging: Random Forest Regressor ---------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
rgr = RandomForestRegressor(n_estimators=100, criterion='mse')


rgr.fit(X, np.ravel(y))
print("Relative importance of GR, RHOB, logRt,  DTCO, NPHI", rgr.feature_importances_)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.05, 'MSE = '+str(round(mse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].text(1.2, 0.1, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()
#----------------------- End Random Forest -----------------------------

# %% ------------------- Boosting: AdaBoost ---------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

rgr =  AdaBoostRegressor(n_estimators=500, random_state=0, learning_rate=0.0001, loss='square')


rgr.fit(X, np.ravel(y))

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.05, 'MSE = '+str(round(mse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].text(1.2, 0.1, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()
#----------------------- End AdaBoost -----------------------------




# %%
model = MLPRegressor(random_state=1, max_iter =500, hidden_layer_sizes=(100, 20, 100, 20, 100, 20, 100, 20), validation_fraction=0)._fit(X,y)


model.predict(X_test)
print(model.score(X_test, y_test))

print(model.get_params(True))

# %%
