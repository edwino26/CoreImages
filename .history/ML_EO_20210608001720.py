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
import itertools
import smogn
import pickle
from scipy import interpolate

#Clustering packages
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS


# ===============================================
from ML_EO_GridSearch import *

GRIDSEARCH = 'on' #GridSearch Option

    

#%% ====================== Main Dataframe Computation =====================

def round_depth(number):
    return round(number * 4, 0) / 4  #Use 2 to round to closest 0.5, 4 to 0.25

# Data load
T2 = pd.read_excel('./Excel_Files/T2_rs.xls',sheet_name='T2_data')
T6 = pd.read_excel('./Excel_Files/T6_rs.xls',sheet_name='T6_data')
U18 = pd.read_excel('./Excel_Files/U18_rs.xls',sheet_name='U18_data')
T2_img = pd.read_excel('./Excel_Files/Processed_Images_T2.xls',sheet_name='T2').rename(columns={"DEPTH": "DEPT"})
T6_img = pd.read_excel('./Excel_Files/Processed_Images_T6.xls',sheet_name='T6').rename(columns={"DEPTH": "DEPT"})
U18_img = pd.read_excel('./Excel_Files/Processed_Images_U18.xls',sheet_name='U18').rename(columns={"DEPTH": "DEPT"})


# %%

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

#Scale Images as bright and dim points realte to presence of oil but images were 
# acquired by diffeent labs at different times, using different equipment, etc
scaler = MinMaxScaler()
T2['GRAY'] = 255*scaler.fit_transform(T2['GRAY'].values.reshape(-1,1))
T6['GRAY'] = 255*scaler.fit_transform(T6['GRAY'].values.reshape(-1,1))
U18['GRAY'] = 255*scaler.fit_transform(U18['GRAY'].values.reshape(-1,1))


# %%
# ---------- Main Data Frame Structuring --------
#Train on T6 adn U18
# .append(U18[logset])

df = T2[logset].append(T6[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "logRT", "RHOZ": "RHOB", "WELL": "Well"}).set_index('Well')

#df = T2[logset].append(T6[logset]).rename(columns={"GR_EDTC": "GR", "AT90": "logRT", "RHOZ": "RHOB", "WELL": "Well"}).set_index('Well')


df['logRT'] = df['logRT'].apply(lambda x: math.log10(x))
df.reset_index(inplace=True)

df.to_excel('./ML_Results/df.xlsx','Original')
df['Pay'] = df['GRAY'].apply(lambda x: 1 if x> 170 else 0)

data = df.copy()
data= data[["GR", "RHOB",	"logRT",	"DTCO",	"NPHI",	"GRAY",	"Pay","DEPT", "Well"]]
data.head(100)

#train, test = train_test_split(data, test_size=0.3, random_state=42)
#train = data[(data['Well']!='T6') & (data['GRAY']>10)].copy()

# train = data[(data['Well']!='T2')] 
# test = data[data['Well']=='T2']    

#test = data[(data['DEPT']>3670) &(data['DEPT']<3720)]
#train = data[~data.index.isin(test.index)]

train, test = train_test_split(data, test_size=0.1, random_state=42)
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)



# %%  Pre-processing:   Data Balancing
# Original data is imbalanced as there's more dark than bright in all UV photos
# SMOGN drops certain samples to re-balance the dataset
# Reference: http://proceedings.mlr.press/v74/branco17a/branco17a.pdf

## specify phi relevance values
balance = 0
if balance == 1:
    train_smogn = smogn.smoter( 
        data = train,  ## pandas dataframe
        y = 'GRAY',  ## string ('target header name')
        pert = 0.1,
        k = 3,
        rel_thres = 0.3,          ## relevance threshold considered rare (pos real)
        rel_method = "auto",      ## relevance method ("auto" or "manual")
        rel_xtrm_type = "both",   ## distribution focus ("high", "low", "both")
        rel_coef = 0.5,           ## coefficient for box plot (pos real)
        rel_ctrl_pts_rg = None
    )

    ## Check changes in target variable distribution
    smogn.box_plot_stats(train['GRAY'])['stats']
    smogn.box_plot_stats(train_smogn['GRAY'])['stats']
    ## plot y distribution 
    sns.kdeplot(train['GRAY'], label = "Original")
    sns.kdeplot(train_smogn['GRAY'], label = "Modified")
    del train
    train = train_smogn.copy()
    train.to_excel('./ML_Results/train_smogn.xlsx','Smogn')
#-------------------------------

# %% =========================  Machine Learning: PROCESSING ===============================
augmentation = 0
## Attempt Augmentation
if augmentation ==1:
    train.sort_values('DEPT', inplace=True)
    aa = train.copy()
    s1 = pd.Series([0])
    aa.GRAY = aa.GRAY[1:].append(s1).values
    ab = train.copy()
    ab.GRAY = s1.append(ab.GRAY[0:-1]).values
    s2 = pd.Series([0, 0])
    ac = train.copy()
    ac.GRAY = aa.GRAY[2:].append(s2).values
    ad = train.copy()
    ad.GRAY = s2.append(ab.GRAY[0:-2]).values
    train = train.append(aa).append(ab).append(ac).append(ad)


scaler = StandardScaler()
print("Test Dataset Shape  :::::: ", test.shape)
print("Train Dataset Shape :::::: ", train.shape)

train.to_excel('./ML_Results/train.xlsx','Train')
test.to_excel('./ML_Results/test.xlsx','Test')

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



# %%
plot_descriptive = 1

if plot_descriptive == 1:
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
    fig, axss = plt.subplots(2, 2, figsize=(5, 5))
    axss[0, 0].hist(df['GRAY'])
    axss[1, 0].plot(df['GR'], df['GRAY'], linestyle='None', markersize=4, marker='o')
    axss[0, 1].plot(df['RHOB'], df['GRAY'], linestyle='None', markersize=4, marker='o')
    axss[1, 1].hist2d(df['logRT'], df['GRAY'])
    plt.show()

    # Matrix Plot
    # variables= ['GR', 'RHOB', 'logRT', 'DTCO', 'NPHI', 'GRAY']
    # fig, axes = scatterplotmatrix(df[df['Well']=='T2'].drop(['Well', 'DEPT'], axis=1).values, figsize=(8, 6), alpha=0.5)
    # fig, axes = scatterplotmatrix(df[df['Well']=='T6'].drop(['Well', 'DEPT'], axis=1).values, fig_axes=(fig, axes), alpha=0.5, names=variables) 
    # #fig, axes = scatterplotmatrix(df[df['Well']=='U18'].drop(['Well', 'DEPT'], axis=1).values, fig_axes=(fig, axes), alpha=0.5, names=variables)
    # plt.tight_layout()
    # plt.show()

    #With SB
    #sns.set_theme(style="ticks")
    #sns.pairplot(train, kind="kde")



# Note: Scaling the target is nor necessary, nor advised
#  ====================================================================
#  ================  Machine Learning: MODELING =======================
#  ====================================================================
print(df.shape)
Methods = ['RandomForest', 'Lasso', 'ElasticNet', 'Ridge', 'SVR',  'GradientBoosting', 'MLP']
error = pd.DataFrame(index=['RMSE', 'MSE'], columns=Methods)


# %% ------------------- Averaging: Random Forest Regressor ---------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

GRIDSEARCH == 'off'
if GRIDSEARCH == 'on':
    rgr = RandomForestRegressorGS(X, np.ravel(y))
    rgr.fit(X, np.ravel(y))
else:
    rgr = RandomForestRegressor(n_estimators=10000, criterion='mse')
    rgr.fit(X, np.ravel(y))


print("Relative importance of GR, RHOB, logRt,  DTCO, NPHI", rgr.feature_importances_)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y, y_pred_train, squared=False)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[0].text(-0.3, 0.08, 'RMSE = '+str(round(rmse_train,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='black', fontsize=10)
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.08, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()

#ct +=1
ct =0
method = Methods[ct]
y_pred_train = pd.DataFrame(y_pred_train, columns=['y_pred_train'], index=y.index)
y_pred_test = pd.DataFrame(y_pred_test, columns=['y_pred_test'], index=y_test.index)
train_all = train.copy()
train_all[method] = y_pred_train
test_all = test.copy()
test_all[method] = y_pred_test
error.loc['MSE', method] = mse.round()
error.loc['RMSE', method] = rmse.round()
fig.savefig('./ML_Results/'+str(ct)+"_"+method+'.jpg')

#Save Model
pkl_filename = method+".pkl"
with open('./ML_Results/models/'+pkl_filename, 'wb') as file:
    pickle.dump(rgr, file)
#Load Model and Validate holdout Well    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)

print(error)
#----------------------- End Random Forest -----------------------------

# %% -------------- Linear Model: LASSO, L1 regularization --------------














































if GRIDSEARCH == 'on':
    rgr = lassoGS(X, y)
    rgr.fit(X, y)
else:
    rgr = linear_model.Lasso(alpha=0.3)
    rgr.fit(X, y)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
print(rgr.coef_, rgr.intercept_)
mse = mean_squared_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y, y_pred_train, squared=False)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[0].text(-0.3, 0.08, 'RMSE = '+str(round(rmse_train,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='black', fontsize=10)
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.08, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()

ct += 1
method = Methods[ct]
y_pred_train = pd.DataFrame(y_pred_train, columns=['y_pred_train'], index=y.index)
y_pred_test = pd.DataFrame(y_pred_test, columns=['y_pred_test'], index=y_test.index)
train_all[method] = y_pred_train
test_all[method] = y_pred_test
error.loc['MSE', method] = mse.round()
error.loc['RMSE', method] = rmse.round()
fig.savefig('./ML_Results/'+str(ct)+"_"+method+'.jpg')

#Save Model
pkl_filename = method+".pkl"
with open('./ML_Results/models/'+pkl_filename, 'wb') as file:
    pickle.dump(rgr, file)
#Load Model and Validate holdout Well    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)

print(error)
#-------------------------------- End Lasso --------------------------
# %%
# %% -------------- Linear Model: ElasticNet, L1+L2 regularization --------------
if GRIDSEARCH == 'on':
    rgr = ElasticNetGS(X, y)
    rgr.fit(X, y)
else:
    rgr = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.1, random_state = 5, selection='random')
    rgr.fit(X, y)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
print(rgr.coef_, rgr.intercept_)
mse = mean_squared_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y, y_pred_train, squared=False)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[0].text(-0.3, 0.08, 'RMSE = '+str(round(rmse_train,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='black', fontsize=10)
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.08, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()

ct +=1
method = Methods[ct]
y_pred_train = pd.DataFrame(y_pred_train, columns=['y_pred_train'], index=y.index)
y_pred_test = pd.DataFrame(y_pred_test, columns=['y_pred_test'], index=y_test.index)
train_all[method] = y_pred_train
test_all[method] = y_pred_test
error.loc['MSE', method] = mse.round()
error.loc['RMSE', method] = rmse.round()
fig.savefig('./ML_Results/'+str(ct)+"_"+method+'.jpg')

#Save Model
pkl_filename = method+".pkl"
with open('./ML_Results/models/'+pkl_filename, 'wb') as file:
    pickle.dump(rgr, file)
#Load Model and Validate holdout Well    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)

print(error)
#-------------------------------- End ElasticNet --------------------------
#--------------------------------------------------------------------------
# %% ------------------- Linear Model: Ridge Regression -------------------
if GRIDSEARCH == 'on':
    rgr = RidgeGS(X, y)
    rgr.fit(X, y)
else:
    rgr = linear_model.Ridge(alpha=0.5, solver='auto')
    rgr.fit(X, y)

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
print(rgr.coef_, rgr.intercept_)
mse = mean_squared_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y, y_pred_train, squared=False)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[0].text(-0.3, 0.08, 'RMSE = '+str(round(rmse_train,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='black', fontsize=10)
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.08, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()

ct +=1
method = Methods[ct]
y_pred_train = pd.DataFrame(y_pred_train, columns=['y_pred_train'], index=y.index)
y_pred_test = pd.DataFrame(y_pred_test, columns=['y_pred_test'], index=y_test.index)
train_all[method] = y_pred_train
test_all[method] = y_pred_test
error.loc['MSE', method] = mse.round()
error.loc['RMSE', method] = rmse.round()
fig.savefig('./ML_Results/'+str(ct)+"_"+method+'.jpg')

#Save Model
pkl_filename = method+".pkl"
with open('./ML_Results/models/'+pkl_filename, 'wb') as file:
    pickle.dump(rgr, file)
#Load Model and Validate holdout Well    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)

print(error)
#--------------------------------- End Ridge ------------------------------
#--------------------------------------------------------------------------
# %% ------------------- Support Vector Machines: SVR ---------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

if GRIDSEARCH == 'on':
    rgr = SVRGS(X, np.ravel(y))
    rgr.fit(X, np.ravel(y))
else:
    rgr = SVR(C= 150, epsilon=0.2)
    rgr.fit(X, np.ravel(y))

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y, y_pred_train, squared=False)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[0].text(-0.3, 0.08, 'RMSE = '+str(round(rmse_train,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='black', fontsize=10)
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.08, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()

ct +=1
method = Methods[ct]
y_pred_train = pd.DataFrame(y_pred_train, columns=['y_pred_train'], index=y.index)
y_pred_test = pd.DataFrame(y_pred_test, columns=['y_pred_test'], index=y_test.index)
train_all[method] = y_pred_train
test_all[method] = y_pred_test
error.loc['MSE', method] = mse.round()
error.loc['RMSE', method] = rmse.round()
fig.savefig('./ML_Results/'+str(ct)+"_"+method+'.jpg')

#Save Model
pkl_filename = method+".pkl"
with open('./ML_Results/models/'+pkl_filename, 'wb') as file:
    pickle.dump(rgr, file)
#Load Model and Validate holdout Well    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)

print(error)
#--------------------------------- End SVR --------------------------------


#-o-o-o-o-o-o-o-o-o-o-o-o-o-  ENSEMBLE METHODS -o-o-o-o-o-o-o-o-o-o-o-o-o-o
# Ref: https://scikit-learn.org/stable/modules/ensemble.html


# %% ------------------- Boosting: Gradient Tree Boosting ---------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
GRIDSEARCH == 'off'
if GRIDSEARCH == 'on':
    rgr = GradientBoostingRegressorGS(X, np.ravel(y))
    rgr.fit(X, np.ravel(y))
else:
    rgr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, alpha=0.336)
    rgr.fit(X, np.ravel(y))

y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y, y_pred_train, squared=False)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[0].text(-0.3, 0.08, 'RMSE = '+str(round(rmse_train,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='black', fontsize=10)
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.08, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()

ct +=1
method = Methods[ct]
y_pred_train = pd.DataFrame(y_pred_train, columns=['y_pred_train'], index=y.index)
y_pred_test = pd.DataFrame(y_pred_test, columns=['y_pred_test'], index=y_test.index)
train_all[method] = y_pred_train
test_all[method] = y_pred_test
error.loc['MSE', method] = mse.round()
error.loc['RMSE', method] = rmse.round()
fig.savefig('./ML_Results/'+str(ct)+"_"+method+'.jpg')

#Save Model
pkl_filename = method+".pkl"
with open('./ML_Results/models/'+pkl_filename, 'wb') as file:
    pickle.dump(rgr, file)
#Load Model and Validate holdout Well    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)

print(error)
#----------------------- End Gradient Boosting -----------------------------

# %% ----------------------------------------------- Neural Network ----------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
# Default: solver: Adam, activation: relu, learning rate: constant given by learning_rate_init, batch_size='auto'
# hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,),which means one hidden layer with 100 neurons
# hidden_layer_sizes (30, 30, 30) means 3 hidden layers with 30 neurons each. 
# Use e.g. [x for x in itertools.product((10,50,100,30),repeat=4)] to generate all possible 4-hidden layer combinations
GRIDSEARCH == 'off'
if GRIDSEARCH == 'on':
    rgr = MLPRegressorGS(X, np.ravel(y))
    rgr.fit(X, np.ravel(y))
else:
    rgr = MLPRegressor(hidden_layer_sizes=(10, 50, 100), alpha=0.66333, batch_size='auto', learning_rate_init=0.01)
    rgr.fit(X, np.ravel(y))



y_pred_train = rgr.predict(X)
y_pred_test = rgr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y, y_pred_train, squared=False)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8,4))
axs[0].set_title('Train '+option)
axs[0].plot(y, y, 'blue'); axs[0].set_xlabel('True '+option); axs[0].set_ylabel('Predicted '+option)
axs[0].plot(y, y_pred_train, 'ko')
axs[0].text(-0.3, 0.08, 'RMSE = '+str(round(rmse_train,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='black', fontsize=10)
axs[1].plot(y_test, y_test,  'blue')
axs[1].plot(y_test, y_pred_test, 'go')
axs[1].set_title('Test '+option)
axs[1].text(1.2, 0.08, 'RMSE = '+str(round(rmse,2)), verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes,color='green', fontsize=10)
axs[1].plot(y, y, 'blue'); axs[1].set_xlabel('True '+option);
plt.show()

ct +=1
method = Methods[ct]
y_pred_train = pd.DataFrame(y_pred_train, columns=['y_pred_train'], index=y.index)
y_pred_test = pd.DataFrame(y_pred_test, columns=['y_pred_test'], index=y_test.index)
train_all[method] = y_pred_train
test_all[method] = y_pred_test
error.loc['MSE', method] = mse.round()
error.loc['RMSE', method] = rmse.round()
fig.savefig('./ML_Results/'+str(ct)+"_"+method+'.jpg')

#Save Model
pkl_filename = method+".pkl"
with open('./ML_Results/models/'+pkl_filename, 'wb') as file:
    pickle.dump(rgr, file)
#Load Model    
with open('./ML_Results/models/'+pkl_filename, 'rb') as file:
    model = pickle.load(file)

print(error)
#----------------------- End Neural Network -----------------------------

#%%
#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
#--*--*--*--*--*--*--*--*--* Save Results *--*--*--*--*--*--*--*--*--*--*
#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*


error.to_excel('./ML_Results/Errors.xlsx')
#train_all = train_all.merge(df['DEPT'], left_index=True, right_index=True)
train_all['Set'] = 'Train'
#test_all = test_all.merge(df['DEPT'], left_index=True, right_index=True)
test_all['Set'] = 'Test'
full_set = train_all.append(test_all)
full_set.to_excel('./ML_Results/Train_Test_Results.xls')
full_set['Well'=='T2'].sort_values('DEPT').to_excel('./ML_Results/T2_test/Results.xls')

#%%

