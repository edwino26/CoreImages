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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import SCORERS

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


# %% =========================  Machine Learning: PROCESSING ===============================
print(df.shape)
df['Pay'] = df['GRAY'].apply(lambda x: 1 if x> 170 else 0)
data = df.drop(['Well', 'DEPT'], axis=1).copy()
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
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# %%
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o--o-  GRID SEARCH for ALL ALGORITHMS o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
score_RMSE = 'neg_root_mean_squared_error'
# Lasso
rgr = linear_model.Lasso()
rgr.get_params() 

tuned_parameters = [{'alpha':np.linspace(0.01,0.99, 10)}]
gsr_lasso = GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
gsr_lasso.fit(X, y)
print(gsr_lasso.best_params_)
print("Best RMSE: ", gsr_lasso.best_score_)


# for score in scores:
#     print()
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     gsr = GridSearchCV(
#         GradientBoostingRegressor(), tuned_parameters, scoring=score  #use cv=None for default 5-fold cross validation
#     )
#     gsr.fit(X, np.ravel(y))

#     print("Best parameters set found on development set:")
#     print()
#     print(gsr.best_params_)
#     print("Best RMSE: ", gsr.best_score_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = gsr.cv_results_['mean_test_score']
#     stds = gsr.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, gsr.cv_results_['params']):
#         if abs(mean) < 1000:  #Remove "exploding errors"
#             print("%0.3f (+/-%0.03f) for %r"
#                 % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("Done: the model is trained on the full development set.")
#     print("Done: the scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, rgr.predict(X_test)
#     #print(classification_report(y_true, y_pred))


# %%
