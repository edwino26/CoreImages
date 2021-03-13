# %%
import glob
import lasio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#Clustering packages
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


# %%
files = glob.glob('./LAS/T2/T2_Logs.las')
las = lasio.read('./LAS/T2/T2_Logs.las')
las_DM = lasio.read('./LAS/T2/T2_DM-2077-4158ft.las')
##

# %%
well = las.well
headers = las.curves
params = las.params
logs = las.data
No_logs = len(headers)
dims = las.data.shape

print(dims)
i=0
for curve in las.curves:
    print(curve.mnemonic + ": " + str(curve.data) + " " + str(i))
    i += 1

data = pd.DataFrame(las.data)

DEPTH = las.index
GR = data[53] #las["GR_EDTC"]
RESD = data[225] #las["AT90"]
RHOB = data[108] #las["RHOZ"]
NPHI= data[96] #las["NPHI"]


BD = 3000
TD = 3880

# %%
# ======== Manual Neural Net ============




# plt.figure()
# plt.subplot(141)
# plt.plot(GR, DEPTH, 'green'); plt.axis([0, 120, BD, TD]); plt.gca().invert_yaxis()
# plt.subplot(142)
# plt.plot(RESD, DEPTH); plt.axis([0.1, 100, BD, TD]); plt.gca().invert_yaxis();plt.gca().yaxis.set_visible(False); plt.grid(True,which='minor',axis='x'); plt.xscale('log')
# plt.subplot(143)
# plt.plot(RHOB, DEPTH, 'red'); plt.axis([1.65, 2.65, BD, TD]); plt.gca().invert_yaxis(); plt.gca().yaxis.set_visible(False)
# plt.subplot(144)
# plt.plot(NPHI, DEPTH, 'blue')
# plt.gca().invert_yaxis(); plt.axis([0.6, 0, BD, TD]); plt.gca().invert_yaxis();plt.gca().yaxis.set_visible(False)
#
# plt.suptitle('Well logs for ' + las.well['WELL']['value'])
#
# # Plot Input Logs
# #plt.show()

#Explore
#data_o = data.dropna()
print(data.head())
data_o = data.iloc[:, [53, 225, 108, 96]]  #Removed Depth 0
data_o = data_o.dropna()
data_o = data_o.rename(columns={53: "GR", 225: 'RESD', 108: "RHOB", 96: "NPHI"})

train, test = train_test_split(data_o, test_size=0.2)
print("Size is:")
print(data_o.size)
print("Header")
print(data_o.head())

print(data_o.shape)

# %%
#Scale and Prepare Data
scaler = StandardScaler()


X = np.array([train['GR'], train['RESD'], train['RHOB']]).T
X_test = np.array([test['GR'], test['RESD'], test['RHOB']]).T

scaler.fit(X)   #Fit scaler on training data
X_test = scaler.transform(X_test)
X = scaler.transform(X)

y = np.array(train['NPHI'])
y_test = np.array(test['NPHI'])

#CLustering
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c


model = MLPRegressor(random_state=1, max_iter =500, hidden_layer_sizes=(100, 80, 50, 40, 30), validation_fraction=0)._fit(X,y)
model.predict(X_test)
print(model.score(X_test, y_test))

print(model.get_params(True))

fig = plt.figure()
plt.subplot(121)
plt.plot(y, model.predict(X), 'o')
plt.plot(y, y,  'blue')
ax = fig.add_subplot(122)
plt.plot(y_test, model.predict(X_test), 'o')
plt.plot(y_test, y_test,  'blue')
plt.suptitle('Training vs Test Data for NPHI prediction ' + las.well['WELL']['value'] + 'Well')
ax.text(0.95, 0.01, str(round(model.score(X_test, y_test),2)),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
plt.show()

# %%
