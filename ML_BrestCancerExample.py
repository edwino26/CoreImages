
# %%
import numpy as np
import keras.datasets as keras_data

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# %%
