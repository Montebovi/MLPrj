import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Layers import Dense
from LossRegularization import RegL1
from Optimizers import SGD
from TrainCallbacks import TrainCallback, TestCallback, BestCheckPoint, EarlyStopping
from model import Model
import matplotlib.pyplot as plt
import os

DATASET_PATH = "H:/PycharmProjects/mynn/MLCUP1.csv"
dataset = pd.read_csv(DATASET_PATH)

X = dataset.drop(["output1", "output2", "ID"], axis=1)

y = dataset.drop(["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"],
                 axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=39)

plt.hist(np.array(y_val.output1), bins=100, log=False)
plt.xlabel('test output1 (39)')
plt.show()

plt.hist(np.array(y_val.output2), bins=100, log=False)
plt.xlabel('test output2 (39)')
plt.show()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

plt.hist(np.array(y_val.output1), bins=100, log=False)
plt.xlabel('test output1 (42)')
plt.show()

plt.hist(np.array(y_val.output2), bins=100, log=False)
plt.xlabel('test output2 (42)')
plt.show()

min = np.amin(y.output2) + 1.0
max = np.amax(y.output2) - 0.5

# 5 bins may be too few for larger datasets.
bins = np.linspace(start=min, stop=max, num=50)
y_binned = np.digitize(y.output2, bins, right=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y_binned, random_state=42)

plt.hist(np.array(y_train.output2), bins=50, log=False)
plt.xlabel('train output2 (stratified)')
plt.show()

plt.hist(np.array(y_test.output2), bins=50, log=False)
plt.xlabel('test output2 (stratified)')
plt.show()

# plt.hist(np.array(y_train.output1), bins=50, log=False)
# plt.xlabel('train output1 (stratified)')
# plt.show()
#
# plt.hist(np.array(y_test.output1), bins=50, log=False)
# plt.xlabel('test output1 (stratified)')
# plt.show()


X = np.array(X.values.tolist())
y = np.array(y.values.tolist())

min = np.min(y[:,1]) + 1
max = np.max(y[:,1]) - 0.5

# 5 bins may be too few for larger datasets.
bins = np.linspace(start=min, stop=max, num=50)
y_binned = np.digitize(y[:,1], bins, right=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y_binned, random_state=42)
