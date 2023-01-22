import os
import pathlib
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import utils
from accuracy import Accuracy_Categorical
from model import Model
from layers import Dense
from optimizers import SGD, Optimizer_Adam
from regularizarions import RegL1
from trainCallback import EarlyStopping


def convertOneShot(x_input):
    l = []
    for x in x_input:
        b = np.zeros((x.size, 4))
        b[np.arange(6), x - 1] = 1
        b = np.reshape(b, 24)
        l.append(b)
    l = np.array(l)
    dropped = np.delete(l, [3, 7,10,11,15,22,23], 1)
    return dropped


# configuration ########################################################
DATASET_PATH = "classificazione/monks-3.csv"
# ######################################################################

dataset = pd.read_csv(DATASET_PATH)

X_train = dataset.drop(["output"], axis=1)
y_train = dataset.drop(["input1", "input2", "input3", "input4", "input5", "input6"], axis=1)

X_values = np.array(X_train.values.tolist())
y = np.array(y_train.values.tolist())

X = convertOneShot(X_values)

X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.3,random_state=3)
X_val, X_test, y_val, y_test = utils.straified_split(X_val, y_val, y_idx=0, test_size=0.2)

model = Model()
model.add(Dense(8, activationFunName="tanh"))
model.add(Dense(1, activationFunName="sigmoid"))

accuracy = Accuracy_Categorical()
earlystopping = EarlyStopping(patience=500, monitor="val_data_loss")
# model.compile(lossname="bce", optimizer=SGD(learning_rate=0.002, decay=0.0, momentum=0.0), accuracy= accuracy)
model.compile(lossname="bce", optimizer=Optimizer_Adam(learning_rate=0.0005, decay=0.00001), accuracy=accuracy)

model.train(X_train, y_train, epochs=10000, batch_size=16, log_every=500, X_val=X_val, y_val=y_val, callbacks=[earlystopping])

utils.printLossHistory(model,0,1)
utils.printAccuracyHistory(model)

res_test = model.model_assessment(X_test, y_test)
print("--------------------------------------")
print("TEST dataset results")
print(res_test)