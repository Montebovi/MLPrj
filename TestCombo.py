import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from Layers import Dense
from LossRegularization import RegL1
from Optimizers import SGD, Optimizer_Adam
from TrainCallbacks import TrainCallback, TestCallback, BestCheckPoint, EarlyStopping
from model import Model


def straified_split(x, y, y_idx=0):
    ysorted = np.sort(y)
    min = np.max(ysorted[0:4, y_idx])
    max = np.min(ysorted[-4:, y_idx])

    bins = np.linspace(start=min, stop=max, num=30)
    y_binned = np.digitize(y[:, y_idx], bins, right=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y_binned, random_state=42)
    return X_train, X_test, y_train, y_test


def showResults(modelName, y_val, predict):
    yVeri1 = y_val.transpose()[0]
    yPredetti1 = predict.transpose()[0]
    plt.scatter(yVeri1, yPredetti1)
    plt.scatter(np.arange(np.min(yVeri1), np.max(yVeri1), 0.4),
                np.arange(np.min(yVeri1), np.max(yVeri1), 0.4), color="red", linewidths=0.1)
    plt.xlabel(modelName + ' - output1')
    plt.show()

    yVeri2 = y_val.transpose()[1]
    yPredetti2 = predict.transpose()[1]
    plt.scatter(yVeri2, yPredetti2)
    plt.scatter(np.arange(np.min(yVeri2), np.max(yVeri2), 0.4),
                np.arange(np.min(yVeri2), np.max(yVeri2), 0.4), color="red", linewidths=0.1)
    plt.xlabel(modelName + ' - output2')
    plt.show()

    diff1 = yPredetti1 - yVeri1
    tot1 = np.sum(abs(diff1))
    print(modelName + " - differenza cumulativa (output1): ", tot1)
    print(modelName + " - Differenza media (output1): ", tot1 / len(X_val))

    diff2 = yPredetti2 - yVeri2
    tot2 = np.sum(abs(diff2))
    print(modelName + " - differenza cumulativa (output2): ", tot2)
    print(modelName + " - Differenza media (output2): ", tot2 / len(X_val))


DATASET_PATH = "C:/Users/Michele/Documents/MLCUP.csv"

filename1 = "testmodel2_10bins2.pkl"
filename2 = "testmodel2_10bins.pkl"

dataset = pd.read_csv(DATASET_PATH)

X_train = dataset.drop(["output1", "output2", "ID"], axis=1)
y_train = dataset.drop(
    ["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)

X = np.array(X_train.values.tolist())
y = np.array(y_train.values.tolist())

X_train, X_val, y_train, y_val = straified_split(X, y, y_idx=0)

# model1 = Model()
# model1.add(Dense(6, activationFunName="tanh",
#                  regularization=RegL1(kernel_regularizer=1e-2, bias_regularizer=0)))
# model1.add(Dense(4, activationFunName="tanh",
#                  regularization=RegL1(kernel_regularizer=1e-3, bias_regularizer=0)))
# model1.add(Dense(4, activationFunName="tanh",
#                  regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=0)))
# model1.add(Dense(2))
# model1.compile(lossname="mse", optimizer=SGD(learning_rate=0.001, decay=0.0, momentum=0.0))
# model1.load_weights(filename1)
#
# predict1 = model1.predict(X_val)
# showResults("modello 1", y_val, predict1)
#
# model2 = Model()
# model2.add(Dense(6, activationFunName="tanh",
#                  regularization=RegL1(kernel_regularizer=1e-2, bias_regularizer=0)))
# model2.add(Dense(4, activationFunName="tanh",
#                  regularization=RegL1(kernel_regularizer=1e-3, bias_regularizer=0)))
# model2.add(Dense(4, activationFunName="tanh",
#                  regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=0)))
# model2.add(Dense(2))
# model2.compile(lossname="mse", optimizer=Optimizer_Adam())
# model2.load_weights(filename2)
#
# predict2 = model2.predict(X_val)
# showResults("modello 2", y_val, predict2)
#
# predict3 = (predict1+predict2)/2
# showResults("COMBO",y_val,predict3)
#
# model2.save("prova.pkl")
model1 = Model.load("models/model.pkl")
predict1 = model1.predict(X_val)
showResults("modello 1", y_val, predict1)

model2 = Model.load("models/model3.pkl")
predict2 = model2.predict(X_val)
showResults("modello 2", y_val, predict2)

predict3 = (predict2 + predict1)/2
showResults("modello combo", y_val, predict3)

# model3 = Model.load("models/model3.pkl")
# predict3 = model3.predict(X_val)
# showResults("modello 2", y_val, predict3)
#
# predict4 = (predict3 + predict2 + predict1)/3
# showResults("modello combo", y_val, predict4)
