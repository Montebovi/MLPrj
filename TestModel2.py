import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Layers import Dense
from LossRegularization import RegL1
from Optimizers import SGD, Optimizer_Adam
from TrainCallbacks import TrainCallback, TestCallback, BestCheckPoint, EarlyStopping
from model import Model

retraining = True

def straified_split(x,y,y_idx=0):
    ysorted = np.sort(y)
    min = np.max(ysorted[0:4, y_idx])
    max = np.min(ysorted[-4:, y_idx])

    bins = np.linspace(start=min, stop=max, num=50)
    y_binned = np.digitize(y[:,y_idx], bins, right=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y_binned, random_state=42)
    return X_train, X_test, y_train, y_test


class LogTotaleDiff(TrainCallback):
    def __init__(self, everyNumEpochs=10):
        super().__init__()
        self.everyNumEpochs = everyNumEpochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.everyNumEpochs == 0):
            predict = model.predict(X_val)
            yVeri1 = y_val.transpose()[0]
            yPredetti1 = predict.transpose()[0]
            diff1 = yPredetti1 - yVeri1
            tot1 = np.sum(abs(diff1))
            print("differenza cumulativa (output1): ", tot1)
            print("Differenza media (output1): ", tot1 / len(X_val))


# MODE = "out2"
# MODE = "out1"
MODE = "all"

filename = "models/model3.pkl"
DATASET_PATH = "C:/Users/Michele/Documents/MLCUP.csv"

dataset = pd.read_csv(DATASET_PATH)

X_train = dataset.drop(["output1","output2",  "ID"], axis=1)
# X = X.drop("output2", axis=1)

if MODE == 'all':
    y_train = dataset.drop(
        ["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)
elif MODE == 'out1':
    y_train = dataset.drop(
        ["output2", "ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"],
        axis=1)
elif MODE == 'out2':
    y_train = dataset.drop(
        ["output1", "ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"],
        axis=1)

# print(X_train.head())
# print(y_train.head())

X = np.array(X_train.values.tolist())
y = np.array(y_train.values.tolist())

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=39)
X_train, X_val, y_train, y_val = straified_split(X, y,y_idx=0)

model = Model()
model.add(Dense(4, activationFunName="tanh",
                regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=1e-4)))
model.add(Dense(4, activationFunName="tanh",
                regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=1e-4)))
model.add(Dense(2, activationFunName="tanh",
                regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=1e-4)))

if MODE == 'all':
    model.add(Dense(2,
                regularization=RegL1(kernel_regularizer=1e-3, bias_regularizer=0)))
else:
    model.add(Dense(1))

cbackLogTotaleDiff = LogTotaleDiff(everyNumEpochs=100)
cbackBestModel = BestCheckPoint(filename,monitor="val_data_loss")
earlystopping = EarlyStopping(patience=300,monitor="val_data_loss")

bestPrms = cbackBestModel.getParams()
print('best last checkpoint',bestPrms)

# uso SGD
# model.compile(lossname="mse", optimizer=SGD(learning_rate=0.001, decay=0.0, momentum=0.5))
model.compile(lossname="mse", optimizer=Optimizer_Adam(learning_rate=0.001))

if os.path.exists(filename):
    model.load_weights(filename)
    do_train = retraining
else:
    do_train = True
if do_train:
    model.train(X_train, y_train, epochs=10000, batch_size=32, log_every=50, X_val=X_val, y_val=y_val,
                 callbacks=[cbackBestModel,earlystopping])

predict = model.predict(X_val)

import matplotlib.pyplot as plt

yVeri1 = y_val.transpose()[0]
yPredetti1 = predict.transpose()[0]
plt.scatter(yVeri1, yPredetti1)
plt.scatter(np.arange(np.min(yVeri1), np.max(yVeri1), 0.4),
            np.arange(np.min(yVeri1), np.max(yVeri1), 0.4), color="red", linewidths=0.1)
plt.xlabel('output1')
plt.show()

if MODE == 'all':
    yVeri2 = y_val.transpose()[1]
    yPredetti2 = predict.transpose()[1]
    plt.scatter(yVeri2, yPredetti2)
    plt.scatter(np.arange(np.min(yVeri2), np.max(yVeri2), 0.4),
            np.arange(np.min(yVeri2), np.max(yVeri2), 0.4), color="red", linewidths=0.1)
    plt.xlabel('output2')
    plt.show()

diff1 = yPredetti1 - yVeri1
tot1 = np.sum(abs(diff1))
print("differenza cumulativa (output1): ", tot1)
print("Differenza media (output1): ", tot1 / len(X_val))

if MODE == 'all':
    diff2 = yPredetti2 - yVeri2
    tot2 = np.sum(abs(diff2))
    print("differenza cumulativa (output2): ", tot2)
    print("Differenza media (output2): ", tot2 / len(X_val))


if any(model.history):
    pd.DataFrame(model.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.ylim([1.0, 3])
    plt.show()

model.save(filename)
