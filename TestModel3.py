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

retraining1 = False

class LogTotaleDiff(TrainCallback):
    def __init__(self, everyNumEpochs=50):
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


def straified_split(x,y,y_idx=0):
    min = np.min(y[:,y_idx]) + 1.0
    max = np.amax(y[:,y_idx]) - 0.5
    bins = np.linspace(start=min, stop=max, num=50)
    y_binned = np.digitize(y[:,y_idx], bins, right=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y_binned, random_state=42)
    return X_train, X_test, y_train, y_test


# random_split = 42
random_split = 39

DATASET_PATH = "H:/PycharmProjects/mynn/MLCUP1.csv"

dataset = pd.read_csv(DATASET_PATH)

X_train = dataset.drop(["output1","output2",  "ID"], axis=1)
# X = X.drop("output2", axis=1)

y_train = dataset.drop(["ID", "output2", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)

# print(X_train.head())
# print(y_train.head())

X = np.array(X_train.values.tolist())
y = np.array(y_train.values.tolist())

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_split)
X_train, X_val, y_train, y_val = straified_split(X,y,y_idx=0)

model = Model()
model.add(Dense(4, activationFunName="tanh",
                regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=0)))
model.add(Dense(4, activationFunName="tanh",
                regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=0)))
model.add(Dense(1,regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=0)))

filename_model1 = "models/model1-best.pkl"

cbackLogTotaleDiff = LogTotaleDiff(everyNumEpochs=50)
cbackBestModel1 = BestCheckPoint(filename_model1,monitor="val_data_loss")
earlystopping1 = EarlyStopping(patience=200,monitor="val_data_loss")

bestPrms = cbackBestModel1.getParams()
print('best last checkpoint',bestPrms)

model.compile(lossname="mse", optimizer=SGD(learning_rate=0.002, decay=0.0001, momentum=0.8))
if True:
    if os.path.exists(filename_model1):
        model.load_weights(filename_model1)
        do_train1 = retraining1
    else:
        do_train1 = True
    if do_train1:
        model.train(X_train, y_train, epochs=20000, batch_size=32, log_every=50, X_val=X_val, y_val=y_val,
            callbacks=[cbackBestModel1,earlystopping1])
        model.load_weights(filename_model1)

    predict = model.predict(X_val)

    yVeri1 = y_val.transpose()[0]
    yPredetti1 = predict.transpose()[0]
    plt.scatter(yVeri1, yPredetti1)
    plt.scatter(np.arange(np.min(yVeri1), np.max(yVeri1), 0.4),
            np.arange(np.min(yVeri1), np.max(yVeri1), 0.4), color="red", linewidths=0.1)
    plt.xlabel('output1')
    plt.show()

    diff1 = yPredetti1 - yVeri1
    tot1 = np.sum(abs(diff1))
    print("differenza cumulativa (output1): ", tot1)
    print("Differenza media (output1): ", tot1 / len(X_val))



# #####################################################################
teacher_forcing = False
retraining2 = True
filename_model2 = "models/model2-best.pkl"

X_traintemp = dataset.drop(["output1","output2",  "ID"], axis=1)
predict_traintemp = model.predict(X_traintemp)    # predict output1

X_train2 = dataset.drop(["output2", "ID"], axis=1)
X_train2.output1 = predict_traintemp

#X_train2['input10'] = X_train2.input1 * X_train2.input2

# X = X.drop("output2", axis=1)
y_train2 = dataset.drop(["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)


X = np.array(X_train2.values.tolist())
y = np.array(y_train2.values.tolist())

if teacher_forcing:
    X_train2originale = dataset.drop(["output2",  "ID"], axis=1)
    #X_train2originale['input10'] = X_train2originale.input1 * X_train2originale.input2
    Xorig = np.array(X_train2originale.values.tolist())
    #_, X_val2, y_train2, y_val2 = train_test_split(X, y, test_size=0.2, random_state=random_split)
    _, X_val2, y_train2, y_val2 = straified_split(X, y)
    X_train2, _, _, _ = train_test_split(Xorig, y, test_size=0.2, random_state=random_split)
else:
    #X_train2, X_val2, y_train2, y_val2 = train_test_split(X, y, test_size=0.2, random_state=random_split)
    X_train2, X_val2, y_train2, y_val2 = straified_split(X, y)



cbackBestModel2 = BestCheckPoint(filename_model2,monitor="val_data_loss")
earlystopping = EarlyStopping(patience=200,monitor="val_data_loss")

bestPrms2 = cbackBestModel2.getParams()
print('best last checkpoint model 2',bestPrms2)

model2 = Model()
model2.add(Dense(8, activationFunName="tanh",
                regularization=RegL1(kernel_regularizer=1e-2, bias_regularizer=0)))
model2.add(Dense(6, activationFunName="tanh",
                regularization=RegL1(kernel_regularizer=1e-3, bias_regularizer=0)))
# model2.add(Dense(4, activationFunName="tanh",
#                 regularization=RegL1(kernel_regularizer=1e-3, bias_regularizer=0)))

model2.add(Dense(2,regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=0)))

cbackLogTotaleDiff = LogTotaleDiff(everyNumEpochs=50)

model2.compile(lossname="mse",
               optimizer=SGD(learning_rate=0.00000001, decay=0.000006, momentum=0.8, min_learning_rate=4.0e-06))

if os.path.exists(filename_model2):
    model2.load_weights(filename_model2)
    do_train = retraining2
else:
    do_train = True

if do_train:
    model2.train(X_train2, y_train2, epochs=30000, batch_size=32, log_every=50, X_val=X_val2, y_val=y_val2,
            callbacks=[cbackBestModel2,earlystopping])
    model2.load_weights(filename_model2)

predict2 = model2.predict(X_val2)

import matplotlib.pyplot as plt

yVeri1a = y_val2.transpose()[0]
yPredetti1a = predict2.transpose()[0]
plt.scatter(yVeri1a, yPredetti1a)
plt.scatter(np.arange(np.min(yVeri1a), np.max(yVeri1a), 0.4),
            np.arange(np.min(yVeri1a), np.max(yVeri1a), 0.4), color="red", linewidths=0.1)
plt.xlabel('output2')
plt.show()

diff1a = yPredetti1a - yVeri1a
tot1a = np.sum(abs(diff1a))
print("differenza cumulativa (output1): ", tot1a)
print("Differenza media (output1): ", tot1a / len(X_val2))

yVeri2 = y_val2.transpose()[1]
yPredetti2 = predict2.transpose()[1]
plt.scatter(yVeri2, yPredetti2)
plt.scatter(np.arange(np.min(yVeri2), np.max(yVeri2), 0.4),
            np.arange(np.min(yVeri2), np.max(yVeri2), 0.4), color="red", linewidths=0.1)
plt.xlabel('output2')
plt.show()

diff2 = yPredetti2 - yVeri2
tot2 = np.sum(abs(diff2))
print("differenza cumulativa (output2): ", tot2)
print("Differenza media (output2): ", tot2 / len(X_val2))


