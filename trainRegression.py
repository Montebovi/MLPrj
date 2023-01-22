import os
import pathlib

import pandas as pd
import numpy as np

import utils
from layers import Dense
from model import Model
from optimizers import SGD
from regularizarions import RegL1, RegL2
from trainCallback import BestCheckPoint, EarlyStopping

# configuration ########################################################
RETRAIN_MODEL = False
MODEL_FILENAME = "model_regression_mee5.pkl"
DATASET_PATH = "MLCUP1.csv"
MODELS_PATH = "models"
# ######################################################################

def trainM(layerDim, optimizer):
    dataset = pd.read_csv(DATASET_PATH)

    X_train = dataset.drop(["output1", "output2", "ID"], axis=1)
    y_train = dataset.drop(
        ["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)

    X = np.array(X_train.values.tolist())
    y = np.array(y_train.values.tolist())

    X_train, X_val, y_train, y_val = utils.straified_split(X, y, y_idx=0)

    model = Model()
    model.add(Dense(layerDim[0], activationFunName="tanh",
                    regularization=RegL1(kernel_regularizer=0.005, bias_regularizer=0.001)))
    model.add(Dense(layerDim[1], activationFunName="tanh",
                    regularization=RegL1(kernel_regularizer=0.005, bias_regularizer=0.001)))
    model.add(Dense(layerDim[2], activationFunName="tanh",
                    regularization=RegL1(kernel_regularizer=0.005, bias_regularizer=0.001)))
    model.add(Dense(2))

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    model_filename = os.path.join(MODELS_PATH, MODEL_FILENAME)

    cbackBestModel = BestCheckPoint(model_filename, monitor="val_data_loss")
    earlystopping = EarlyStopping(patience=500, monitor="val_data_loss")

    bestPrms = cbackBestModel.getParams()
    # print('best last checkpoint', bestPrms)

    model.compile(lossname="mee", optimizer=SGD(learning_rate=optimizer[0], decay=optimizer[1], momentum=optimizer[2], min_learning_rate=0.0004))
    # model.compile(lossname="mse", optimizer=SGD(learning_rate=0.01, decay=0.0001, momentum=0.5))

    do_train = True
    if os.path.exists(model_filename):
        model.load_weights(model_filename)
        do_train = RETRAIN_MODEL

    if do_train:
        model.train(X_train, y_train, epochs=10000, batch_size=32, log_every=5000, X_val=X_val, y_val=y_val,
                    callbacks=[earlystopping])

    # utils.printResults(model,X_val,y_val)
    # utils.printHistory(model)

    # model.save(model_filename + "_model")
    return earlystopping.getMinDataLoss()
