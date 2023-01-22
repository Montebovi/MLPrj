import sys
import os
import pathlib
import json
import time

import pandas as pd
import numpy as np

import utils
from layers import Dense
from model import Model
from optimizers import SGD, Optimizer_Adam
from regularizarions import RegL1, RegL2
from trainCallback import BestCheckPoint, EarlyStopping

from multiprocessing.pool import ThreadPool
from threading import Thread, Lock

# configuration ########################################################
DATASET_PATH = "MLCUP1.csv"
MODE_MULTI_THREAD = False
# ######################################################################

LAST_DENSE_LAYER = 2
denseLayersList = [
    # [10, 8, 6, 4],
    # [8, 6, 4, 2], [8, 4, 4, 2],
    # [6, 4, 4, 4], [6, 4, 4, 2],
    [16, 10, 8], [16, 8, 6], [16, 8, 4],
    # [12, 8, 8], [12, 8, 6], [12, 8, 4],
    # [10, 8, 8], [10, 8, 6], [10, 8, 4],
    # [8, 8, 6], [8, 6, 6], [8, 6, 4],
    # [6, 8, 6], [6, 6, 6], [6, 6, 4],
    # [6, 4, 4], [6, 4, 2], [4, 4, 4],
    # [32, 16], [16, 16], [16, 8],
    # [8, 8], [4, 4]

]

optimizersList = [
    [0.01, 0.0001, 0.6],  [0.01, 0.0001, 0.8],
    [0.01, 0.0003, 0.6],  [0.01, 0.0003, 0.8],
    [0.01, 0.0005, 0.6],  [0.01, 0.0005, 0.8],
]

regularizationsList = [
    ["L2", 0.0001, 0.0001],  ["L2", 0.0005, 0.0005], ["L2", 0.001, 0.001],
    ["L2", 0.002, 0.002], ["L2", 0.003, 0.003], ["L2", 0.004, 0.004],
    ["L2", 0.005, 0.005]
]

lock = Lock()

dataset = pd.read_csv(DATASET_PATH)
X_train = dataset.drop(["output1", "output2", "ID"], axis=1)
y_train = dataset.drop(
    ["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)

X = np.array(X_train.values.tolist())
y = np.array(y_train.values.tolist())

X_train, X_val, y_train, y_val = utils.straified_split(X, y, y_idx=0)
X_val, X_test, y_val, y_test = utils.straified_split(X_val, y_val, y_idx=0, test_size=0.25)


def createModel(layers, regCfg):
    model = Model()
    for units in layers:
        if (regCfg[0] == "L1"):
            r = RegL1(kernel_regularizer=regCfg[1], bias_regularizer=regCfg[2])
        elif (regCfg[0] == "L2"):
            r = RegL2(kernel_regularizer=regCfg[1], bias_regularizer=regCfg[2])
        else:
            raise Exception("Nome regolarizzazione non previsto.")
        model.add(Dense(units, activationFunName="tanh",regularization=r))
    model.add(Dense(2, activationFunName="linear"))
    return model

results = []
bestModel = {'val_data_loss': None}

# -------------------------------------------------------------------------------
def log_results(args):
    with lock:
        results.append(args)

        if bestModel['val_data_loss'] is None or bestModel['val_data_loss'] > args['min_val_loss']:
            bestModel['val_data_loss'] = args['min_val_loss']
            bestModel['num model'] = countModel
            bestModel['model'] = args
            json_object = json.dumps(bestModel, indent=4)
            text_file = open("best_model.txt", "w")
            text_file.write(json_object)
            text_file.close()

        json_object = json.dumps(args, indent=4)
        text_file = open("log_models.txt", "a")
        text_file.write(json_object)
        text_file.close()


# -------------------------------------------------------------------------------
def trainModel(args):
    model_desc = json.dumps(args)
    print("-------------------------------------------")
    # num += 1
    # perc = "{:.2f}".format(100.0 * num / totalModels)
    # print(f"({num} su {totalModels} - {perc}%) - {model_desc}")
    with lock:
        print(f"{model_desc}")

    model = createModel(args['model'], args['regularization'])
    earlystopping = EarlyStopping(patience=500, monitor="val_data_loss")
    optimizer = args['optimizer']
    model.compile(lossname="mee",
                  optimizer=SGD(learning_rate=optimizer[0], decay=optimizer[1], momentum=optimizer[2], min_learning_rate=0.0001))
    model.train(X_train, y_train, epochs=10000, batch_size=32, log_every=2500, X_val=X_val, y_val=y_val,
                callbacks=[earlystopping])

    with lock:
        args['min_val_loss'] = earlystopping.getMinDataLoss()
        args['epoch_of_min_val_loss'] = earlystopping.getEpochMinValue()
    log_results(args)

    return True




text_file = open("log_models.txt", "w")
text_file.write("")
text_file.close()


totalModels = len(denseLayersList)*len(optimizersList)*len(regularizationsList)
print(f"Modelli in generazione: {totalModels}\n")

errormsg = None
num = 0
poolargs = []

st = time.time()

try:
    countModel = 0
    for modelCfg in denseLayersList:
        countModel += 1
        countOptimizer = 0
        for optimizer in optimizersList:
            countOptimizer += 1
            countRegularizations = 0
            for regularization in regularizationsList:
                countRegularizations += 1

                result = dict()
                result['num'] = f"({countModel}.{countOptimizer}.{countRegularizations})"
                result['model'] = modelCfg
                result['optimizer'] = optimizer
                result['regularization'] = regularization

                if not MODE_MULTI_THREAD:
                    trainModel(result)
                else:
                    poolargs.append(result)

    if (MODE_MULTI_THREAD):
        with ThreadPool(16) as pool:
            for result in pool.map(trainModel, poolargs):
                pass

except Exception as e:
    errormsg = "Whew!" + str(e) + " occurred."
    print(errormsg)

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

json_object = json.dumps(results, indent=4)
text_file = open("log_models.txt", "w")
n = text_file.write(json_object)
if (errormsg is not None):
    text_file.write("\n" + errormsg)
text_file.close()

print("Terminato")
