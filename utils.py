import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------------------------------
# split dataset
def straified_split(x,y,y_idx=0,test_size=0.3):
    ysorted = np.sort(y)
    min = np.max(ysorted[0:4, y_idx])
    max = np.min(ysorted[-4:, y_idx])

    bins = np.linspace(start=min, stop=max, num=50)
    y_binned = np.digitize(y[:,y_idx], bins, right=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y_binned, random_state=42)
    return X_train, X_test, y_train, y_test

#---------------------------------------------------------------------
# show history of model by plot
def printLossHistory(model, ylim_min = 1.0, ylim_max = 3.0):
    h = {'loss':model.history['loss'],'val_loss':model.history['val_loss']}
    if any(h):
        pd.DataFrame(h).plot()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.ylim([ylim_min, ylim_max])
        plt.show()

def printAccuracyHistory(model):
    h = {'accuracy':model.history['accuracy'],'val_accuracy':model.history['val_accuracy']}
    if any(h):
        pd.DataFrame(h).plot()
        plt.ylabel("accuracy")
        plt.xlabel("epochs")
        plt.ylim([0.0, 1.1])
        plt.show()


#---------------------------------------------------------------------
# show some results
def printResults(model, X_val, y_val):
    predict = model.predict(X_val)

    yVeri1 = y_val.transpose()[0]
    yPredetti1 = predict.transpose()[0]
    plt.scatter(yVeri1, yPredetti1)
    plt.scatter(np.arange(np.min(yVeri1), np.max(yVeri1), 0.4),
                np.arange(np.min(yVeri1), np.max(yVeri1), 0.4), color="red", linewidths=0.1)
    plt.xlabel('output1')
    plt.show()

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

    diff2 = yPredetti2 - yVeri2
    tot2 = np.sum(abs(diff2))
    print("differenza cumulativa (output2): ", tot2)
    print("Differenza media (output2): ", tot2 / len(X_val))