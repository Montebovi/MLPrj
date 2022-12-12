import tensorflow as tf
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import timeit
from sklearn.model_selection import train_test_split


DATASET_PATH = "H:/PycharmProjects/mynn/MLCUP1.csv"

dataset = pd.read_csv(DATASET_PATH)

X_train = dataset.drop(["output1", "output2", "ID"], axis=1)
# X = X.drop("output2", axis=1)

y_train = dataset.drop(
    ["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)

print(X_train.head())
print(y_train.head())

X = np.array( X_train.values.tolist())
y = np.array(y_train.values.tolist())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=39)

initializer = tf.keras.initializers.RandomUniform(minval=-0.7, maxval=0.7)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="tanh", kernel_initializer=initializer,
                                                   kernel_regularizer=regularizers.L1(1e-4),
                                                   bias_regularizer=regularizers.L1(1e-4)),
    tf.keras.layers.Dense(16, activation="tanh", kernel_initializer=initializer,
                          kernel_regularizer=regularizers.L1(1e-4),
                          bias_regularizer=regularizers.L1(1e-4)),
    tf.keras.layers.Dense(8, activation="tanh", kernel_initializer=initializer,
                                                   kernel_regularizer=regularizers.L1(1e-4),
                                                   bias_regularizer=regularizers.L1(1e-4)),
    tf.keras.layers.Dense(2)])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.8))
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    callbacks=[], shuffle=False, epochs=15000, batch_size=32)
