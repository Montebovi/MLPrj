import tensorflow as tf
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras.models import Model
import timeit
from sklearn.model_selection import train_test_split
from keras.layers import Concatenate, Dense, Input, concatenate


DATASET_PATH = "H:/PycharmProjects/mynn/MLCUP1.csv"

dataset = pd.read_csv(DATASET_PATH)

X = dataset.drop(["output1", "output2", "ID"], axis=1)
# X = X.drop("output2", axis=1)

y = dataset.drop(
    ["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)

print(X.head())
print(y.head())

# X = np.array( X_train.values.tolist())
# y = np.array(y_train.values.tolist())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=39)

initializer = tf.keras.initializers.RandomUniform(minval=-0.7, maxval=0.7)

first_input = Input(shape=(X.shape[1], ))
dense1 = Dense(4,activation="tanh",kernel_initializer=initializer,
                    kernel_regularizer=regularizers.L1(1e-4),
                    bias_regularizer=regularizers.L1(1e-4) )(first_input)
dense2 = Dense(2,activation="tanh",kernel_initializer=initializer,
                    kernel_regularizer=regularizers.L1(1e-4),
                    bias_regularizer=regularizers.L1(1e-4) )(dense1)
# dense3 = Dense(4,activation="tanh",kernel_initializer=initializer,
#                     kernel_regularizer=regularizers.L1(1e-4),
#                     bias_regularizer=regularizers.L1(1e-4) )(dense2)
out1 = Dense(1,activation="tanh",kernel_initializer=initializer)(dense2)
merge1 = concatenate([out1,first_input])
dense4 = Dense(4,activation="tanh",kernel_initializer=initializer,
                    kernel_regularizer=regularizers.L1(1e-4),
                    bias_regularizer=regularizers.L1(1e-4) )(merge1)
# dense5 = Dense(4,activation="tanh",kernel_initializer=initializer,
#                     kernel_regularizer=regularizers.L1(1e-4),
#                     bias_regularizer=regularizers.L1(1e-4) )(dense4)
out2 = Dense(2,kernel_initializer=initializer)(dense4)

model = Model(inputs=[first_input], outputs=out2)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, activation="tanh", kernel_initializer=initializer,
#                                                    kernel_regularizer=regularizers.L1(1e-4),
#                                                    bias_regularizer=regularizers.L1(1e-4)),
#     tf.keras.layers.Dense(16, activation="tanh", kernel_initializer=initializer,
#                           kernel_regularizer=regularizers.L1(1e-4),
#                           bias_regularizer=regularizers.L1(1e-4)),
#     tf.keras.layers.Dense(8, activation="tanh", kernel_initializer=initializer,
#                                                    kernel_regularizer=regularizers.L1(1e-4),
#                                                    bias_regularizer=regularizers.L1(1e-4)),
#     tf.keras.layers.Dense(2)])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, weight_decay=0.0001,  momentum=0.8))
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    callbacks=[], shuffle=False, epochs=500, batch_size=32)

predict = model.predict(X_val)

diff1 = predict - y_val
tot1 = np.sum(abs(diff1))
print("differenza cumulativa: ", tot1)
print("Differenza media : ", tot1 / len(X_val))

