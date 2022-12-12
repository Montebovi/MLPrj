import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import timeit
from sklearn.model_selection import train_test_split
from ActivationFunctions import ActivationBase

np.random.seed(8)
X = np.random.rand(500, 2)
y = [(x[0] * x[0] + 2 * x[1] + 0) for x in X]
y = y + np.random.rand(500)/4
y = np.reshape(y, (500, 1))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=39)
# X_train=X
# y_train=y
# X_val=None
# y_val = None

# X_train = np.random.rand(100, 2)
# y_train = [(x[0] * x[0] + 2 * x[1]+0) for x in X_train]
# y_train = np.reshape(y_train, (100, 1))

initializer = tf.keras.initializers.RandomUniform(minval=-0.7, maxval=0.7)

model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=initializer,
                                                   kernel_regularizer=regularizers.L1(1e-4),
                                                   bias_regularizer=regularizers.L1(1e-4)),
                             tf.keras.layers.Dense(1)])

# model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=initializer),
#                              tf.keras.layers.Dense(1)])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.8))

start = timeit.default_timer()

early_stopping = EarlyStopping(patience=10, verbose=1, monitor='val_loss')

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    callbacks=[early_stopping], shuffle=False, epochs=15000, batch_size=10)

stop = timeit.default_timer()

print('Time: ', stop - start)
