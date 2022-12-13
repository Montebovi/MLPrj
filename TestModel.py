import numpy as np
from sklearn.model_selection import train_test_split

from Layers import Dense
from LossRegularization import RegL1
from TrainCallbacks import TestCallback, EarlyStopping
from model import Model
from Optimizers import SGD

import timeit

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

model = Model()
model.add(Dense(32, activationFunName="tanh", regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=1e-4)))
model.add(Dense(1))
model.compile(lossname="mse", optimizer=SGD(learning_rate=0.03, decay=0.00, momentum=0.8))

start = timeit.default_timer()


early_stopping = EarlyStopping(patience=10)
# test_cback = TestCallback()

model.train(X_train, y_train, epochs=15000, batch_size=10, log_every=50, X_val=X_val, y_val=y_val, callbacks=[early_stopping])

stop = timeit.default_timer()
print('Time: ', stop - start)
