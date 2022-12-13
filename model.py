# Model class
import Optimizers
from Layers import Input
from BaseLoss import BaseLoss
from Optimizers import OptimezerBase
import pickle

class Model:
    def __init__(self):
        # Create a list of network objects
        self.endTraining = None
        self.trainable_layers = None
        self.input_layer = None
        self.optimizer = None
        self.loss = None
        self.layers = []
        self.parameters = dict()
        self.history = dict()

    @staticmethod
    def load(filename):
        with open(filename, "rb") as filehandler:
            return pickle.load(filehandler)

    def save(self,filename):
        with open(filename, "wb") as filehandler:
            pickle.dump(self, filehandler)

    def load_weights(self, filename):
        layer_count = len(self.layers)
        with open(filename, "rb") as filehandler:
            all_weights = pickle.load(filehandler)
        assert len(all_weights) == layer_count, f"numero layer ({layer_count}) diverso da quello atteso ({len(all_weights)})."
        for i in range(layer_count):
            self.layers[i].setWeights(all_weights[i])

    def save_weights(self, filename):
        layer_count = len(self.layers)
        all_weights = []
        for i in range(layer_count):
            weightsOfLayer = self.layers[i].getWeights()
            all_weights.append(weightsOfLayer)
        with open(filename, "wb") as filehandler:
            pickle.dump(all_weights, filehandler)

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, *, lossname, optimizer):
        self.loss = BaseLoss.GetLossByName(lossname)
        self.optimizer = optimizer
        self.input_layer = Input()
        layer_count = len(self.layers)
        # self.trainable_layers = []
        assert (layer_count > 1), "It's not a deep network"

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

            # if hasattr(self.layers[i], 'weights'):
            #     self.trainable_layers.append(self.layers[i])

    def stopTraining(self):
        self.endTraining = True

    def predict(self,X):
        predictions = self.forward(X, training=False)
        return predictions

    def train(self, X, y, *, epochs=1, batch_size=None, X_val=None, y_val=None, log_every=1,
              callbacks=[]):

        self.history = dict()
        self.history['loss'] = []
        self.history['val_loss'] = []

        self.endTraining = False

        self.parameters['loss'] = None
        self.parameters['val_loss'] = None

        train_steps = 1
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        for cback in callbacks:
            cback.set_model(self)
            cback.on_train_begin()

        for epoch in range(1, epochs + 1):

            for cback in callbacks:
                cback.on_epoch_begin(epoch)

            self.loss.new_pass()
            for step in range(train_steps):
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = self.loss.calculate(output, batch_y, self.layers)
                '''
                data_loss = self.loss.forward(output, batch_y)

                regularization_loss = 0
                for layer in self.layers:
                    if layer.isTrainable():
                        regularization_loss += layer.regularization_loss()
                '''

                loss = data_loss + regularization_loss
                predictions = output

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.layers:
                    if layer.isTrainable():
                        self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(self.layers)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            self.parameters['loss'] = epoch_loss
            self.parameters['data_loss'] = epoch_data_loss
            self.parameters['regularization_loss'] = epoch_regularization_loss

            self.history['loss'].append(epoch_loss)


            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val, training=False)
                # Calculate the loss
                val_data_loss, val_regularization_loss = self.loss.calculate(val_output, y_val, self.layers)
                val_loss = val_data_loss + regularization_loss
                # predictions = val_output
                self.parameters['val_data_loss'] = val_data_loss
                self.parameters['val_loss'] = val_loss
                self.history['val_loss'].append(val_loss)


            if not epoch % log_every:
                print(f'epoch: {epoch}, ' +
                      f'loss: {epoch_loss:.8f} (' +
                      f'data_loss: {epoch_data_loss:.4f}, ' +
                      f'reg_loss: {epoch_regularization_loss:.4f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
                if X_val is not None and y_val is not None:
                    # Print a summary
                    print(f'validation loss: {val_loss:.8f}   data_loss: {val_data_loss}')

            for cback in callbacks:
                cback.on_epoch_end(epoch)

            if self.endTraining:
                break

        for cback in callbacks:
            cback.on_train_end(epoch)

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
