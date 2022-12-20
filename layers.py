import numpy as np
from activationFunctions import ActivationBase

class LayerBase:
    def __init__(self):
        self.regularization = None
        self.prev = None
        self.next = None
        pass

    def getWeights(self):
        return []

    def setWeights(self, data):
        pass

    def isTrainable(self):
        return False

    # Forward pass
    def forward(self, inputs, training):
        pass

    def backward(self, dvalues):
        pass

    def regularization_loss(self):
        if self.regularization is None:
            return 0.
        else:
            return self.regularization.regularization_loss(self)

class Input(LayerBase):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs, training):
        self.output = inputs

class Dense(LayerBase):
    def __init__(self, n_neurons, *, n_inputs=None, activationFunName=None, regularization=None):
        super().__init__()
        self.regularization = regularization
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        if n_inputs is not None:
            self._set_n_inputs(n_inputs)

        self._startWeights = None
        self._startBiases = None

        self.output = None
        self.dinputs = None
        self.dbiases = None
        self.dweights = None

        if activationFunName is not None:
            self.activation = ActivationBase.GetActivationByName(activationFunName)
        else:
            self.activation = None

    def getWeights(self):
        data = dict()
        data['w'] = self.weights
        data['b'] = self.biases
        return data

    def setWeights(self, data):
        if self.n_inputs is None:
            self._startWeights = data['w']
            self._startBiases = data['b']
        else:
            self.weights = data['w']
            self.biases = data['b']
        pass

    def isTrainable(self):
        return True

    def _set_n_inputs(self, n_inputs):
        self.n_inputs = n_inputs
        if self._startWeights is not None:
            assert self._startWeights.shape[0] == n_inputs
            assert self._startWeights.shape[1] == self.n_neurons
            self.weights = self._startWeights
        else:
            # self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            # self.weights = np.random.uniform(-0.7, 0.7, (n_inputs, self.n_neurons))
            # Glorot/Xavier initialization
            # https://pyimagesearch.com/2021/05/06/understanding-weight-initialization-for-neural-networks/
            F_in = n_inputs
            F_out = self.n_neurons
            limit = np.sqrt(2 / float(F_in + F_out))
            self.weights = np.random.normal(0.0, limit, size=(F_in, F_out))


        if self._startBiases is not None:
            assert self._startBiases.shape[0] == 1
            assert self._startBiases.shape[1] == self.n_neurons
            self.biases = self._startBiases
        else:
            # self.biases = np.random.rand(1, self.n_neurons)
            self.biases = np.zeros((1, self.n_neurons))

    # Forward pass
    def forward(self, inputs, training):
        if self.n_inputs is None:
            self._set_n_inputs(inputs.shape[1])

        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation is not None:
            self.activation.forward(self.output)
            self.output = self.activation.output

    # Backward pass
    def backward(self, dvalues):
        if self.activation is not None:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.regularization is not None:
            self.dweights, self.dbiases = self.regularization.regularization_layer(self)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
