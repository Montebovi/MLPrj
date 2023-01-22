import numpy as np

# -------------------------------------------------------------------------------------------------
# activation function base class
class ActivationBase:
    def  __init__(self):
        pass

    # Forward pass
    def forward(self, inputs):
        pass

    # Backward pass
    def backward(self, dvalues):
        pass

    def predictions(self, outputs):
        pass

    # -------------------------------------------------------------------------------------------------
    @staticmethod
    def GetActivationByName(activationName):
        if activationName.lower() == "linear":
            return Linear()
        elif activationName.lower() == "sigmoid":
            return Sigmoid()
        elif activationName.lower() == "relu":
            return ReLU()
        elif activationName.lower() == "tanh":
            return TanH()
        else:
            raise NotImplementedError("Unexpected activation function")



# -------------------------------------------------------------------------------------------------
# linear activation function
class Linear(ActivationBase):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs.copy()

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs


# -------------------------------------------------------------------------------------------------
# ReLU activation function
class ReLU(ActivationBase):

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


# -------------------------------------------------------------------------------------------------
# TanH activation function
# https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
class TanH(ActivationBase):
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.tanh(inputs)

    #https://discuss.pytorch.org/t/what-is-pytorchs-backwards-function-for-tanh/119389/4
    def backward(self, dvalues):
        # f'(x) = 1-f(x)^2
        self.dinputs = (1.0 - np.tanh(self.inputs) ** 2.0) * dvalues

    def predictions(self, outputs):
        return outputs

# -------------------------------------------------------------------------------------------------
# Sigmoid activation function
class Sigmoid(ActivationBase):

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
