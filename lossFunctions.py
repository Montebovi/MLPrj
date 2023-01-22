import numpy as np

# ------------------------------------------------------------------------------
# loss base class
class LossBase:
    def __init__(self):
        self.accumulated_sum = None
        self.accumulated_count = None

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def forward(self, y_pred, y_true):
        pass

    def backward(self, dvalues, y_true):
        pass

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, layers):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        if isinstance(sample_losses, list):
            self.accumulated_count += len(sample_losses)
        else:
            self.accumulated_count += 1

        regularization_loss = 0
        for layer in layers:
            if layer.isTrainable():
                regularization_loss += layer.regularization_loss()
        return data_loss, regularization_loss

    # Calculates accumulated loss
    def calculate_accumulated(self, layers):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        regularization_loss = 0
        for layer in layers:
            if layer.isTrainable():
                regularization_loss += layer.regularization_loss()

        # Return the data and regularization losses
        return data_loss, regularization_loss

    @staticmethod
    def GetLossByName(lossName):
        if lossName.lower() == "bce": #Binary cross-entropy loss
            return BinaryCrossEntropy()
        elif lossName.lower() == "mse":
            return MeanSquaredError()
        elif lossName.lower() == "mee":
            return MeanEuclideanError()
        else:
            raise NotImplementedError("Unexpected loss function")


# ------------------------------------------------------------------------------
# Mean Squared Error loss
class MeanSquaredError(LossBase):  # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)

        data_loss = np.mean(sample_losses)
        # Return losses
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        # self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = -2 * (y_true - dvalues)
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# ------------------------------------------------------------------------------
# Mean Euclidean Error loss
class MeanEuclideanError(LossBase):
    def forward(self, y_pred, y_true):
        sample_losses = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=-1))
        data_loss = np.mean(sample_losses)
        return data_loss

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        denom = np.sqrt(np.sum((y_true - dvalues) ** 2, axis=-1))
        self.dinputs = -(y_true - dvalues)
        for i in range(0, samples):
            self.dinputs[i] = self.dinputs[i] / denom[i]
        self.dinputs = self.dinputs / samples


# ------------------------------------------------------------------------------
# Binary cross-entropy loss
class BinaryCrossEntropy(LossBase):

    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples
