import numpy as np
from LossFunctions.BaseLoss import BaseLoss

# Mean Squared Error loss
class Mse(BaseLoss):  # L2 loss

    # Forward pass .
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
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
