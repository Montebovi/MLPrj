import numpy as np

class BaseLoss:
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
        if lossName.lower() == "mse":
            return Mse()
        else:
            raise NotImplementedError("Unexpected loss function")