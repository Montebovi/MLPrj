from LossFunctions.Mse import Mse

class LossMapper:
    @staticmethod
    def GetByName(name):
        match name:
            case 'mse':
                print("Here")
                return Mse()
            case other:
                raise NotImplementedError("Unexpected loss function")