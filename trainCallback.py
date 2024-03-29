import os
import pickle

#--------------------------------------------------------------------------------
# base class for callback
class TrainCallback:
    def __init__(self):
        self.model = None
        pass

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass


#--------------------------------------------------------------------------------
# callback to save best checkpoint model (default is val_loss based)
class BestCheckPoint(TrainCallback):
    def __init__(self, filename, monitor="val_loss"):
        super().__init__()
        self.__monitor = monitor
        self.__minMonitorValue = None
        self.__epochMinValue = None
        self.__filenameModel = filename
        self.__filenameCheckpoint = filename+".checkpoint"
        if os.path.exists(self.__filenameCheckpoint):
            self.__loadcheckpoint()
        else:
            self.__savecheckpoint()

    def getParams(self):
        prms = dict()
        prms["monitor"] = self.__monitor
        prms["minMonitorValue"] = self.__minMonitorValue
        prms["epochMinValue"] = self.__epochMinValue
        return prms


    def __loadcheckpoint(self):
        with open(self.__filenameCheckpoint, "rb") as filehandler:
            prms = pickle.load(filehandler)
            if prms["monitor"] != self.__monitor:
                raise Exception("Parametro non coincidente")
            self.__minMonitorValue = prms["minMonitorValue"]
            self.__epochMinValue = prms["epochMinValue"]

    def __savecheckpoint(self):
        with open(self.__filenameCheckpoint, "wb") as filehandler:
            prms = dict()
            prms["monitor"] = self.__monitor
            prms["minMonitorValue"] = self.__minMonitorValue
            prms["epochMinValue"] = self.__epochMinValue
            pickle.dump(prms,filehandler)

    def on_train_begin(self, logs=None):
        if os.path.exists(self.__filenameCheckpoint):
            self.__loadcheckpoint()
        else:
            self.__savecheckpoint()

    def on_epoch_end(self, epoch, logs=None):
        actualVal = self.model.parameters[self.__monitor]
        if (self.__minMonitorValue is None) or (actualVal < self.__minMonitorValue):
            self.__minMonitorValue = self.model.parameters[self.__monitor]
            self.__epochMinValue = epoch
            self.__saveModel()

    def __saveModel(self):
        self.model.save_weights(self.__filenameModel)
        self.__savecheckpoint()
        pass

#--------------------------------------------------------------------------------
# callback to manage Early stopping (default is val_loss based)
class EarlyStopping(TrainCallback):
    def __init__(self, patience=0, monitor="val_loss"):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.minMonitorValue = None
        self.epochMinValue = None

    def getMinDataLoss(self):
        return self.minMonitorValue

    def getEpochMinValue(self):
        return self.epochMinValue

    def on_train_end(self, batch, logs=None):
        print(f"Best epoch: {self.epochMinValue}  - {self.monitor}: {self.minMonitorValue}")
        pass

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 1:
            self.minMonitorValue = self.model.parameters[self.monitor]
            self.epochMinValue = epoch
        else:
            actualVal = self.model.parameters[self.monitor]
            if actualVal < self.minMonitorValue:
                self.minMonitorValue = actualVal
                self.epochMinValue = epoch
            elif self.epochMinValue + self.patience < epoch:
                self.model.stopTraining()

        '''
        tenere il min del parametro (pmin) ed il numero dell'epoca (emin)
        se il valore corrente è >= emin al minimo e epoch >= emin+pazienza fare stop
        '''
        pass



