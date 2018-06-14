from keras import callbacks
from util import hdf5_util
import time


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.file_path, overwrite=True)
        time.sleep(0.01)
        hdf5_util.set_property(self.file_path, 'epochs_trained', epoch + 1)
