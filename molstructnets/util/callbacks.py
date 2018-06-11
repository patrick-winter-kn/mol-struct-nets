from keras import callbacks
from util import hdf5_util, file_util
import time


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.file_path, overwrite=True)
        while not file_util.file_exists(self.file_path):
            time.sleep(0.001)
        hdf5_util.set_property(self.file_path, 'epochs_trained', epoch + 1)
