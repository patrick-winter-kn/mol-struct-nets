from keras import callbacks
from util import hdf5_util


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        hdf5_util.set_property(self.file_path, 'epochs_trained', epoch + 1)
